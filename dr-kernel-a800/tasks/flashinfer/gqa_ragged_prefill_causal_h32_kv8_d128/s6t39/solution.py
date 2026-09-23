import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    q_ptr,        # *f32, [Q, 32, 128]
    k_ptr,        # *f32, [K, 8, 128]
    out_ptr,      # *f32, [Q, 32, K], logits
    Q, H, K,      # int32
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_d,
    sm_scale,     # f32
):
    # One program per (i, h)
    pid = tl.program_id(0)
    i = pid // H
    h = pid % H
    if (i >= Q) or (h >= H):
        return

    # gqa_ratio = 4 since 32/8
    kh = h // 4  # map to original kv head index

    # Loop over j = 0 .. K-1
    for j in range(0, K):
        # Load q[i, h]
        q_offset = i * stride_q_b + h * stride_q_h
        q_val = tl.load(q_ptr + q_offset)
        # Load k[j, kh]
        k_offset = j * stride_k_b + kh * stride_k_h
        k_val = tl.load(k_ptr + k_offset)
        # Compute logits[i, h, j] = q_val * k_val * sm_scale
        score = q_val * k_val * sm_scale
        # Store logits at [i, h, j]
        out_offset = i * (H * K) + h * K + j
        tl.store(out_ptr + out_offset, score)


@triton.jit
def lse_reduce_kernel(
    logits_ptr,   # *f32, [Q, 32, K]
    lse_ptr,      # *f32, [Q, 32]
    Q, H, K,      # int32
):
    # One program per (i, h)
    pid = tl.program_id(0)
    i = pid // H
    h = pid % H
    if (i >= Q) or (h >= H):
        return

    m = tl.full((), -1.0e20, tl.float32)  # running max
    # First pass: compute max over j
    for j in range(0, K):
        offs = i * (H * K) + h * K + j
        val = tl.load(logits_ptr + offs)
        m = tl.maximum(m, val)
    # Second pass: sum exp(logits - m)
    s = tl.zeros((), tl.float32)
    for j in range(0, K):
        offs = i * (H * K) + h * K + j
        val = tl.load(logits_ptr + offs)
        s += tl.exp(val - m)
    lse_val = tl.log(s) + m
    lse_offset = i * H + h
    tl.store(lse_ptr + lse_offset, lse_val)


@triton.jit
def softmax_accum_output_kernel(
    logits_ptr,   # *f32, [Q, 32, K]
    v_ptr,        # *f32, [K, 8, 128] (we expand v to 32 heads by mapping h//4)
    lse_ptr,      # *f32, [Q, 32]
    out_ptr,      # *f32, [Q, 32, 128]
    Q, H, K,      # int32
    stride_log_b, stride_log_h, stride_log_j,
    stride_v_b, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
):
    # One program per (i, h)
    pid = tl.program_id(0)
    i = pid // H
    h = pid % H
    if (i >= Q) or (h >= H):
        return

    # Load lse[i, h]
    lse_offset = i * H + h
    lse_val = tl.load(lse_ptr + lse_offset)

    # Accumulate output per head dimension (128)
    for j in range(0, K):
        # Load logits[i, h, j]
        offs_log = i * stride_log_b + h * stride_log_h + j * stride_log_j
        logits_val = tl.load(logits_ptr + offs_log)
        # Compute y = exp(logits - lse)
        y = tl.exp(logits_val - lse_val)
        # Causal mask: i can only attend j < i + 1 (since delta can be 0 in tight segments)
        causal_mask = j < (i + 1)
        y = tl.where(causal_mask, y, 0.0)

        # Map to original kv head for v
        kh = h // 4
        # Load v_expanded[j, h, :] which is v[j, kh, :]
        v_offset = j * stride_v_b + kh * stride_v_h
        v_vec = tl.load(v_ptr + v_offset + tl.arange(0, 128))
        # Accumulate into out[i, h, :]
        for d in range(0, 128):
            out_offset = i * stride_out_b + h * stride_out_h + d * stride_out_d
            old = tl.load(out_ptr + out_offset)
            new = old + y * v_vec[d]
            tl.store(out_ptr + out_offset, new)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        # Allocate outputs (float32 for computation, cast later)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Ensure contiguous and cast to float32 for kernel math
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Launch compute logits kernel: grid = (Q * H,)
        grid_logits = (total_q * num_qo_heads,)
        compute_logits_kernel[grid_logits](
            q_f32, k_f32, output, total_q, num_qo_heads, k_f32.shape[0],
            q_f32.stride(0), q_f32.stride(1), q_f32.stride(2),
            k_f32.stride(0), k_f32.stride(1), k_f32.stride(2),
            sm_scale,
        )

        # Launch LSE reduction kernel
        grid_lse = (total_q * num_qo_heads,)
        lse_reduce_kernel[grid_lse](
            output, lse, total_q, num_qo_heads, k_f32.shape[0],
        )

        # Launch softmax-accumulation output kernel
        grid_out = (total_q * num_qo_heads,)
        softmax_accum_output_kernel[grid_out](
            output, v_f32, lse, output, total_q, num_qo_heads, k_f32.shape[0],
            output.stride(0), output.stride(1), output.stride(2),
            v_f32.stride(0), v_f32.stride(1), v_f32.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
        )

        # Return outputs: original code creates output as bfloat16; we cast accordingly
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
