import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_all_segments_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: one program per (i, h) across all segments.
    - q_ptr: float32 [total_q, 32, 128]
    - k_ptr: float32 [total_kv, 8, 128]
    - v_ptr: float32 [total_kv, 8, 128]
    - out_ptr: float32 [total_q, 32, 128] (output to be written by kernel)
    - lse_ptr: float32 [total_q, 32] (lse per (i, h))
    - qo_indptr_ptr, kv_indptr_ptr: int32 [NUM_SEGMENTS+1]
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    # Process each segment b
    for b in tl.static_range(NUM_SEGMENTS):
        q_start = tl.load(qo_indptr_ptr + b)     # int32
        q_end = tl.load(qo_indptr_ptr + b + 1)  # int32
        kv_start = tl.load(kv_indptr_ptr + b)   # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1) # int32

        Nq = q_end - q_start
        Nk = kv_end - kv_start

        if Nq == 0 or Nk == 0:
            continue

        delta = Nk - Nq  # segment-specific delta

        # Prepare logits vector for j in 0..7
        logits = tl.zeros((8,), dtype=tl.float32)

        # Load q[i, h, :]
        q_base = (q_start + i) * 32 * 128 + h * 128
        q_vec = tl.load(q_ptr + q_base)  # [128] float32

        # Accumulate dot-products for each j across Nk rows of k
        for row in tl.static_range(Nk):
            for j in tl.static_range(8):
                # k_expanded for this segment has 32 heads; map h to original kv head orig_h = h % 8
                orig_h = h % 8
                k_off = kv_start + row
                k_vec = tl.load(k_ptr + k_off * 8 * 128 + j * 128 + orig_h * 128)  # [128] float32
                dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale  # scalar
                # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
                if j >= (i + 1 + delta):
                    dot = -float("inf")
                logits[j] = dot

        # Compute base-2 logsumexp over the 8 positions
        m = tl.max(logits, axis=0)
        sum_exp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

        # Softmax across 8 positions
        soft = tl.exp(logits - lse_val)  # [8], float32

        # Accumulate output[i, h, :] = sum_j soft[j] * v_exp[kv_start + row, j, h%8]
        for row in tl.static_range(Nk):
            for j in tl.static_range(8):
                orig_h = h % 8
                v_off = kv_start + row
                v_vec = tl.load(v_ptr + v_off * 8 * 128 + j * 128 + orig_h * 128)  # [128] float32
                out_base = (q_start + i) * 32 * 128 + h * 128
                tl.atomic_add(out_ptr + out_base, v_vec * soft[j])

        # Store lse[i, h]
        lse_idx = i * 32 + h
        tl.store(lse_ptr + lse_idx, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized forward: all computation inside Triton kernels.
        Returns: output [total_q, 32, 128] bfloat16, lse [total_q, 32] float32 (base-2 logsumexp)
        """
        # Cast to float32 for compute; make contiguous
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]
        num_qo_heads = q_f32.shape[1]  # 32
        num_kv_heads = k_f32.shape[1]  # 8

        # Output and lse buffers
        output = torch.zeros(
            (total_q, num_qo_heads, 128),
            dtype=torch.float32,
            device=q.device
        )
        lse = torch.full(
            (total_q, num_qo_heads),
            -float("inf"),
            dtype=torch.float32,
            device=q.device
        )

        NUM_SEGMENTS = qo_indptr.numel() - 1
        grid = (total_q * num_qo_heads,)

        attention_gqa_all_segments_kernel[grid](
            q_f32, k_f32, v_f32, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
        )

        # Convert output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# Helper functions for testing:
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)], dim=0).to(torch.int32).to(device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)], dim=0).to(torch.int32).to(device='cuda')
    sm_scale = 1.0 / math.sqrt(128)
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
