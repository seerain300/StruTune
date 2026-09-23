import math
import torch
import triton
import triton.language as tl


@triton.jit
def per_query_attention_gqa_kernel(
    q_ptr,          # *float32, [total_q, 32, 128]
    k_exp_ptr,      # *float32, [total_kv, 32, 128]
    v_exp_ptr,      # *float32, [total_kv, 32, 128]
    out_ptr,        # *float32, [total_q, 32, 128]
    lse_ptr,        # *float32, [total_q, 32]
    qo_indptr_ptr,  # *int32, [len_indptr]
    kv_indptr_ptr,  # *int32, [len_indptr]
    sm_scale,       # float32
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: one program per (i, h). Iterates over segments, computes logits, lse, and output.
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    for b in range(NUM_SEGMENTS):
        q_start = tl.load(qo_indptr_ptr + b)  # int32
        q_end = tl.load(qo_indptr_ptr + b + 1)  # int32
        kv_start = tl.load(kv_indptr_ptr + b)  # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1)  # int32

        Nq = q_end - q_start
        Nk = kv_end - kv_start

        if Nq == 0 or Nk == 0:
            continue

        delta = Nk - Nq  # per-segment delta for causal mask

        # Load q[i, h, :] vector across 128 dims
        q_base = (q_start + i) * 32 * 128 + h * 128
        q_vec = tl.load(q_ptr + q_base)  # [128] float32

        # Compute 8 logits for j in 0..7
        logits_j = tl.zeros((8,), dtype=tl.float32)
        for j in tl.static_range(8):
            dot_acc = 0.0
            r = kv_start
            while r < kv_end:
                # Address for k_exp[r, j, :] is r*32*128 + j*128 + d
                k_base = r * 32 * 128 + j * 128
                k_vec = tl.load(k_exp_ptr + k_base)  # [128] float32
                dot_acc += tl.sum(q_vec * k_vec)
                r += 1
            logits_j[j] = dot_acc * sm_scale
            # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
            if j >= (i + 1 + delta):
                logits_j[j] = -float("inf")

        # Base-2 logsumexp over 8 positions
        m = tl.max(logits_j, axis=0)
        sum_exp = tl.sum(tl.exp(logits_j - m), axis=0)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

        # Softmax across 8 positions
        soft = tl.exp(logits_j - lse_val)  # [8] float32

        # Compute and store output[i, h, d] = sum_j soft[j] * sum_r v_exp[r, j, d]
        for d in tl.static_range(128):
            contrib = 0.0
            for j in tl.static_range(8):
                acc_j = 0.0
                r = kv_start
                while r < kv_end:
                    v_base = r * 32 * 128 + j * 128 + d
                    v_elem = tl.load(v_exp_ptr + v_base)  # scalar float32
                    acc_j += v_elem
                    r += 1
                contrib += soft[j] * acc_j
            out_base = (q_start + i) * 32 * 128 + h * 128 + d
            tl.store(out_ptr + out_base, contrib)

        # Store lse[i, h]
        lse_index = (q_start + i) * 32 + h
        tl.store(lse_ptr + lse_index, lse_val)

class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Returns:
          - output: [total_q, 32, 128], dtype bfloat16
          - lse: [total_q, 32], dtype float32 (base-2 logsumexp)
        """
        # Ensure inputs are on CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda

        # Shapes and assertions
        total_q = q.shape[0]
        total_kv = k.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[2]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Pre-expand k and v to 32 heads (GQA mapping)
        k_exp_f32 = torch.repeat_interleave(k_f32, repeats=4, dim=1).contiguous()
        v_exp_f32 = torch.repeat_interleave(v_f32, repeats=4, dim=1).contiguous()

        # Prepare output and lse
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        NUM_SEGMENTS = qo_indptr.numel() - 1
        # Launch Triton kernel: one program per (i, h)
        grid = (total_q * 32,)
        per_query_attention_gqa_kernel[grid](
            q_f32, k_exp_f32, v_exp_f32, output, lse,
            qo_indptr, kv_indptr,
            sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# The original get_inputs and fused_operator helper functions can be reused; only ModelNew is required for evaluation.


def run(*args):
    return ModelNew()(*args)
