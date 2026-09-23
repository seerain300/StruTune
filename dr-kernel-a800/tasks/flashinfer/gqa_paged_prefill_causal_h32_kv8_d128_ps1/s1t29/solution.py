import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_out_single_kernel(
    q_row_ptr,        # *float32, shape [HEAD_DIM]
    k_ptr,            # *float32, shape [MAX_KV, HEAD_DIM]
    v_ptr,            # *float32, shape [MAX_KV, HEAD_DIM]
    out_ptr,          # *float32, shape [HEAD_DIM]
    sm_scale,         # float32 scalar
    HEAD_DIM: tl.constexpr,
    MAX_KV: tl.constexpr,
    h: tl.constexpr,  # query head index (for signature; not used in kernel but can be specialized)
):
    # We perform vectorized operations across the head_dim (compile-time constant).
    # Each thread will write to out[i] = sum_m softmax(q_row[i] * k_ptr[m, i]) * v_ptr[m, i]
    # Note: Triton requires no Python loops/conditionals in the kernel. This kernel uses only elementwise ops.
    for i in range(HEAD_DIM):
        # q_row[i]
        q_val = tl.load(q_row_ptr + i)
        total = 0.0  # scalar accumulator
        # Iterate over m in [0, MAX_KV)
        for m in range(MAX_KV):
            # k_ptr[m, i]
            k_val = tl.load(k_ptr + m * HEAD_DIM + i)
            # v_ptr[m, i]
            v_val = tl.load(v_ptr + m * HEAD_DIM + i)
            # scaled dot for this position
            scaled = q_val * k_val * sm_scale
            # softmax over max_kv rows: compute numerator and sum
            # softmax(scaled) = exp(scaled) / sum_j exp(scaled_j)
            # We do this by building the numerator and denominator per i across m
            # Note: Triton allows scalar operations; we implement via per-thread accumulation.
            # For simplicity, we assume MAX_KV is small (<= 128), and use scalar accumulation.
            # Compute numerator
            num = tl.exp(scaled)
            # Compute denominator: sum over all m
            # We need to loop again or compute pairwise. Here we recompute per m using scalar accumulation.
            # Initialize denom = 0.0, then add num for each m
            denom = 0.0
            for mm in range(MAX_KV):
                kk = tl.load(k_ptr + mm * HEAD_DIM + i)
                vv = tl.load(v_ptr + mm * HEAD_DIM + i)
                s_mm = q_val * kk * sm_scale
                denom += tl.exp(s_mm)
            # softmax contribution
            attn = num / denom
            total += attn * v_val
        # store result
        tl.store(out_ptr + i, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move tensors to CUDA for Triton execution
        device = torch.device("cuda")
        q = q.to(device).contiguous()
        k_cache = k_cache.to(device).contiguous()
        v_cache = v_cache.to(device).contiguous()
        qo_indptr = qo_indptr.to(device).contiguous()
        kv_indptr = kv_indptr.to(device).contiguous()
        kv_indices = kv_indices.to(device).contiguous()

        # Extract constants
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        # The original asserts: num_qo_heads == 32, num_kv_heads == 8, head_dim == 128
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"

        len_indptr = qo_indptr.shape[0]
        assert len_indptr == 2, "This implementation supports len_indptr == 2; adjust if needed"

        # Process first interval (batch 0): qo_start, qo_end = qo_indptr[0], qo_indptr[1]
        qo_start = int(qo_indptr[0].item())
        qo_end = int(qo_indptr[1].item())
        assert total_q == qo_end, "total_q must equal qo_indptr[-1]"

        kv_start = int(kv_indptr[0].item())
        kv_end = int(kv_indptr[1].item())

        # Flatten k/v since squeezed dimension is 1
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, 8, 128]

        # Gather kv indices for this interval
        kv_indices_batch = kv_indices[kv_start:kv_end]  # [num_kv_tokens]
        num_kv_tokens = kv_indices_batch.shape[0]

        # Load q_batch as float32 for computation
        q_f32 = q.to(torch.float32)  # [total_q, 32, 128]
        # K and V batch: shape [num_kv_tokens, 8, 128]
        k_batch = k_cache_flat[kv_indices_batch]  # [num_kv_tokens, 8, 128], dtype float32
        v_batch = v_cache_flat[kv_indices_batch]  # [num_kv_tokens, 8, 128], dtype float32

        # Number of queries in this interval
        num_q_tokens = qo_end - qo_start
        assert num_q_tokens >= 0, "Invalid qo interval"

        # Output and LSE (float32 for stability)
        output = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device) - float("inf")

        # Compute per (q_idx, h) output using Triton kernel
        # We avoid loops and conditionals in the Triton kernel. Host-side uses simple Python loops to call it.
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        for q_idx in range(num_q_tokens):
            global_q_idx = q_idx + qo_start
            q_row = q_f32[global_q_idx]  # [32, 128]
            # Compute delta for causal-like masking
            delta = num_kv_tokens - num_q_tokens  # can be negative
            max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv_idx <= 0:
                # Skip as original code would
                continue
            # For GQA, map query head to KV head
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # h // 4
                # Prepare K and V rows for this head
                # k_rows: [max_kv_idx, 128], v_rows: [max_kv_idx, 128]
                k_rows = k_batch[:max_kv_idx, kv_head]  # [max_kv_idx, 128]
                v_rows = v_batch[:max_kv_idx, kv_head]  # [max_kv_idx, 128]
                # Triton kernel expects q_row[h] -> take only the head h slice
                q_row_h = q_row[h]  # [128] float32
                # Allocate output vector
                out_vec = torch.empty(head_dim, dtype=torch.float32, device=device)
                # Launch Triton kernel: we pass pointers and scalars; no loops/conditionals in kernel
                compute_out_single_kernel[(1,)](
                    q_row_h,                    # *float32, [128]
                    k_rows.view(-1, head_dim), # *float32, [MAX_KV, 128]
                    v_rows.view(-1, head_dim), # *float32, [MAX_KV, 128]
                    out_vec,                    # *float32, [128]
                    sm_scale,                   # float32 scalar
                    HEAD_DIM=head_dim,          # constexpr
                    MAX_KV=max_kv_idx,          # constexpr
                    h=h,                        # constexpr (ignored in kernel)
                )
                # Store output and lse
                output[q_idx, h] = out_vec  # float32, will be cast later to bfloat16
                # lse for this (query, head): logsumexp of scaled logits
                # We compute logits per m as q_row[h] * k_rows[m] * sm_scale and reduce.
                # To avoid dynamic loops here too, we compute using PyTorch reductions for correctness.
                # Compute logits as a vector
                logits = (q_row_h * k_rows.squeeze(1)).view(-1) * sm_scale  # [MAX_KV]
                # LSE base-2
                lse_base2 = torch.logsumexp(logits, dim=0) / math.log(2.0)
                lse[q_idx, h] = lse_base2

        # Cast output to bfloat16 to match original run signature
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
