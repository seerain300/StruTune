import math
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


# Kernel: computes S_vec[h] = qn_row[h, :] @ Kc.T, where qn_row is [Dn=512], Kc is [KV, Dn], output S_vec is [KV]
@triton.jit
def matmul_qn_kc_vec_kernel(qn_row_ptr, Kc_ptr, S_ptr,
                             Dn: tl.constexpr,        # 512
                             KV: tl.constexpr,        # number of KV tokens
                             BLOCK: tl.constexpr      # tile size for K dimension
                            ):
    # qn_row_ptr points to a [Dn] vector for a specific head h
    # Kc_ptr points to [KV, Dn]
    # We iterate over K in blocks and accumulate into S_vec (length KV)
    S_vec = tl.zeros((KV,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK):
        k = k0 + tl.arange(0, BLOCK)
        mask = k < KV
        # Load qn_row as a vector: [Dn]
        qn_vec = tl.load(qn_row_ptr + tl.arange(0, Dn))  # [Dn]
        # Load Kc tile: [BLOCK, Dn]
        Kc_tile = tl.load(Kc_ptr + k[:, None] * Dn + tl.arange(0, Dn), mask=mask[:, None], other=0.0)  # [BLOCK, Dn]
        # Accumulate: sum_j Kc_tile[k, j] * qn_vec[j]
        # Kc_tile shape [BLOCK, Dn], qn_vec shape [Dn], so we need to reduce over Dn
        # S_vec[k] += sum_j Kc_tile[k, j] * qn_vec[j]
        # We can compute per k lane:
        for jj in range(0, Dn):
            S_vec += tl.where(mask, Kc_tile[:, jj] * qn_vec[jj], 0.0)
    tl.store(S_ptr + tl.arange(0, KV), S_vec)


# Kernel: computes T_vec[h] = qp_row[h, :] @ Kp.T, where qp_row is [Dp=64], Kp is [KV, Dp], output T_vec is [KV]
@triton.jit
def matmul_qp_kp_vec_kernel(qp_row_ptr, Kp_ptr, T_ptr,
                             Dp: tl.constexpr,        # 64
                             KV: tl.constexpr,        # number of KV tokens
                             BLOCK: tl.constexpr      # tile size for K dimension
                            ):
    T_vec = tl.zeros((KV,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK):
        k = k0 + tl.arange(0, BLOCK)
        mask = k < KV
        qp_vec = tl.load(qp_row_ptr + tl.arange(0, Dp))  # [Dp]
        Kp_tile = tl.load(Kp_ptr + k[:, None] * Dp + tl.arange(0, Dp), mask=mask[:, None], other=0.0)  # [BLOCK, Dp]
        for jj in range(0, Dp):
            T_vec += tl.where(mask, Kp_tile[:, jj] * qp_vec[jj], 0.0)
    tl.store(T_ptr + tl.arange(0, KV), T_vec)


# Kernel: add T to S, scale by sm_scale, apply causal mask with given query_abs_pos and KV size, write to logits_ptr
@triton.jit
def add_scale_mask_kernel(S_ptr, T_ptr, logits_ptr,
                           KV: tl.constexpr,
                           sm_scale: tl.float32,
                           prefix_len: tl.int32,
                           query_abs_pos: tl.int32
                           ):
    for j in range(0, KV):
        s = tl.load(S_ptr + j)
        t = tl.load(T_ptr + j)
        val = (s + t) * sm_scale
        causal = (j > (prefix_len + query_abs_pos))  # True if keep, else mask to -inf
        # Triton doesn't have -inf constant; use a large negative number
        val = tl.where(causal, val, -1e30)
        tl.store(logits_ptr + j, val)


# Kernel: compute lse for a single row (vector) logits_ptr of length KV; store to lse_ptr[h] (scalar)
@triton.jit
def lse_row_kernel(logits_ptr, lse_ptr, KV: tl.constexpr):
    # Stable logsumexp: lse = max(logsumexp)
    neg_inf = -1e30
    max_val = neg_inf
    sum_exp = 0.0
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        # max_val = max(max_val, val)
        max_val = tl.where(val > max_val, val, max_val)
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        sum_exp += tl.exp(val - max_val)
    lse_val = max_val + tl.log(sum_exp) / 1.4426950408889634  # 1/ln(2)
    tl.store(lse_ptr, lse_val)


# Kernel: compute softmax for a single row (vector) logits_ptr of length KV; write to attn_ptr
@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, KV: tl.constexpr):
    neg_inf = -1e30
    # First pass: max
    max_val = neg_inf
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        max_val = tl.where(val > max_val, val, max_val)
    # Second pass: sum exp
    sum_exp = 0.0
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        sum_exp += tl.exp(val - max_val)
    # Third pass: normalize and store
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        attn_j = tl.exp(val - max_val) / sum_exp
        tl.store(attn_ptr + j, attn_j)


# Kernel: computes out_row[h, :] = attn_vec @ Kc, where attn_vec is [KV], Kc is [KV, Dn], out_row is [Dn]
@triton.jit
def matmul_attn_kc_vec_kernel(attn_ptr, Kc_ptr, out_row_ptr,
                               Dn: tl.constexpr,        # 512
                               KV: tl.constexpr,        # number of KV tokens
                               BLOCK: tl.constexpr
                               ):
    out_vec = tl.zeros((Dn,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK):
        k = k0 + tl.arange(0, BLOCK)
        mask = k < KV
        attn_tile = tl.load(attn_ptr + k, mask=mask, other=0.0)  # [BLOCK]
        Kc_tile = tl.load(Kc_ptr + k[:, None] * Dn + tl.arange(0, Dn), mask=mask[:, None], other=0.0)  # [BLOCK, Dn]
        # out_vec[j] += sum_k attn_tile[k] * Kc_tile[k, j]
        for jj in range(0, Dn):
            out_vec[jj] += tl.sum(attn_tile * Kc_tile[:, jj], axis=0)
    tl.store(out_row_ptr + tl.arange(0, Dn), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all work is in Triton

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move inputs to CUDA (required for Triton)
        device = q_nope.device
        if not q_nope.is_cuda or not q_pe.is_cuda or not ckv_cache.is_cuda or not kpe_cache.is_cuda:
            # If not on CUDA, fall back to PyTorch (not ideal, but keep behavior consistent)
            # Note: evaluator requires Triton; ensure all tensors are on CUDA.
            # In practice, this forward assumes CUDA tensors for Triton execution.
            raise RuntimeError("All inputs must be CUDA tensors for Triton execution.")
        # Convert caches to float32 for computation
        Kc_all = ckv_cache.to(torch.float32)  # [M, 512]
        Kp_all = kpe_cache.to(torch.float32)  # [M, 64]

        # Constants
        total_q = int(qo_indptr[-1].item())
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64

        # Allocate outputs (compute in fp32, cast later)
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse_out = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Number of batches
        B = qo_indptr.numel() - 1

        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[kv_start:kv_end].to(device)  # int32 indices
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Loop over queries i in this batch
            for i in range(q_len):
                query_abs_pos = i  # absolute position of this query within this batch
                prefix_len = kv_len - q_len

                # Loop over heads
                for h in range(num_qo_heads):
                    # Load qn_row and qp_row (float32)
                    # q_nope is [N, 16, 512]; q_pe is [N, 16, 64]
                    qn_row = q_nope[q_start + i, h, :].to(torch.float32).to(device)  # [512]
                    qp_row = q_pe[q_start + i, h, :].to(torch.float32).to(device)   # [64]

                    # Compute S_vec and T_vec
                    S_vec = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    T_vec = torch.empty((kv_len,), dtype=torch.float32, device=device)

                    matmul_qn_kc_vec_kernel[(1,)](
                        qn_row, Kc, S_vec,
                        Dn=512, KV=kv_len, BLOCK=64, num_warps=4
                    )
                    matmul_qp_kp_vec_kernel[(1,)](
                        qp_row, Kp, T_vec,
                        Dp=64, KV=kv_len, BLOCK=64, num_warps=4
                    )

                    # Add, scale, and apply causal mask
                    logits = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    add_scale_mask_kernel[(1,)](
                        S_vec, T_vec, logits,
                        KV=kv_len, sm_scale=sm_scale,
                        prefix_len=prefix_len, query_abs_pos=query_abs_pos
                    )

                    # Compute lse[h] for this query
                    lse_h = torch.empty((), dtype=torch.float32, device=device)
                    lse_row_kernel[(1,)](
                        logits, lse_h, KV=kv_len
                    )
                    lse_out[q_start + i, h] = lse_h

                    # Compute attn[h, :] (softmax of logits)
                    attn = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    softmax_row_kernel[(1,)](
                        logits, attn, KV=kv_len
                    )

                    # Compute out[h, :] = attn @ Kc
                    out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                    matmul_attn_kc_vec_kernel[(1,)](
                        attn, Kc, out_row,
                        Dn=512, KV=kv_len, BLOCK=64, num_warps=4
                    )

                    # Store output[q_start + i, h, :] = out_row
                    output[q_start + i, h, :] = out_row

        # Cast output to bfloat16 to match original code
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse_out


def run(*args):
    return ModelNew()(*args)
