import torch
import math

import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,        # *float32, [1, K] (we pass [1, K] pointer and load as [K])
    B_ptr,        # *float32, [M, K]
    C_ptr,        # *float32, [1, M]
    K: tl.constexpr,     # int: reduction dimension (e.g., 512 or 64)
    M,                     # int: number of rows in B (runtime)
    BLOCK_K: tl.constexpr  # tile size along K (e.g., 128 or 64)
):
    # Output vector [M]
    out_vec = tl.zeros((M,), dtype=tl.float32)
    # Iterate over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_idx < K
        # Load A chunk (single row vector of length K)
        a_chunk = tl.load(A_ptr + k_idx, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Accumulate dot product with each row of B
        for m in range(0, M):
            b_row = tl.load(B_ptr + m * K + k_idx, mask=mask_k, other=0.0)  # [BLOCK_K]
            out_vec[m] += tl.sum(a_chunk * b_row, axis=0)
    # Write out to C[0, :]
    tl.store(C_ptr + tl.arange(0, M), out_vec, mask=tl.arange(0, M) < M)


@triton.jit
def softmax_lse_kernel(
    x_ptr,          # *float32, [M] (logits_scaled)
    out_lse_ptr,    # *float32, [1] (lse per head)
    attn_ptr,       # *float32, [M] (attn vector per head)
    M: tl.constexpr,     # int: length of x (compile-time specialized per batch)
    sm_scale,            # float32 scalar
    inv_ln2,             # float32 = 1.0 / ln(2)
):
    # Vector of indices
    idx = tl.arange(0, M)
    x = tl.load(x_ptr + idx)  # [M]
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    x_shifted = x - max_val
    # Scale
    x_scaled = x_shifted * sm_scale
    # Compute sum(exp(x_scaled))
    exp_x = tl.exp(x_scaled)
    sum_exp = tl.sum(exp_x, axis=0)
    # lse = log(sum_exp) / ln(2)
    lse_val = tl.log(sum_exp) * inv_ln2
    # Store lse
    tl.store(out_lse_ptr + tl.arange(0, 1), lse_val)
    # Compute attn = exp(x_scaled) / sum_exp
    attn = exp_x / sum_exp
    # Store attn
    tl.store(attn_ptr + idx, attn, mask=idx < M)


@triton.jit
def matvec_row_out_kernel(
    A_ptr,          # *float32, [M] (attention vector)
    B_ptr,          # *float32, [M, Kc_dim]
    C_ptr,          # *float32, [1, Kc_dim]
    M: tl.constexpr,       # int: length of A (compile-time specialized per batch)
    Kc_dim: tl.constexpr,  # int: reduction dimension (e.g., 512)
    BLOCK_M: tl.constexpr  # tile size along M (e.g., 256)
):
    out_vec = tl.zeros((Kc_dim,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_m = m_idx < M
        a_chunk = tl.load(A_ptr + m_idx, mask=mask_m, other=0.0)  # [BLOCK_M]
        for k_start in range(0, Kc_dim, 64):
            k_idx = k_start + tl.arange(0, 64)  # [64], 64 is a good tile for 512
            mask_k = k_idx < Kc_dim
            # Load B tile: shape [BLOCK_M, 64]
            b_tile = tl.load(B_ptr + m_idx[:, None] * Kc_dim + k_idx[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            # Accumulate dot per k: out_vec[k] += sum_m a_chunk[m] * b_tile[m, :]
            for kk in range(64):
                col = b_tile[:, kk]  # [BLOCK_M]
                out_vec[k_start + kk] += tl.sum(a_chunk * col, axis=0)
    tl.store(C_ptr + tl.arange(0, Kc_dim), out_vec, mask=tl.arange(0, Kc_dim) < Kc_dim)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Dimensions
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Kc_dim = q_nope.shape[2]  # 512
        # Squeeze caches (original asserts second dim is 1)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        # Prepare outputs
        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Precompute constants
        inv_ln2 = 1.0 / math.log(2.0)

        for b in range(B):
            # Compute token range for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                # No tokens, output zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[start:end]  # [M]
            Kc = Kc_all[tok_idx]             # [M, 512]
            Kp = Kp_all[tok_idx]             # [M, 64]

            # Prepare q vectors per head
            for h in range(H):
                qn = q_nope[b, h].to(torch.float32)  # [512]
                qp = q_pe[b, h].to(torch.float32)   # [64]

                # Compute logits1 = qn @ Kc.T → [M]
                logits1 = torch.empty((M,), dtype=torch.float32, device=device)
                # Launch matvec_row_kernel with K=512, M specialized at runtime
                matvec_row_kernel[(1,)](
                    qn.unsqueeze(0), Kc, logits1.unsqueeze(0),
                    K=512, M=M, BLOCK_K=128, num_warps=4, num_stages=2
                )

                # Compute logits2 = qp @ Kp.T → [M] and add
                logits2 = torch.empty((M,), dtype=torch.float32, device=device)
                matvec_row_kernel[(1,)](
                    qp.unsqueeze(0), Kp, logits2.unsqueeze(0),
                    K=64, M=M, BLOCK_K=64, num_warps=4, num_stages=2
                )
                logits = logits1 + logits2  # [M]

                # Compute lse and attn
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                lse[b, h] = torch.empty((), dtype=torch.float32, device=device)
                softmax_lse_kernel[(1,)](
                    logits, lse[b, h], attn,
                    M=M, sm_scale=sm_scale, inv_ln2=inv_ln2,
                    num_warps=4, num_stages=2
                )

                # Compute out[h, :] = attn @ Kc → [512]
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                matvec_row_out_kernel[(1,)](
                    attn, Kc, out_vec.unsqueeze(0),
                    M=M, Kc_dim=Kc_dim, BLOCK_M=256, num_warps=4, num_stages=2
                )
                output[b, h] = out_vec

        # Cast output to bfloat16 as in original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
