import math
import torch

import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,          # *float32, [1, K] (we pass [1, K] pointer; kernel treats as [K])
    B_ptr,          # *float32, [M, K]
    C_ptr,          # *float32, [1, M]
    K: tl.constexpr,         # reduction dimension (e.g., 512 or 64)
    M,                       # number of rows in B (runtime int)
    BLOCK_K: tl.constexpr    # tile size along K (e.g., 128 or 64)
):
    # We compute the output vector of length M: out_vec[m] = sum_k A[k] * B[m, k]
    out_vec = tl.zeros((M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K], constexpr
        mask_k = k_idx < K
        a_chunk = tl.load(A_ptr + k_idx, mask=mask_k, other=0.0)  # [BLOCK_K]
        for m in range(0, M):
            b_row = tl.load(B_ptr + m * K + k_idx, mask=mask_k, other=0.0)  # [BLOCK_K]
            out_vec[m] += tl.sum(a_chunk * b_row, axis=0)
    # Store output row vector of length M
    # C_ptr is [1, M]; we write to row 0
    for m in range(0, M):
        tl.store(C_ptr + m, out_vec[m])


@triton.jit
def softmax_lse_kernel(
    x_ptr,             # *float32, [M] (logits_scaled)
    out_lse_ptr,       # *float32, [1] (lse per head)
    attn_ptr,          # *float32, [M] (attn vector per head)
    M: tl.constexpr,   # number of elements
    sm_scale: tl.constexpr,  # scale factor (float), will use float literals
):
    # Compute max over x
    max_x = -float("inf")
    for i in range(0, M):
        vi = tl.load(x_ptr + i)
        if vi > max_x:
            max_x = vi
    # Compute sum of exp(x - max_x)
    sum_exp = 0.0
    for i in range(0, M):
        vi = tl.load(x_ptr + i)
        sum_exp += tl.exp((vi - max_x) * sm_scale)
    # lse = (max_x + log(sum_exp)) / ln(2)
    ln2 = 0.6931471805599453
    lse_val = (max_x + tl.log(sum_exp)) / ln2
    tl.store(out_lse_ptr, lse_val)
    # Compute attn and store
    for i in range(0, M):
        vi = tl.load(x_ptr + i)
        attn_i = tl.exp((vi - max_x) * sm_scale) / sum_exp
        tl.store(attn_ptr + i, attn_i)


@triton.jit
def matvec_row_out_kernel(
    A_ptr,             # *float32, [M] (single row of attn)
    B_ptr,             # *float32, [M, Kc_dim]
    C_ptr,             # *float32, [1, Kc_dim]
    M: tl.constexpr,   # number of rows in A (attn length)
    Kc_dim: tl.constexpr,  # output dimension (e.g., 512)
    BLOCK_M: tl.constexpr, # tile size along M (e.g., 128 or 256)
    BLOCK_Kc: tl.constexpr  # tile size along Kc_dim (e.g., 64 or 128)
):
    out_vec = tl.zeros((Kc_dim,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_m = m_idx < M
        a_chunk = tl.load(A_ptr + m_idx, mask=mask_m, other=0.0)  # [BLOCK_M]
        for k_start in range(0, Kc_dim, BLOCK_Kc):
            k_idx = k_start + tl.arange(0, BLOCK_Kc)  # [BLOCK_Kc]
            mask_k = k_idx < Kc_dim
            # B_ptr layout is [M, Kc_dim] with row stride Kc_dim
            b_chunk = tl.load(B_ptr + m_idx[:, None] * Kc_dim + k_idx[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            # out_vec[k_idx] += sum_m a_chunk[m] * b_chunk[m, k]
            contrib = tl.sum(b_chunk * a_chunk[:, None], axis=0)  # [BLOCK_Kc]
            out_vec[k_idx] += contrib
    # Store output row vector [Kc_dim]
    for k in range(0, Kc_dim):
        tl.store(C_ptr + k, out_vec[k])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, 16, 512], bfloat16
        q_pe: [B, 16, 64], bfloat16
        ckv_cache: [N, 1, 512], bfloat16
        kpe_cache: [N, 1, 64], bfloat16
        kv_indptr: [B+1], int32
        kv_indices: [L], int32
        sm_scale: float32 scalar
        Returns:
        output: [B, 16, 512], bfloat16
        lse: [B, 16], float32
        """
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]  # num_qo_heads
        Kc_dim = q_nope.shape[2]  # 512
        # Squeeze caches to [N, *]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)

        # Output and lse tensors
        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Determine token range for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                # No valid tokens for this batch
                # output[b] remains zeros, lse[b] remains zeros
                continue

            tokens = kv_indices[start:end]  # [M]
            M = tokens.numel()

            # Gather Kc and Kp rows
            Kc = Kc_all[tokens]  # [M, 512], float32
            Kp = Kp_all[tokens]  # [M, 64], float32

            # qn and qp for each head
            for h in range(H):
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [64]

                # 1) Compute logits = qn @ Kc.T
                logits1 = torch.empty((1, M), dtype=torch.float32, device=device)
                matvec_row_kernel[(1,)](
                    qn, Kc, logits1,
                    K=Kc_dim, M=M, BLOCK_K=128
                )
                # 2) Compute logits += qp @ Kp.T
                logits2 = torch.empty((1, M), dtype=torch.float32, device=device)
                matvec_row_kernel[(1,)](
                    qp, Kp, logits2,
                    K=64, M=M, BLOCK_K=128
                )
                logits = logits1 + logits2  # [1, M]

                # 3) Softmax + LSE: lse[h], attn[M]
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                lse[b, h] = torch.empty((), dtype=torch.float32, device=device)  # dummy for assignment; overwritten by kernel
                softmax_lse_kernel[(1,)](
                    logits[0], lse[b, h], attn,
                    M=M, sm_scale=sm_scale
                )
                # 4) out[h, :] = attn @ Kc
                out_vec = torch.empty((1, Kc_dim), dtype=torch.float32, device=device)
                matvec_row_out_kernel[(1,)](
                    attn, Kc, out_vec,
                    M=M, Kc_dim=Kc_dim, BLOCK_M=256, BLOCK_Kc=64
                )
                output[b, h] = out_vec[0]

        # Cast output to bfloat16 as required by original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
