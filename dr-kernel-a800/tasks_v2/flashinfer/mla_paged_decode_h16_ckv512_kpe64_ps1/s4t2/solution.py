import torch
import math
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,        # *float32, [1, K]  (qn[h, :] or qp[h, :] reshaped as [1, K])
    B_ptr,        # *float32, [M, K]  (Kc rows or Kp rows)
    C_ptr,        # *float32, [M]     (output vector)
    M,            # int               (number of tokens)
    K: tl.constexpr,  # Kc_dim=512 or Kp_dim=64, constexpr
    BLOCK_K: tl.constexpr = 64,
):
    # One program computes the entire output vector C of length M.
    # A is a single row vector of length K; B has M rows of length K.
    # We compute C[i] = sum_k A[k] * B[i, k] for i in 0..M-1.
    for i in range(0, M):
        acc = 0.0
        for k0 in range(0, K, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            a_vec = tl.load(A_ptr + k_offsets)      # [BLOCK_K]
            b_row = tl.load(B_ptr + i * K + k_offsets)  # [BLOCK_K]
            acc += tl.sum(a_vec * b_row, axis=0)
        tl.store(C_ptr + i, acc)


@triton.jit
def softmax_lse_kernel(
    x_ptr,        # *float32, [M] (logits_scaled)
    out_lse_ptr,  # *float32, [1] (lse per head)
    attn_ptr,     # *float32, [M] (attn vector per head)
    M,            # int
    sm_scale,     # float32
):
    idx = tl.arange(0, M)
    x = tl.load(x_ptr + idx)
    max_val = tl.max(x, axis=0)
    x_shift = x - max_val
    x_scaled = x_shift * sm_scale
    exp_x = tl.exp(x_scaled)
    sum_exp = tl.sum(exp_x, axis=0)
    ln2 = 0.6931471805599453  # math.log(2)
    lse = tl.log(sum_exp) / ln2
    tl.store(out_lse_ptr + tl.arange(0, 1), lse)
    attn = exp_x / sum_exp
    tl.store(attn_ptr + idx, attn)


@triton.jit
def matvec_reduce_kernel(
    A_ptr,        # *float32, [M]     (attn vector)
    B_ptr,        # *float32, [M, Kc_dim] (Kc rows)
    C_ptr,        # *float32, [Kc_dim]    (output vector)
    M,            # int
    Kc_dim: tl.constexpr,  # 512
    BLOCK_M: tl.constexpr = 128,
):
    # Accumulate output vector of length Kc_dim
    out_vec = tl.zeros((Kc_dim,), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask = m_offsets < M
        a_chunk = tl.load(A_ptr + m_offsets, mask=mask, other=0.0)  # [BLOCK_M]
        # Load B chunk as [BLOCK_M, Kc_dim] rows
        b_chunk = tl.zeros((BLOCK_M, Kc_dim), dtype=tl.float32)
        # Iterate rows in the chunk
        for mm in range(BLOCK_M):
            row_idx = m0 + mm
            row_valid = row_idx < M
            if row_valid:
                b_row = tl.load(B_ptr + row_idx * Kc_dim + tl.arange(0, Kc_dim))
                b_chunk[mm, :] = b_row
        # Reduce across rows: out_vec += sum_{rr in chunk} a_chunk[rr] * b_chunk[rr, :]
        for rr in range(BLOCK_M):
            # scalar multiply and reduce over Kc_dim
            out_vec += tl.sum(b_chunk[rr, :] * a_chunk[rr], axis=0)
    tl.store(C_ptr + tl.arange(0, Kc_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        assert H == 16, "num_qo_heads must be 16"
        Kc_dim = q_nope.shape[2]
        assert Kc_dim == 512, "head_dim_ckv must be 512"
        Kp_dim = q_pe.shape[2]
        assert Kp_dim == 64, "head_dim_kpe must be 64"
        assert kv_indptr.numel() == B + 1, "kv_indptr length must be batch_size + 1"

        # Prepare outputs (float32 for computation, cast later)
        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather cache rows
            tok_idx = kv_indices[start:end].to(torch.long)
            # Squeeze the 1-dim from caches
            Kc = ckv_cache[tok_idx, 0, :].to(torch.float32)  # [M, 512]
            Kp = kpe_cache[tok_idx, 0, :].to(torch.float32)  # [M, 64]
            qn = q_nope[b].to(torch.float32)                 # [16, 512]
            qp = q_pe[b].to(torch.float32)                  # [16, 64]

            # For each head, compute logits_scaled, lse, attn, and output
            for h in range(H):
                # Compute logits = qn[h, :] @ Kc.T and qp[h, :] @ Kp.T
                logits_qn = torch.empty((M,), dtype=torch.float32, device=device)
                logits_qp = torch.empty((M,), dtype=torch.float32, device=device)

                # A is a single row: [1, K]
                A_qn = qn[h, :].view(1, Kc_dim)  # [1, 512]
                A_qp = qp[h, :].view(1, Kp_dim)  # [1, 64]

                # Launch Triton matvec kernels
                matvec_row_kernel[(1,)](
                    A_qn, Kc, logits_qn, M, Kc_dim, BLOCK_K=64
                )
                matvec_row_kernel[(1,)](
                    A_qp, Kp, logits_qp, M, Kp_dim, BLOCK_K=64
                )
                logits_scaled = logits_qn + logits_qp  # [M]
                logits_scaled = logits_scaled * sm_scale

                # Compute lse and attn in Triton
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                lse_elem = torch.empty((1,), dtype=torch.float32, device=device)

                softmax_lse_kernel[(1,)](
                    logits_scaled, lse_elem, attn, M, sm_scale
                )
                lse[b, h] = lse_elem[0]

                # Compute out = attn @ Kc -> [512]
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                matvec_reduce_kernel[(1,)](
                    attn, Kc, out_vec, M, Kc_dim, BLOCK_M=128
                )
                output[b, h, :] = out_vec

        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
