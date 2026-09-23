import torch
import math
import triton
import triton.language as tl


@triton.jit
def matmul_row_kernel(
    A_ptr,        # *float32, [1, K]
    B_ptr,        # *float32, [M, K]
    C_ptr,        # *float32, [1, M]
    M,            # int
    K,            # constexpr int (512 or 64)
    stride_b,     # int, stride of B rows (typically K)
):
    # one program computes the row result
    # A is a single row vector of length K; B has M rows of length K
    # We accumulate output vector c_vec of length M
    c_vec = tl.zeros((M,), dtype=tl.float32)

    BLOCK_K = 64
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        a_vec = tl.load(A_ptr + k_offsets)      # [BLOCK_K]
        # Load a block of rows from B: shape [BLOCK_K, M]
        # B[idx, j] = tl.load(B_ptr + idx * stride_b + j)
        b_block = tl.zeros((BLOCK_K, M), dtype=tl.float32)
        for kk in range(BLOCK_K):
            row_idx = k0 + kk
            # valid if row_idx < K
            # mask ensures we don't read beyond K
            # Since we loop up to K, this is always valid
            b_row = tl.load(B_ptr + row_idx * stride_b + tl.arange(0, M))
            b_block[kk, :] = b_row
        # Accumulate dot for each column j
        # a_vec[None, :] is [1, BLOCK_K], b_block[:, j] is [BLOCK_K]
        c_vec += tl.sum(b_block * a_vec[None, :], axis=0)

    # store to C
    tl.store(C_ptr + tl.arange(0, M), c_vec)


@triton.jit
def softmax_lse_kernel(
    x_ptr,        # *float32, [M]
    out_lse_ptr,  # *float32, scalar
    attn_ptr,     # *float32, [M]
    M,            # int
    sm_scale,     # float32
):
    # Compute max for numerical stability
    max_val = -float('inf')
    for i in range(0, M):
        max_val = tl.maximum(max_val, tl.load(x_ptr + i))
    # compute x = (x - max) * sm_scale, exp, sum
    sum_exp = 0.0
    for i in range(0, M):
        xi = tl.load(x_ptr + i)
        x_scaled = (xi - max_val) * sm_scale
        sum_exp += tl.exp(x_scaled)
    # lse = log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453  # math.log(2)
    tl.store(out_lse_ptr, tl.log(sum_exp) / ln2)
    # write attn
    for i in range(0, M):
        xi = tl.load(x_ptr + i)
        x_scaled = (xi - max_val) * sm_scale
        attn_i = tl.exp(x_scaled) / sum_exp
        tl.store(attn_ptr + i, attn_i)


@triton.jit
def out_matmul_row_kernel(
    A_ptr,        # *float32, [M] (attn vector)
    B_ptr,        # *float32, [M, Kc_dim] (Kc rows)
    C_ptr,        # *float32, [Kc_dim]
    M,            # int
    Kc_dim: tl.constexpr,  # 512
):
    # Accumulate out vector of length Kc_dim
    out_vec = tl.zeros((Kc_dim,), dtype=tl.float32)

    BLOCK_M = 128
    for m0 in range(0, M, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask = m_offsets < M
        # Load A chunk
        a_chunk = tl.load(A_ptr + m_offsets, mask=mask, other=0.0)  # [BLOCK_M]
        # Load B chunk: [BLOCK_M, Kc_dim]
        b_chunk = tl.zeros((BLOCK_M, Kc_dim), dtype=tl.float32)
        for mm in range(BLOCK_M):
            row_idx = m0 + mm
            row_mask = row_idx < M
            b_row = tl.load(B_ptr + row_idx * Kc_dim + tl.arange(0, Kc_dim), mask=row_mask, other=0.0)
            b_chunk[mm, :] = b_row
        # out_vec += sum over rows: a_chunk * b_chunk_row for each row in chunk
        # We can use dot since a_chunk is [BLOCK_M] and b_chunk[:, kk] is [BLOCK_M], and we want sum over rows (mm dimension)
        # For each kk, we compute dot(a_chunk, b_chunk[:, kk])
        for kk in range(Kc_dim):
            out_vec[kk] += tl.sum(b_chunk[:, kk] * a_chunk, axis=0)

    tl.store(C_ptr + tl.arange(0, Kc_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Shapes
        B, H, Kc_dim = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        assert H == 16, "num_qo_heads must be 16"
        assert Kc_dim == 512, "head_dim_ckv must be 512"
        assert Kp_dim == 64, "head_dim_kpe must be 64"
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "page_size must be 1 (second dim is 1)"

        # Prepare output buffers
        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)  # [B, 16, 512] fp32 compute
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Precompute Kc_all and Kp_all from caches (device tensors)
        # Keep them as float32 for compute
        # Note: squeeze(1) on device tensor returns a view; we cast to fp32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                # No valid tokens for this batch
                output[b].zero_()
                lse[b].zero_()
                continue

            tok_idx = kv_indices[start:end].to(torch.long)  # [M]
            # Gather rows
            Kc = Kc_all[tok_idx]                           # [M, 512]
            Kp = Kp_all[tok_idx]                          # [M, 64]

            # For each head h
            for h in range(H):
                # Cast q vectors to fp32 for compute
                qn = q_nope[b, h].to(torch.float32)       # [512]
                qp = q_pe[b, h].to(torch.float32)        # [64]

                # Compute logits[h, :] = qn @ Kc.T + qp @ Kp.T, shape [M]
                # Allocate logits_vec
                logits_vec = torch.empty((M,), dtype=torch.float32, device=device)
                # Kernel: matmul_row for qn @ Kc.T
                matmul_row_kernel[(1,)](
                    qn, Kc, logits_vec, M, Kc_dim, Kc.stride(0)
                )
                # Kernel: matmul_row for qp @ Kp.T
                out_qp = torch.empty((M,), dtype=torch.float32, device=device)
                matmul_row_kernel[(1,)](
                    qp, Kp, out_qp, M, Kp_dim, Kp.stride(0)
                )
                # Add results
                logits_vec = logits_vec + out_qp
                # Scale by sm_scale
                logits_vec = logits_vec * sm_scale

                # Compute lse and attn in Triton
                lse_scalar = torch.empty((1,), dtype=torch.float32, device=device)
                attn_vec = torch.empty((M,), dtype=torch.float32, device=device)
                softmax_lse_kernel[(1,)](
                    logits_vec, lse_scalar, attn_vec, M, sm_scale
                )
                # Store per-head lse for this batch
                lse[b, h] = lse_scalar[0]

                # Compute out vector for this head: attn @ Kc
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                out_matmul_row_kernel[(1,)](
                    attn_vec, Kc, out_vec, M, Kc_dim
                )
                # Store to output [B, H, 512]
                output[b, h] = out_vec

        # Cast final output to bfloat16 as per original
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
