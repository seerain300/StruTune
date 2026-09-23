import torch
import math
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,          # *float32, [1, K], we pass a 1xK pointer (row vector)
    B_ptr,          # *float32, [M, K]
    C_ptr,          # *float32, [1, M]
    K: tl.constexpr,  # int, e.g., 512
    M,              # int, number of rows in B (runtime)
    BLOCK_K: tl.constexpr  # tile size along K, e.g., 64 or 128
):
    # One program computes the entire C row vector of length M
    offs = tl.arange(0, BLOCK_K)
    # Initialize output accumulator for the M-vector
    c_vec = tl.zeros([M], dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in tl.static_range(0, K, BLOCK_K):
        k_idx = k0 + offs  # [BLOCK_K]
        # Load A chunk (A is 1xK, so we index A_ptr + k_idx)
        a_chunk = tl.load(A_ptr + k_idx, mask=k_idx < K, other=0.0)
        # Accumulate dot with each row of B over these K elements
        # For each row i in 0..M-1, compute dot = sum_j a_chunk[j] * B[i, j]
        # We'll loop over i explicitly and accumulate into c_vec[i]
        # Note: M is runtime; Triton supports loops over runtime M
        for i in range(0, M):
            b_vec = tl.load(B_ptr + i * K + k_idx, mask=k_idx < K, other=0.0)
            c_vec[i] += tl.sum(a_chunk * b_vec, axis=0)

    # Store the result: C_ptr points to a 1xM row; write c_vec to it
    tl.store(C_ptr + tl.arange(0, M), c_vec)


@triton.jit
def softmax_lse_kernel(
    x_ptr,          # *float32, [M] (logits_scaled)
    out_lse_ptr,    # *float32, [1] (lse per head)
    attn_ptr,       # *float32, [M] (attn vector per head)
    M: tl.constexpr,     # int, must be constexpr for tl.arange
    sm_scale,       # float32
):
    # Compute max over x
    max_val = -float("inf")
    for i in range(0, M):
        val = tl.load(x_ptr + i)
        if val > max_val:
            max_val = val

    # Compute sum of exp((x - max_val) * sm_scale)
    sum_exp = 0.0
    for i in range(0, M):
        val = tl.load(x_ptr + i)
        sum_exp += tl.exp((val - max_val) * sm_scale)

    # lse = log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse_val = tl.log(sum_exp) / ln2
    tl.store(out_lse_ptr, lse_val)

    # Write attention vector
    for i in range(0, M):
        val = tl.load(x_ptr + i)
        attn_i = tl.exp((val - max_val) * sm_scale) / sum_exp
        tl.store(attn_ptr + i, attn_i)


@triton.jit
def matvec_reduce_kernel(
    A_ptr,          # *float32, [M] (attention vector)
    B_ptr,          # *float32, [M, Kc] (Kc rows)
    C_ptr,          # *float32, [1, Kc] (output per head)
    M: tl.constexpr,      # int, number of rows in A/B
    Kc: tl.constexpr,     # int, Kc_dim (e.g., 512)
    BLOCK_M: tl.constexpr  # tile size along M, e.g., 128
):
    # One program computes the entire C row vector of length Kc
    offs = tl.arange(0, Kc)
    c_vec = tl.zeros([Kc], dtype=tl.float32)

    for m0 in tl.static_range(0, M, BLOCK_M):
        m_idx = m0 + tl.arange(0, BLOCK_M)
        mask = m_idx < M
        # Load A chunk vector: A[m_idx]
        a_chunk = tl.load(A_ptr + m_idx, mask=mask, other=0.0)
        # For each d in 0..Kc-1, accumulate sum over m of a_chunk[m] * B[m, d]
        for d in range(0, Kc):
            b_col = tl.load(B_ptr + tl.arange(0, M) * Kc + d, mask=mask, other=0.0)
            # We need to multiply elementwise a_chunk[m] * b_col[m] for all m in chunk and sum
            # Since m_idx is [BLOCK_M], we need to match a_chunk with b_col by indexing correctly.
            # Better approach: loop over m in the chunk and accumulate.
            for j in range(0, BLOCK_M):
                if (m0 + j) < M:
                    c_vec[d] += a_chunk[j] * b_col[j]

    tl.store(C_ptr + tl.arange(0, Kc), c_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA (Triton runs on GPU)
        device = q_nope.device
        assert device.type == 'cuda', "ModelNew requires CUDA tensors for Triton kernels"

        B, H, Kc = q_nope.shape
        assert H == 16, "num_qo_heads must be 16"
        assert Kc == 512, "head_dim_ckv must be 512"

        Kp_dim = q_pe.shape[-1]
        assert Kp_dim == 64, "head_dim_kpe must be 64"

        # Prepare Kc_all and Kp_all
        # Note: squeeze(1) removes the size-1 dim, yielding [N, 512] and [N, 64]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)

        # Output buffers
        output = torch.empty((B, H, Kc), dtype=torch.bfloat16, device=device)  # final output
        lse = torch.empty((B, H), dtype=torch.float32, device=device)           # lse per batch, per head

        # Process each batch element
        for b in range(B):
            # Determine token range for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                # No tokens for this batch element: output zeros and skip
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp rows
            tok_idx = kv_indices[start:start + M].to(torch.int32).contiguous()
            Kc_sub = Kc_all[tok_idx]  # [M, 512]
            Kp_sub = Kp_all[tok_idx]  # [M, 64]

            # For each head h
            for h in range(H):
                # Prepare qn and qp (float32 for accumulation)
                qn = q_nope[b, h, :].contiguous().to(torch.float32)  # [512]
                qp = q_pe[b, h, :].contiguous().to(torch.float32)    # [64]

                # Allocate temporary buffers for logits from qn @ Kc.T and qp @ Kp.T
                logits_qn = torch.empty((M,), dtype=torch.float32, device=device)
                logits_qp = torch.empty((M,), dtype=torch.float32, device=device)

                # Launch Triton matvec kernels for both terms
                # Note: pass A as [1, K] by flattening qn to shape (K,) and making it 1xK in pointer usage
                # For matvec_row_kernel, A is a 1xK row; we pass qn as 1xK pointer by constructing a 2D tensor or using pointer directly.
                # Simpler: use A_ptr pointing to qn as 1xK row; Triton expects 1D for A, so we pass qn directly as A_ptr.
                # We'll pass qn as 1xK by creating a 2D tensor [1, K].
                A_qn = qn.view(1, -1)  # [1, 512]
                A_qp = qp.view(1, -1)  # [1, 64]
                B_qn = Kc_sub            # [M, 512]
                B_qp = Kp_sub            # [M, 64]
                C_qn = logits_qn          # [1, M] in pointer terms
                C_qp = logits_qp          # [1, M] in pointer terms

                # Choose BLOCK sizes
                BLOCK_K_qn = 64
                BLOCK_K_qp = 64

                # Run qn @ Kc.T
                matvec_row_kernel[(1,)](
                    A_qn, B_qn, C_qn,
                    K=Kc, M=M, BLOCK_K=BLOCK_K_qn,
                    num_warps=4, num_stages=2
                )
                # Run qp @ Kp.T
                matvec_row_kernel[(1,)](
                    A_qp, B_qp, C_qp,
                    K=Kp_dim, M=M, BLOCK_K=BLOCK_K_qp,
                    num_warps=4, num_stages=2
                )

                # Compute logits_scaled = logits_qn + logits_qp
                logits_scaled = (logits_qn[0] + logits_qp[0]) * sm_scale  # [M], float32

                # Prepare output buffers for lse and attn (1-element and M-element vectors)
                out_lse = torch.empty((1,), dtype=torch.float32, device=device)
                attn_vec = torch.empty((M,), dtype=torch.float32, device=device)

                # Launch softmax_lse_kernel: Triton requires constexpr M; we pass M as constexpr
                softmax_lse_kernel[(1,)](
                    logits_scaled, out_lse, attn_vec,
                    M=M, sm_scale=sm_scale,
                    num_warps=2, num_stages=2
                )

                lse[b, h] = out_lse[0]

                # Compute out[h, :] = attn_vec @ Kc_sub -> [512]
                # Allocate output vector for head h
                out_vec = torch.empty((Kc,), dtype=torch.float32, device=device)

                # Run matvec_reduce_kernel: out_vec = attn_vec @ Kc_sub
                # attn_vec: [M], Kc_sub: [M, Kc], out_vec: [1, Kc]
                matvec_reduce_kernel[(1,)](
                    attn_vec, Kc_sub, out_vec,
                    M=M, Kc=Kc, BLOCK_M=128,
                    num_warps=4, num_stages=2
                )

                # Assign to output tensor
                output[b, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
