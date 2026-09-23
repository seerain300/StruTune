import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute C[j] = sum_i A[i] * B[j, i] for j in [0, N).
# A is 1D vector (float32) of length M. B is a contiguous [N, M] matrix (float32).
# We process M in chunks of BLOCK_M, accumulate into acc[j], and store acc[j] for j < N.
@triton.jit
def matmul_row(A_ptr, B_ptr, C_ptr,
               N, M: tl.constexpr, BLOCK_M: tl.constexpr):
    # Vector of output indices handled by this program instance
    j_vec = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over M in chunks of BLOCK_M
    for m0 in range(0, M, BLOCK_M):
        # Load A chunk: elements A[m0 + j_vec]
        a_chunk = tl.load(A_ptr + (m0 + j_vec), mask=(m0 + j_vec) < M, other=0.0)
        # Load corresponding block of B: shape [BLOCK_M, BLOCK_M], rows are j_vec, cols are (m0 + j_vec)
        # For a contiguous [N, M] B, address of B[j, i] is B_ptr + j * M + i.
        b_block = tl.load(
            B_ptr + (j_vec[:, None] * M) + ((m0 + j_vec)[None, :]),
            mask=(j_vec[:, None] < N) & ((m0 + j_vec)[None, :] < M),
            other=0.0,
        )
        # Accumulate: acc += a_chunk * row of b_block
        for ii in range(BLOCK_M):
            i = m0 + ii
            ai = a_chunk[ii]  # scalar
            row = b_block[ii, :]  # vector length BLOCK_M
            acc += ai * row

    # Store results for j in [0, N)
    tl.store(C_ptr + j_vec, acc, mask=j_vec < N)


# Triton kernel: compute stable softmax over TOPK entries in X with validity mask (Valid).
# Writes softmax probabilities to Out_ptr for valid entries and base-2 logsumexp to LSE_ptr[0].
@triton.jit
def softmax_logsumexp2_row(X_ptr, Valid_ptr, Out_ptr, LSE_ptr,
                            N: tl.constexpr, TOPK: tl.constexpr):
    # Pass 1: find max over valid entries
    m = -float("inf")
    for j in range(0, TOPK):
        valid_j = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        if valid_j != 0:
            m = tl.maximum(m, xj)

    # Pass 2: sum of exp(xj - m) over valid entries
    s = 0.0
    for j in range(0, TOPK):
        valid_j = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        if valid_j != 0:
            s += tl.exp(xj - m)

    # LSE = m + log(s) / ln(2)
    lse_val = m + tl.log(s) * 1.4426950408889634  # 1/ln(2)
    tl.store(LSE_ptr + 0, lse_val)

    # Pass 3: write normalized softmax to Out_ptr for valid entries
    inv_s = 1.0 / s
    for j in range(0, TOPK):
        valid_j = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        if valid_j != 0:
            out_j = tl.exp(xj - m) * inv_s
        else:
            out_j = 0.0
        tl.store(Out_ptr + j, out_j)


# Triton kernel: compute Out[k] = sum_{i=0..N-1} Attn[i] * B[k, i] for k in [0, OUT).
# Attn is 1D of length N (float32). B is a contiguous [N, OUT] matrix (float32).
@triton.jit
def reduction_row(Attn_ptr, B_ptr, Out_ptr,
                  N, OUT: tl.constexpr, BLOCK_OUT: tl.constexpr):
    k_vec = tl.arange(0, BLOCK_OUT)
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)

    # Loop over N in chunks
    for n0 in range(0, N, BLOCK_OUT):
        attn_chunk = tl.load(Attn_ptr + (n0 + k_vec), mask=(n0 + k_vec) < N, other=0.0)
        # Load B chunk: shape [BLOCK_OUT, BLOCK_OUT], rows correspond to n indices (n0 + k_vec),
        # cols correspond to k indices (k_vec). For B[n, k], address is B_ptr + n * OUT + k.
        b_block = tl.load(
            B_ptr + ((n0 + k_vec)[:, None] * OUT) + (k_vec[None, :]),
            mask=(n0 + k_vec)[:, None] < N,
            other=0.0,
        )
        # Accumulate: acc += attn_chunk * column of b_block
        for ii in range(BLOCK_OUT):
            n = n0 + ii
            an = attn_chunk[ii]
            col = b_block[:, ii]  # vector length BLOCK_OUT
            acc += an * col

    tl.store(Out_ptr + k_vec, acc, mask=k_vec < OUT)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure device consistency
        device = q_nope.device
        # Reshape flattened KV caches and cast to float32
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        total_pages = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 64 and ckv_cache.shape[2] == 512
        assert kpe_cache.shape[1] == 64 and kpe_cache.shape[2] == 64
        Kc_all = ckv_cache.reshape(-1, 512).contiguous().to(torch.float32)  # [num_tokens*64, 512]
        Kp_all = kpe_cache.reshape(-1, 64).contiguous().to(torch.float32)   # [num_tokens*64, 64]

        # Output buffers
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # int32 tensor of length TOPK=2048
            valid_mask = (indices != -1)
            valid_indices = indices[valid_mask].to(torch.int32)
            num_valid = valid_indices.numel()
            if num_valid == 0:
                output[t].zero_()
                lse[t] = 0.0
                continue

            # Gather rows from flattened caches
            Kc_rows = Kc_all[valid_indices]           # [num_valid, 512]
            Kp_rows = Kp_all[valid_indices]           # [num_valid, 64]

            # qn and qp for this token
            qn = q_nope[t]  # [num_qo_heads, 512]
            qp = q_pe[t]    # [num_qo_heads, 64]
            qn_f32 = qn.to(torch.float32)  # [num_qo_heads, 512]
            qp_f32 = qp.to(torch.float32)  # [num_qo_heads, 64]

            for h in range(num_qo_heads):
                A_qn = qn_f32[h]   # [512]
                A_qp = qp_f32[h]   # [64]

                # Compute qn @ Kc_rows.T and qp @ Kp_rows.T
                logits_qn = torch.empty((num_valid,), dtype=torch.float32, device=device)
                logits_qp = torch.empty((num_valid,), dtype=torch.float32, device=device)

                grid_qn = (triton.cdiv(num_valid, 64),)
                matmul_row[grid_qn](A_qn, Kc_rows, logits_qn, num_valid, 512, 64, num_warps=2, num_stages=2)

                grid_qp = (triton.cdiv(num_valid, 64),)
                matmul_row[grid_qp](A_qp, Kp_rows, logits_qp, num_valid, 64, 64, num_warps=2, num_stages=2)

                logits = logits_qn + logits_qp  # [num_valid], float32
                logits_scaled = logits * sm_scale  # base-2 logsumexp and attention will use this

                # Prepare Valid mask: we only have num_valid valid positions
                # Build int32 Valid tensor: 1 for valid indices, 0 otherwise. Length TOPK=2048.
                Valid = torch.ones((2048,), dtype=torch.int32, device=device)
                # Compute base-2 logsumexp
                Out = torch.empty((2048,), dtype=torch.float32, device=device)
                LSE = torch.empty((1,), dtype=torch.float32, device=device)
                softmax_logsumexp2_row[(1,)](logits_scaled, Valid, Out, LSE, TOPK=2048, num_warps=2, num_stages=2)
                lse[t, h] = LSE[0]

                # Extract attention vector for valid positions: Out[0:num_valid]
                attn = Out[0:num_valid]  # [num_valid], float32

                # Compute final output: attn @ Kc_rows → [512]
                Out_final = torch.empty((512,), dtype=torch.float32, device=device)
                grid_reduce = (triton.cdiv(512, 64),)
                reduction_row[grid_reduce](attn, Kc_rows, Out_final, num_valid, 512, 64, num_warps=2, num_stages=2)

                output[t, h] = Out_final.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
