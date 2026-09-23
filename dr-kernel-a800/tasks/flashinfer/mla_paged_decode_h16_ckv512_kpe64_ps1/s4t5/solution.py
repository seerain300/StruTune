import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,          # *float32, length K
    B_ptr,          # *float32, shape [M, K]
    C_ptr,          # *float32, output length M (we will write to C as [M] contiguous)
    K: tl.constexpr,      # int, e.g., 512
    M,              # int, number of rows in B (runtime)
    BLOCK_K: tl.constexpr # tile size along K, e.g., 64 or 128
):
    """
    Compute C[i] = sum_k A[k] * B[i, k] for i in 0..M-1, with A of length K and B of shape [M, K].
    We implement this as accumulating BLOCK_M rows at a time into a vector c_vec and store with mask.
    """
    # Vector of output indices we handle per program
    BLOCK_M = 64
    grid_m = tl.cdiv(M, BLOCK_M)
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask = m_offsets < M

    # Accumulator for BLOCK_M outputs
    c_vec = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Reduction over K in tiles
    for k_start in tl.static_range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + k_offsets)  # [BLOCK_K]
        # Load a tile of B rows: shape [BLOCK_M, BLOCK_K]
        # We load B[m, k] for m in m_offsets, k in k_offsets
        # Pointer arithmetic: B_ptr + m_offsets[:, None] * K + k_offsets[None, :]
        b_tile = tl.load(
            B_ptr + m_offsets[:, None] * K + k_offsets[None, :],
            mask=mask[:, None],
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        # Accumulate: dot(a, b_tile) for each m in m_offsets
        # Sum over axis=1 (K dimension) -> [BLOCK_M]
        c_vec += tl.sum(b_tile * a[None, :], axis=1)

    # Store results to C_ptr (contiguous [M])
    tl.store(C_ptr + m_offsets, c_vec, mask=mask)


@triton.jit
def softmax_lse_kernel(
    x_ptr,        # *float32, [M] (logits_scaled vector)
    out_lse_ptr,  # *float32, [1] (lse per head)
    attn_ptr,     # *float32, [M] (attn vector per head)
    M: tl.constexpr,   # int, vector length (compile-time)
    sm_scale: tl.constexpr,  # float32
):
    """
    Compute lse = logsumexp(x / ln(2)) and write attn = exp(x / ln(2)) / sum_exp.
    x_ptr is a vector of length M, sm_scale is a scalar.
    """
    # Constants
    LN2 = 0.6931471805599453  # math.log(2.0)

    # Load vector x
    idx = tl.arange(0, M)
    x = tl.load(x_ptr + idx)  # [M]
    y = x / LN2  # scale by 1/log(2)

    # Numerically stable softmax
    max_val = tl.max(y)
    y_shifted = y - max_val
    exp_y = tl.exp(y_shifted)
    sum_exp = tl.sum(exp_y)
    attn = exp_y / sum_exp

    # logsumexp
    lse = tl.log(sum_exp) / LN2  # logsumexp(y) / ln(2)

    # Store results
    tl.store(attn_ptr + idx, attn)
    tl.store(out_lse_ptr, lse)


@triton.jit
def matvec_row_to_K_kernel(
    A_ptr,          # *float32, length M (row vector we pass as 1D)
    B_ptr,          # *float32, shape [M, K] (e.g., K=512)
    C_ptr,          # *float32, output length K (we write [K] contiguous)
    K: tl.constexpr,      # int, e.g., 512
    M,              # int, number of rows in A (runtime)
    BLOCK_M: tl.constexpr  # tile size along M, e.g., 128
):
    """
    Compute C[k] = sum_m A[m] * B[m, k] for k in 0..K-1.
    A is [M], B is [M, K]. We process M in tiles and accumulate per k.
    """
    # Reduction over M in tiles; output is vector of length K
    for m_start in tl.static_range(0, M, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M

        # Load A tile
        a_tile = tl.load(A_ptr + m_offsets, mask=mask_m, other=0.0)  # [BLOCK_M]

        # Accumulator for output vector of length K
        c_acc = tl.zeros([K], dtype=tl.float32)

        # For each k in 0..K-1, accumulate sum_m a_tile[m] * B[m, k]
        # We do this by looping k_seq across 0..K-1; Triton supports static_range with compile-time K.
        for k_seq in tl.static_range(0, K):
            # Load column k_seq across M rows: B[m, k_seq] for m in m_offsets
            b_col = tl.load(B_ptr + m_offsets * K + k_seq, mask=mask_m, other=0.0)  # [BLOCK_M]
            c_acc[k_seq] += tl.sum(a_tile * b_col, axis=0)

        # Store to C_ptr as contiguous [K]
        # We need to store each k element; Triton allows vectorized stores to contiguous memory.
        tl.store(C_ptr + tl.arange(0, K), c_acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln2 = math.log(2.0)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation of the original run function.
        Returns (output, lse) where:
          - output: [batch_size, 16, 512] bfloat16
          - lse: [batch_size, 16] float32
        """
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Kc_dim = q_nope.shape[2]  # 512
        Kp_dim = q_pe.shape[2]    # 64

        # Prepare output tensors
        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)  # we will fill and cast later
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Precompute Kc_all and Kp_all as float32 (squeezing batch dim from caches)
        # Note: ckv_cache and kpe_cache are [num_pages, 1, dim] -> squeeze(1) to [num_pages, dim]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)

        # Process each batch b
        for b in range(B):
            # Determine valid token range
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                # No valid tokens for this batch, set outputs to zero
                lse[b] = -float("inf")
                # output[b, :, :] = 0; but we will compute later to keep consistent dtype
                continue

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[start:end]
            Kc = Kc_all[tok_idx]  # [M, Kc_dim]
            Kp = Kp_all[tok_idx]  # [M, Kp_dim]

            # Ensure contiguous
            Kc = Kc.contiguous()
            Kp = Kp.contiguous()

            # For each head h
            for h in range(H):
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [Kc_dim]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [Kp_dim]

                # 1) Compute logits_qc = qn @ Kc.T -> [M]
                logits_qc = torch.empty((M,), dtype=torch.float32, device=device)
                # Launch Triton kernel: A is qn (length Kc_dim), B is Kc (shape [M, Kc_dim]), output is logits_qc (length M)
                grid_qc = (1,)
                matvec_row_kernel[grid_qc](
                    qn, Kc, logits_qc, Kc_dim, M, BLOCK_K=128
                )

                # 2) Compute logits_qp = qp @ Kp.T -> [M]
                logits_qp = torch.empty((M,), dtype=torch.float32, device=device)
                grid_qp = (1,)
                matvec_row_kernel[grid_qp](
                    qp, Kp, logits_qp, Kp_dim, M, BLOCK_K=128
                )

                # 3) logits = logits_qc + logits_qp
                logits = logits_qc + logits_qp

                # 4) Compute lse and attn per head
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                # Run Triton softmax_lse_kernel: we need M as constexpr. Triton will specialize per M.
                # Note: logits is a torch tensor [M] in float32.
                grid_lse = (1,)
                softmax_lse_kernel[grid_lse](
                    logits, lse[b].unsqueeze(0), attn, M, sm_scale
                )

                # 5) Compute out[h, :] = attn @ Kc -> [Kc_dim]
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                matvec_row_to_K_kernel[(1,)](
                    attn, Kc, out_vec, Kc_dim, M, BLOCK_M=128
                )

                # Store out vector for this head
                output[b, h] = out_vec

            # For batches with M > 0, lse[b] is computed by Triton; for M <= 0, we set it to -inf
            if M <= 0:
                lse[b].fill_(-float("inf"))

        # Cast output to bfloat16 as in the original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
