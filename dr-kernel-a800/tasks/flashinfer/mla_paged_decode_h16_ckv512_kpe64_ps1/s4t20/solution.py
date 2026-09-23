import torch
import triton
import triton.language as tl


# Compute a single-row matvec: A: [1, K], B: [M, K] -> C: [1, M]
# Reduction across K in tiles of BLOCK_K; M is runtime but we specialize per call by passing M as tl.constexpr.
@triton.jit
def matvec_row_kernel(
    A_ptr,           # *float32, pointer to row vector of length K (we pass [1, K] but load as [K])
    B_ptr,           # *float32, pointer to matrix of shape [M, K]
    C_ptr,           # *float32, pointer to output row vector of length M (we pass [1, M] but write to [M])
    K: tl.constexpr, # int: reduction dimension (e.g., 512 or 64)
    M: tl.constexpr, # int: number of rows in B (specialize per batch)
    BLOCK_K: tl.constexpr  # tile size along K (e.g., 128)
):
    # Accumulator for output vector
    out_vec = tl.zeros((M,), dtype=tl.float32)
    # Tile over K
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)  # vector of K indices
        mask_k = k_idx < K
        # Load A chunk: A_ptr + k_idx
        a = tl.load(A_ptr + k_idx, mask=mask_k, other=0.0)  # [BLOCK_K]
        # For each row m, compute dot(a, B[m, k_idx]) and accumulate
        # We'll iterate m and update out_vec[m]
        for m in range(0, M):
            b_row_ptr = B_ptr + m * K + k_idx  # pointer to B[m, k_idx]
            b = tl.load(b_row_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]
            contrib = tl.sum(a * b, axis=0)  # scalar
            out_vec[m] += contrib
    # Store the result c_vec to C_ptr (row 0, columns 0:M)
    for m in range(0, M):
        tl.store(C_ptr + m, out_vec[m])


# Compute per-head lse and attn from logits_scaled (vector of length M)
# We assume logits_scaled is stored as a [1, M] tensor (we index [0, m]).
@triton.jit
def softmax_lse_kernel(
    x_ptr,           # *float32, pointer to logits_scaled vector (shape [1, M])
    out_lse_ptr,     # *float32, pointer to lse output (shape [1])
    attn_ptr,        # *float32, pointer to attn output (shape [1, M])
    M: tl.constexpr, # int: length of vector (compile-time)
    sm_scale,        # float32: scaling factor
):
    # Load x as a vector of length M
    idx = tl.arange(0, M)
    x = tl.load(x_ptr + idx)  # [M]
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)  # scalar
    # Compute sum of exp((x - max) * sm_scale)
    shifted = x - max_val
    scaled = shifted * sm_scale
    exps = tl.exp(scaled)
    sum_exp = tl.sum(exps, axis=0)  # scalar
    # Compute lse: log(sum_exp) / ln(2)
    ln2 = 1.4426950408889634  # 1 / log(2)
    lse_val = tl.log(sum_exp) / ln2  # scalar
    tl.store(out_lse_ptr, lse_val)
    # Compute attention vector
    attn = exps / sum_exp  # [M]
    # Store attn into attn_ptr [1, M] at row 0
    for i in range(0, M):
        tl.store(attn_ptr + i, attn[i])


# Compute out = attn @ Kc (single-row matvec): A: [1, M], B: [M, Kc_dim] -> C: [1, Kc_dim]
@triton.jit
def matvec_row_m_kernel(
    A_ptr,           # *float32, pointer to attn vector of length M (we pass [1, M] but load as [M])
    B_ptr,           # *float32, pointer to Kc matrix of shape [M, Kc_dim]
    C_ptr,           # *float32, pointer to output vector of length Kc_dim (we pass [1, Kc_dim] but write to [Kc_dim])
    M: tl.constexpr, # int: length of A/B rows
    Kc_dim: tl.constexpr,  # int: Kc dimension (e.g., 512)
    BLOCK_M: tl.constexpr  # tile size along M (e.g., 64 or 128)
):
    # Accumulator for output vector of length Kc_dim
    out_vec = tl.zeros((Kc_dim,), dtype=tl.float32)
    # Tile over M
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + tl.arange(0, BLOCK_M)  # vector of M indices
        mask_m = m_idx < M
        # Load A chunk
        a = tl.load(A_ptr + m_idx, mask=mask_m, other=0.0)  # [BLOCK_M]
        # For each k in Kc_dim, accumulate dot(a, B[m_idx, k])
        for k in range(0, Kc_dim):
            # Load B[:, k] chunk for m_idx
            b_col = tl.load(B_ptr + m_idx * Kc_dim + k, mask=mask_m, other=0.0)  # [BLOCK_M]
            # out[k] += sum(a * b_col)
            out_vec[k] += tl.sum(a * b_col, axis=0)
    # Store out_vec
    for k in range(0, Kc_dim):
        tl.store(C_ptr + k, out_vec[k])


def run_triton_only(q_nope, q_pe, Kc_all, Kp_all, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    B = q_nope.shape[0]
    H = q_nope.shape[1]
    Kc_dim = Kc_all.shape[1]  # 512

    # Output and lse
    output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        M = end - start
        if M <= 0:
            lse[b, :] = 0.0
            output[b, :] = 0.0
            continue

        tok_idx = kv_indices[start:end].to(torch.int64)  # indices for this batch
        # Gather Kc_all and Kp_all
        Kc_rows = Kc_all[tok_idx].contiguous()  # [M, Kc_dim]
        Kp_rows = Kp_all[tok_idx].contiguous()  # [M, 64]

        # For each head h
        for h in range(H):
            # A1: qn [512], A2: qp [64]
            qn = q_nope[b, h, :].to(torch.float32).contiguous()  # [512]
            qp = q_pe[b, h, :].to(torch.float32).contiguous()   # [64]

            # Compute logits = qn @ Kc.T
            # Allocate C1: [1, M]
            C1 = torch.empty((1, M), dtype=torch.float32, device=device)
            K = 512
            BLOCK_K = 128  # tile size, works well for K=512
            grid = (1,)  # one program computes the entire vector
            matvec_row_kernel[grid](qn, Kc_rows, C1, K, M, BLOCK_K)

            # Compute logits += qp @ Kp.T
            C2 = torch.empty((1, M), dtype=torch.float32, device=device)
            Kp_K = 64
            BLOCK_K2 = 64  # covers Kp_K=64 in one go
            matvec_row_kernel[grid](qp, Kp_rows, C2, Kp_K, M, BLOCK_K2)

            # Sum the two: logits = C1 + C2
            logits_scaled = (C1 + C2)[:, 0] * sm_scale  # [M]
            # Compute lse and attn
            logits_scaled_t = logits_scaled.view(1, M)
            attn_t = torch.empty((1, M), dtype=torch.float32, device=device)
            lse_b_h = torch.empty((1,), dtype=torch.float32, device=device)
            softmax_lse_kernel[(1,)](logits_scaled_t, lse_b_h, attn_t, M, sm_scale)
            # lse[b, h] = lse_b_h[0]
            lse[b, h] = lse_b_h[0]
            # out[b, h, :] = attn @ Kc
            out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
            matvec_row_m_kernel[(1,)](
                attn_t[:, 0],  # A_ptr of length M
                Kc_rows,        # B: [M, Kc_dim]
                out_vec,        # C_ptr of length Kc_dim
                M,              # M is constexpr for the kernel
                Kc_dim,         # constexpr Kc_dim
                BLOCK_M=64      # tile size along M
            )
            output[b, h, :] = out_vec

    # Cast output to bfloat16 as in original
    output = output.to(torch.bfloat16)
    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on the same device
        device = q_nope.device
        q_nope = q_nope.to(device)
        q_pe = q_pe.to(device)
        ckv_cache = ckv_cache.to(device)
        kpe_cache = kpe_cache.to(device)
        kv_indptr = kv_indptr.to(device)
        kv_indices = kv_indices.to(device)

        # Kc_all, Kp_all are [N, Kc_dim] and [N, 64] respectively; squeeze the batch dim
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        # Run Triton-only implementation
        out, lse = run_triton_only(q_nope, q_pe, Kc_all, Kp_all, kv_indptr, kv_indices, sm_scale)
        return out, lse


def run(*args):
    return ModelNew()(*args)
