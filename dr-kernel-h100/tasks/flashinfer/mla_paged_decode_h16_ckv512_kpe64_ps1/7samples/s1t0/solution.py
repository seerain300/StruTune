import torch
import triton
import triton.language as tl

# Matvec add kernel: computes logits[i, j] = sum_k q[i, k] * K[j, k] for i in [0, N_q), j in [0, L)
@triton.jit
def matvec_add_kernel(q_ptr, K_ptr, logits_ptr,
                       N_q: tl.int32, D: tl.int32, L: tl.int32,
                       BLOCK_K: tl.constexpr):
    i = tl.program_id(0)  # which head (or batch's head) we process
    # acc for each token j
    acc = tl.zeros([L], dtype=tl.float32)
    # Loop over K in tiles
    for k0 in range(0, D, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        # q[i, k_range]
        q_row = tl.load(q_ptr + i * D + k_range, mask=k_range < D, other=0.0)  # [BLOCK_K]
        # K[:, k_range]
        K_cols = tl.load(K_ptr + k_range, mask=k_range < D, other=0.0)  # [BLOCK_K]
        # partial dot: sum over k in block
        partial = tl.sum(q_row[:, None] * K_cols[None, :], axis=0)  # [BLOCK_K] reduced over rows -> scalar? We need vector over L tokens. Incorrect approach.
        # Correct approach: we need K's tokens, but K_ptr here is [L, D] ? We need to load per token j. Let's re-implement properly:
        # We should have q of shape [N_q, D], K of shape [L, D], output [N_q, L].
        # The above simplification is wrong. Instead, we need to load K[j, k] per j in vector form.
        # Implement proper matvec: for each j, load K[j, :] and dot with q[i, :].
        # Since Triton doesn't easily support loading a whole column, we keep a vector acc and loop over k.
        # Better: use a 2D tile across (j, k) and reduce. But simpler: do it explicitly via a loop over j then over k.
        # We'll implement a correct version with j loop:

@triton.jit
def matvec_add_kernel(q_ptr, K_ptr, logits_ptr,
                      N_q: tl.int32, D: tl.int32, L: tl.int32,
                      BLOCK_K: tl.constexpr):
    # One program per row i
    i = tl.program_id(0)
    acc = tl.zeros([L], dtype=tl.float32)
    # Reduce over K dimension in blocks
    for k0 in range(0, D, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        # q_row: [BLOCK_K]
        q_row = tl.load(q_ptr + i * D + k_range, mask=k_range < D, other=0.0)
        # We need K[:, k_range] but K is [L, D]. For each j in [0, L), we can load K[j, k_range] vector and accumulate.
        # Implement per j:
        for j in range(0, L):
            Kj = tl.load(K_ptr + j * D + k_range, mask=k_range < D, other=0.0)  # [BLOCK_K]
            acc[j] += tl.sum(q_row * Kj, axis=0)
    # Store acc to logits[i, :]
    tl.store(logits_ptr + i * L + tl.arange(0, L), acc, mask=tl.arange(0, L) < L)

# This kernel is correct but slow because it loops over j inside Triton per program. For small L, it's fine, but for large L it's not ideal.
# To improve, we can use a 2D tiling over (j, k) with BLOCK_J and BLOCK_K to vectorize across j tokens. However, Triton doesn't allow 2D vectorization across j and k easily without more advanced patterns.

# Given the complexity, we can instead implement efficient versions by operating on small L via per-batch host selection and rely on Triton for matvec and softmax. For simplicity and robustness, we’ll use the above kernel and accept that it's simple but not highly optimized; the provided workloads' L tokens are relatively small (e.g., 8, 108, etc.), so it will perform well enough.

# Implement a better matvec kernel: two-dimensional tiling across (j, k)
# We'll keep a single kernel with 2D tiling: BLOCK_J x BLOCK_K. Triton supports this:
@triton.jit
def matvec_add_kernel_v2(q_ptr, K_ptr, logits_ptr,
                         N_q: tl.int32, D: tl.int32, L: tl.int32,
                         BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program per row i
    i = tl.program_id(0)
    # Precompute j indices for this block
    j_base = tl.program_id(1) * BLOCK_J
    j_offsets = j_base + tl.arange(0, BLOCK_J)
    j_mask = j_offsets < L
    # Initialize acc for this block of j
    acc = tl.zeros([BLOCK_J], dtype=tl.float32)
    # Loop over K in tiles
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < D
        # Load q_row for i over k_offsets
        q_row = tl.load(q_ptr + i * D + k_offsets, mask=k_mask, other=0.0)  # [BLOCK_K]
        # Loop over j in the block and accumulate dot with K[j, k_offsets]
        for jj in range(0, BLOCK_J):
            j_idx = j_base + jj
            if j_idx < L:
                # Load K[j_idx, k_offsets]
                Kj = tl.load(K_ptr + j_idx * D + k_offsets, mask=k_mask, other=0.0)  # [BLOCK_K]
                # Accumulate dot
                acc[jj] += tl.sum(q_row * Kj, axis=0)
    # Store results for valid j
    tl.store(logits_ptr + i * L + j_offsets, acc, mask=j_mask)

# Softmax scale kernel: inputs logits [N_q, L], outputs attn [N_q, L], scale and inv_log2 scalars
@triton.jit
def softmax_scale_kernel(logits_ptr, attn_ptr,
                         N_q: tl.int32, L: tl.int32,
                         scale: tl.float32, inv_log2: tl.float32):
    i = tl.program_id(0)  # head index
    # First pass: compute row max
    max_val = -float("inf")
    for j in range(0, L):
        val = tl.load(logits_ptr + i * L + j)
        max_val = tl.maximum(max_val, val)
    # Second pass: compute sum of exp(scaled)
    sum_exp = 0.0
    for j in range(0, L):
        val = tl.load(logits_ptr + i * L + j)
        scaled = (val - max_val) * scale
        sum_exp += tl.exp(scaled)
    # Third pass: write normalized outputs
    for j in range(0, L):
        val = tl.load(logits_ptr + i * L + j)
        scaled = (val - max_val) * scale
        attn_val = tl.exp(scaled) / sum_exp
        # Optionally, divide by inv_log2 if needed; here we just store attn scaled by softmax
        tl.store(attn_ptr + i * L + j, attn_val)

# Matvec attn x Kc: inputs attn [N_q, L], Kc [L, D], outputs out [N_q, D]
@triton.jit
def matvec_attn_kc_kernel(attn_ptr, Kc_ptr, out_ptr,
                          N_q: tl.int32, D: tl.int32, L: tl.int32,
                          BLOCK_D: tl.constexpr, BLOCK_J: tl.constexpr):
    i = tl.program_id(0)
    # Precompute d indices for this block
    d_base = tl.program_id(1) * BLOCK_D
    d_offsets = d_base + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    # Reduce over tokens j in tiles
    for j0 in range(0, L, BLOCK_J):
        j_offsets = j0 + tl.arange(0, BLOCK_J)
        j_mask = j_offsets < L
        # Load attn[i, j_offsets]
        attn_vec = tl.load(attn_ptr + i * L + j_offsets, mask=j_mask, other=0.0)  # [BLOCK_J]
        # For each j in block, load Kc[j, d_offsets] and accumulate
        for jj in range(0, BLOCK_J):
            j_idx = j0 + jj
            if j_idx < L:
                Kj = tl.load(Kc_ptr + j_idx * D + d_offsets, mask=d_mask, other=0.0)  # [BLOCK_D]
                acc += attn_vec[jj] * Kj
    tl.store(out_ptr + i * D + d_offsets, acc, mask=d_mask)

# LSE row kernel: inputs logits [N_q, L], outputs lse [N_q]
@triton.jit
def lse_row_kernel(logits_ptr, lse_ptr,
                   N_q: tl.int32, L: tl.int32,
                   scale: tl.float32, inv_log2: tl.float32):
    i = tl.program_id(0)
    max_val = -float("inf")
    for j in range(0, L):
        val = tl.load(logits_ptr + i * L + j)
        max_val = tl.maximum(max_val, val)
    sum_exp = 0.0
    for j in range(0, L):
        val = tl.load(logits_ptr + i * L + j)
        scaled = (val - max_val) * scale
        sum_exp += tl.exp(scaled)
    lse = tl.log(sum_exp) * inv_log2  # logsumexp in base 2
    tl.store(lse_ptr + i, lse)

class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Ensure dtype and device for computation
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        # Squeeze "page" dimension, convert to float32
        Kc_all_f32 = ckv_cache.squeeze(1).to(torch.float32)
        Kp_all_f32 = kpe_cache.squeeze(1).to(torch.float32)

        batch_size = q_nope_f32.shape[0]
        num_qo_heads = q_nope_f32.shape[1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        head_dim_ckv = q_nope_f32.shape[2]
        head_dim_kpe = q_pe_f32.shape[2]
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Prepare outputs
        output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        inv_log2 = 1.0 / math.log(2.0)

        for b in range(batch_size):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg
            if L <= 0:
                # No tokens for this batch, skip
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # indices into cache
            # Gather Kc and Kp for this batch
            Kc = Kc_all_f32[tok_idx]  # [L, 512], float32
            Kp = Kp_all_f32[tok_idx]  # [L, 64], float32

            # Initialize logits for sum: two parts from q_nope and q_pe
            # We'll compute them via Triton kernels
            # Allocate logits buffers
            logits0 = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)  # from qn @ Kc.T
            logits1 = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)  # from qp @ Kp.T

            # Kernel launch: matvec_add_v2 for both, grid = (N_q, num_tiles_j). Choose BLOCK_J = 64, BLOCK_K = 128.
            N_q = num_qo_heads
            D_qn = head_dim_ckv
            D_qp = head_dim_kpe
            BLOCK_J = 64
            BLOCK_K = 128

            grid0 = (N_q, triton.cdiv(L, BLOCK_J))
            matvec_add_kernel_v2[grid0](q_nope_f32[b], Kc, logits0, N_q, D_qn, L, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K)

            grid1 = (N_q, triton.cdiv(L, BLOCK_J))
            matvec_add_kernel_v2[grid1](q_pe_f32[b], Kp, logits1, N_q, D_qp, L, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K)

            logits = logits0 + logits1  # [16, L]

            # Softmax on logits scaled by sm_scale
            attn = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
            grid_soft = (N_q,)
            softmax_scale_kernel[grid_soft](logits, attn, N_q, L, sm_scale, inv_log2)

            # Compute output = attn @ Kc
            out_vec = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            BLOCK_D = 128
            grid_out = (N_q, triton.cdiv(head_dim_ckv, BLOCK_D))
            matvec_attn_kc_kernel[grid_out](attn, Kc, out_vec, N_q, head_dim_ckv, L, BLOCK_D=BLOCK_D, BLOCK_J=BLOCK_J)

            # Store output in bfloat16
            output[b] = out_vec.to(torch.bfloat16)

            # Compute lse per batch: logsumexp(logits * sm_scale) / ln(2)
            # We can use a Triton kernel for row-wise lse. We pass logits_scaled = logits * sm_scale.
            logits_scaled = logits * sm_scale
            lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            grid_lse = (N_q,)
            lse_row_kernel[grid_lse](logits_scaled, lse_row, N_q, L, sm_scale, inv_log2)
            # Accumulate into lse[b, :]
            lse[b] = lse_row  # each row's lse per head

        return output, lse


def run(*args):
    return ModelNew()(*args)
