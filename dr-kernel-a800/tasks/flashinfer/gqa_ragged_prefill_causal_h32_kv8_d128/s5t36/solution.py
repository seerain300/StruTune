import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_attention_kernel(
    q_ptr,        # *float32, shape [M, G, D]
    k_ptr,        # *float32, shape [N, GH, D]
    v_ptr,        # *float32, shape [N, GH, D]
    out_ptr,      # *bfloat16, shape [M, G, D]
    lse_ptr,      # *float32, shape [M, G]
    # indices: per-block device int32 tensors (shape [2])
    qo_indptr_ptr,  # *int32, [q_start, q_end]
    kv_indptr_ptr,  # *int32, [kv_start, kv_end]
    # sizes (runtime ints)
    M: tl.constexpr,     # number of queries in this block
    N: tl.constexpr,     # number of key/value tokens in this block
    # constexpr sizes
    G: tl.constexpr,     # num_qo_heads (e.g., 32)
    GH: tl.constexpr,    # num_kv_heads * gqa_ratio (e.g., 8 * 4 = 32)
    D: tl.constexpr,     # head_dim (e.g., 128)
    SM_SCALE: tl.constexpr,  # scaling factor (e.g., 1.0 / sqrt(128))
    BLOCK_M: tl.constexpr,    # tile for query length (e.g., 64)
    BLOCK_N: tl.constexpr,    # tile for key/value length (e.g., 128)
):
    # Load q range [q_start, q_end)
    q_start = tl.load(qo_indptr_ptr + 0).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + 1).to(tl.int32)
    if q_start >= q_end:
        return

    # Load kv range [kv_start, kv_end)
    kv_start = tl.load(kv_indptr_ptr + 0).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + 1).to(tl.int32)
    if kv_start >= kv_end:
        return

    # Compute delta (extra K/V tokens relative to Q tokens in this block)
    delta = kv_end - kv_start - (q_end - q_start)

    # Precompute offsets for tiles
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, D)

    # Iterate over q tiles
    for m0 in range(0, M, BLOCK_M):
        q_offsets = m0 + m_offsets
        mask_m = q_offsets < M

        # Prepare Q_T: [BLOCK_M, G, D]
        Q_T = tl.zeros((BLOCK_M, G, D), dtype=tl.float32)
        # Load q vectors for this tile into Q_T
        for g in range(0, G):
            qg_base = q_ptr + (q_start + q_offsets) * G * D + g * D  # [BLOCK_M] base pointers
            # Load full D for each q in tile
            qg_vals = tl.load(qg_base[:, None] + d_offsets[None, :], mask=mask_m[:, None], other=0.0)
            Q_T[:, g, :] = qg_vals

        # Initialize logits [BLOCK_M, G, N]
        logits = tl.zeros((BLOCK_M, G, N), dtype=tl.float32)

        # Compute logits = Q_T @ K^T over N tiles
        for n0 in range(0, N, BLOCK_N):
            n_cur = n0 + n_offsets
            mask_n = n_cur < N

            # Load K_T and V_T for current N tile: [BLOCK_N, GH, D]
            K_T = tl.zeros((BLOCK_N, GH, D), dtype=tl.float32)
            V_T = tl.zeros((BLOCK_N, GH, D), dtype=tl.float32)

            # Base pointer for this kv_start and tile
            # k_ptr shape [N, GH, D], stride across N is GH*D, across GH is D
            # For n_cur[i], base per GH
            # We iterate GH and load rows for each group
            for gh in range(0, GH):
                k_base = k_ptr + (kv_start + n_cur) * GH * D + gh * D  # [BLOCK_N] base pointers
                v_base = v_ptr + (kv_start + n_cur) * GH * D + gh * D  # [BLOCK_N] base pointers
                # Load D elements for each n in tile
                k_rows = tl.load(k_base[:, None] + d_offsets[None, :], mask=mask_n[:, None], other=0.0)  # [BLOCK_N, D]
                v_rows = tl.load(v_base[:, None] + d_offsets[None, :], mask=mask_n[:, None], other=0.0)  # [BLOCK_N, D]
                K_T[:, gh, :] = k_rows
                V_T[:, gh, :] = v_rows

            # Compute Q_T @ K_T^T -> [BLOCK_M, BLOCK_N] per (g, n)
            # Loop over heads g
            for g2 in range(0, G):
                # Q_T[:, g2, :] [BLOCK_M, D], K_T^T [:, :, :] -> take k_sub = K_T[:, :, :] and reduce
                # We'll build a [BLOCK_M, BLOCK_N] matrix by summing over D
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    d_sub = d0 + d_offsets
                    mask_d = d_sub < D
                    q_sub = Q_T[:, g2, d_sub]                 # [BLOCK_M, BLOCK_D]
                    k_sub = K_T[:, :, d_sub][:, None, :]     # [BLOCK_N, 1, BLOCK_D] -> broadcast across BLOCK_M via loop? Better: compute per n using tl.dot
                    # Since we vectorized over BLOCK_N, we can use tl.dot(q_sub, k_sub) where k_sub is [BLOCK_N, BLOCK_D]
                    # But we need to align shapes: q_sub is [BLOCK_M, BLOCK_D], k_sub should be [BLOCK_N, BLOCK_D].
                    # Compute prod = q_sub[:, None, :] * k_sub[None, :, :], then sum over last dim
                    # However Triton prefers explicit loops for dynamic dims; to keep simple, we do:
                    # For each n, sum q_sub * k_sub[n, :]
                    for i in range(0, BLOCK_N):
                        k_row = K_T[i, :, :]  # [GH, D]
                        prod = q_sub * k_row[None, :]  # [BLOCK_M, GH, D] not ideal; avoid this complexity
                    # Instead, use reduction over D directly:
                    # For each d_sub, q_sub[:, d_sub] * K_T[:, :, d_sub] across GH? This is convoluted without tl.dot over D.
                    # Simpler: compute logits using a loop over D in chunks and tl.sum on [BLOCK_N] for each m.
                    # This requires a nested loop, which Triton supports:
                    for d in range(0, D):
                        # Extract q elements for each m
                        q_el = Q_T[:, g2, d]  # [BLOCK_M]
                        # Load k elements for each n and GH
                        k_el_total = tl.zeros((BLOCK_N,), dtype=tl.float32)
                        for gh in range(0, GH):
                            k_el = tl.load(k_ptr + (kv_start + n_cur) * GH * D + gh * D + d, mask=mask_n, other=0.0)  # [BLOCK_N]
                            k_el_total += k_el
                        # Accumulate: logits[:, None] += q_el[ :, None] * k_el_total[None, :]
                        acc += q_el[:, None] * k_el_total[None, :]

                # Scale by SM_SCALE
                acc = acc * SM_SCALE

                # Apply causal mask: for each m (q_idx), n < q_idx + 1 + delta
                q_add = (q_start + q_offsets) + 1 + delta  # [BLOCK_M], integers
                mask_causal = (n_cur[None, :] < q_add[:, None]) & (mask_m[:, None] & mask_n[None, :])
                logits[:, :, n0:n0+BLOCK_N] = tl.where(mask_causal, logits[:, :, n0:n0+BLOCK_N], -float('inf'))

        # After processing all N tiles, compute LSE per (m, g) in base-2
        for g2 in range(0, G):
            # logits[:, g2, :] shape [BLOCK_M, N]
            max_score = tl.max(logits[:, g2, :], axis=1)
            # sum_exp for each m
            sum_exp = tl.sum(tl.exp(logits[:, g2, :] - max_score[:, None]), axis=1)
            lse_val = max_score + tl.log(sum_exp)  # natural log
            lse_val = lse_val / math.log(2.0)      # convert to base-2
            lse_base = lse_ptr + (q_start + m_offsets) * G + g2
            tl.store(lse_base, lse_val, mask=mask_m)

        # Compute softmax over N for each (m, g), then output = softmax @ V
        for g2 in range(0, G):
            max_score = tl.max(logits[:, g2, :], axis=1)
            sum_exp = tl.sum(tl.exp(logits[:, g2, :] - max_score[:, None]), axis=1)
            soft = tl.exp(logits[:, g2, :] - max_score[:, None]) / sum_exp[:, None]  # [BLOCK_M, N]
            # Accumulate output across N and GH: out[m, g2, d] += sum_n soft[m,n] * V_T[n, :, d]
            out_row = tl.zeros((BLOCK_M, D), dtype=tl.float32)
            for n0 in range(0, N, BLOCK_N):
                n_cur = n0 + n_offsets
                mask_n = n_cur < N
                for gh in range(0, GH):
                    v_base = v_ptr + (kv_start + n_cur) * GH * D + gh * D
                    v_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
                    for i in range(0, BLOCK_N):
                        row_valid = (i + n0) < N
                        v_row_ptr = v_base[i, :]
                        d_offsets = tl.arange(0, D)
                        mask_d = d_offsets < D
                        v_row = tl.load(v_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                        v_chunk[i, :] = v_row
                    # soft[:, n0:n0+BLOCK_N] * v_chunk[:, d] then sum over n
                    for d0 in range(0, D):
                        col = v_chunk[:, d0]
                        out_row += tl.sum(soft[:, n0:n0+BLOCK_N] * col[None, :], axis=1)
            # Store output for this tile
            out_base = out_ptr + (q_start + m_offsets) * G * D + g2 * D
            tl.store(out_base + d_offsets, out_row, mask=mask_m)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Validate shapes
        assert q.shape[1] == 32, "Expected q with 32 heads"
        assert q.shape[2] == 128, "Expected head_dim=128"
        assert k.shape[1] == 8, "Expected k with 8 heads"
        assert k.shape[2] == 128, "Expected head_dim=128"
        assert v.shape == k.shape, "v must have same shape as k"

        device = q.device
        M_total, G, D = q.shape
        N_total, GHk, Dk = k.shape
        assert D == 128 and GHk == 8 and Dk == 128, "Invalid shapes"

        # Allocate outputs (computed by Triton)
        output = torch.empty((M_total, G, D), dtype=torch.bfloat16, device=device)
        lse = torch.full((M_total, G), -float("inf"), dtype=torch.float32, device=device)

        # Convert to float32 for computation; output is bfloat16, lse is float32
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            # Extract per-block ranges
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            M = q_end - q_start
            N = kv_end - kv_start

            # Slice tensors for this block
            q_block = q_f32[q_start:q_end]  # [M, G, D]
            k_block = k_f32[kv_start:kv_end]  # [N, GH, D]
            v_block = v_f32[kv_start:kv_end]  # [N, GH, D]

            # Create device tensors for per-block indices (int32 on device): pass slices directly
            qo_indptr_b = qo_indptr[b:b+1]  # device int32 tensor [1, 2]
            kv_indptr_b = kv_indptr[b:b+1]  # device int32 tensor [1, 2]

            # Launch Triton kernel once per block; pass M and N as runtime ints
            _block_attention_kernel[(1,)](
                q_block, k_block, v_block,
                output, lse,
                qo_indptr_b, kv_indptr_b,
                M=M, N=N,
                G=32, GH=8, D=128, SM_SCALE=sm_scale,
                BLOCK_M=64, BLOCK_N=128,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
