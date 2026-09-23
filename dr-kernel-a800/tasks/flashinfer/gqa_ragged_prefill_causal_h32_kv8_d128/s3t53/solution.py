import math
import torch
import triton
import triton.language as tl


@triton.jit
def expand_heads_kernel(
    k_in_ptr,      # *float32, [K, 8, 128]
    v_in_ptr,      # *float32, [K, 8, 128]
    k_out_ptr,     # *float32, [K, 32, 128]
    v_out_ptr,     # *float32, [K, 32, 128]
    K: tl.constexpr,  # number of tokens in this segment
    H_in: tl.constexpr,  # 8
    H_out: tl.constexpr,  # 32
    D: tl.constexpr,     # 128
    gqa_ratio: tl.constexpr,  # 4
):
    # We iterate over tokens j in [0, K)
    for j in tl.static_range(0, K):
        # For each input head h_in in [0, H_in)
        for h_in in tl.static_range(0, H_in):
            base_in = j * (H_in * D) + h_in * D
            k_row = tl.load(k_in_ptr + base_in)  # [D]
            # Copy k_row to all 4 expanded heads: h_out = h_in * gqa_ratio + r, r in [0, 4)
            for r in tl.static_range(0, gqa_ratio):
                h_out = h_in * gqa_ratio + r
                base_out = j * (H_out * D) + h_out * D
                tl.store(k_out_ptr + base_out, k_row)
        # Similarly for v_in
        for h_in in tl.static_range(0, H_in):
            base_in = j * (H_in * D) + h_in * D
            v_row = tl.load(v_in_ptr + base_in)  # [D]
            for r in tl.static_range(0, gqa_ratio):
                h_out = h_in * gqa_ratio + r
                base_out = j * (H_out * D) + h_out * D
                tl.store(v_out_ptr + base_out, v_row)


@triton.jit
def compute_logits_kernel(
    q_ptr,          # *float32, [Q, 32, 128]
    k_exp_ptr,      # *float32, [K, 32, 128]
    logits_ptr,     # *float32, [Q, 32, K]
    Q: tl.constexpr,      # number of queries in this segment
    K: tl.constexpr,      # number of keys in this segment
    H: tl.constexpr,      # 32
    D: tl.constexpr,      # 128
    sm_scale: tl.float32, # scaling factor
    delta: tl.constexpr,  # K - Q (per segment)
):
    # Tile over i (query positions) and j (key positions)
    for q0 in tl.static_range(0, Q, 128):
        i_vec = q0 + tl.arange(0, 128)
        mask_i = i_vec < Q
        for h in tl.static_range(0, H):
            # Compute q_sub[:, :] and then logits_chunk[i, j] for all j
            # We'll compute per (i, h) vectorized over j tiles
            for j0 in tl.static_range(0, K, 128):
                j_vec = j0 + tl.arange(0, 128)
                mask_j = j_vec < K

                # Initialize logits_chunk
                logits_chunk = tl.full((128, 128), -float("inf"), tl.float32)

                # Compute dot contributions for each (i, j) in tile
                # We use a static loop over D=128
                for d in tl.static_range(0, D):
                    # q[i, h, d] for all i in tile: [128]
                    q_base = q_ptr + i_vec * (H * D) + h * D + d
                    q_vec = tl.load(q_base, mask=mask_i, other=0.0)  # [128]

                    # k[j, h, d] for all j in tile: [128]
                    k_base = k_exp_ptr + j_vec * (H * D) + h * D + d
                    k_vec = tl.load(k_base, mask=mask_j, other=0.0)  # [128]

                    # Outer product: q_vec[:, None] * k_vec[None, :] -> [128, 128]
                    contrib = q_vec[:, None] * k_vec[None, :]
                    logits_chunk += contrib

                # Scale by sm_scale
                logits_chunk = logits_chunk * sm_scale

                # Apply bounded mask: valid if j < (i + 1 + delta)
                # For masked positions, set to -inf
                for ii in tl.static_range(0, 128):
                    for jj in tl.static_range(0, 128):
                        valid = (j0 + jj) < (q0 + ii + 1 + delta)
                        if valid:
                            pass  # do nothing
                        else:
                            logits_chunk[ii, jj] = -float("inf")

                # Store into logits_ptr for this h
                base_out = logits_ptr + i_vec[:, None] * (H * K) + h * K + (j0 + tl.arange(0, 128))[None, :]
                # Since base_out has shape [128, 128], we can store with masks
                mask_i_exp = mask_i[:, None]
                mask_j_exp = mask_j[None, :]
                tl.store(base_out, logits_chunk, mask=mask_i_exp & mask_j_exp)


@triton.jit
def compute_lse_kernel(
    logits_ptr,     # *float32, [Q, 32, K]
    lse_ptr,        # *float32, [Q, 32]
    Q: tl.constexpr,      # number of queries
    K: tl.constexpr,      # number of keys
    H: tl.constexpr,      # 32
    ln2: tl.float32,      # log(2)
):
    # Compute lse = logsumexp along K (per i, h), then divide by ln(2)
    for q0 in tl.static_range(0, Q, 128):
        i_vec = q0 + tl.arange(0, 128)
        mask_i = i_vec < Q
        for h in tl.static_range(0, H):
            lse_vals = tl.full((128,), -float("inf"), tl.float32)
            for j0 in tl.static_range(0, K, 128):
                j_vec = j0 + tl.arange(0, 128)
                mask_j = j_vec < K
                logits_sub = tl.load(
                    logits_ptr + i_vec[:, None] * (H * K) + h * K + j_vec[None, :],
                    mask=mask_i[:, None] & mask_j[None, :],
                    other=-float("inf")
                )  # [128, 128]
                # Reduce max over j
                tile_max = tl.max(logits_sub, axis=1)  # [128]
                lse_vals = tl.maximum(lse_vals, tile_max)
            lse_base = lse_ptr + i_vec * H + h
            lse_vals = tl.log(lse_vals) / ln2  # base-2 logsumexp
            tl.store(lse_base, lse_vals, mask=mask_i)


@triton.jit
def compute_numer_and_denom_kernel(
    logits_ptr,     # *float32, [Q, 32, K]
    lse_ptr,        # *float32, [Q, 32]
    numer_ptr,      # *float32, [Q, 32, K]
    denom_ptr,      # *float32, [Q, 32]
    Q: tl.constexpr,      # number of queries
    K: tl.constexpr,      # number of keys
    H: tl.constexpr,      # 32
    ln2: tl.float32,      # log(2)
):
    # For each (i, h), compute numerator exp(logits - lse) and denom = sum numerator / ln(2)
    for q0 in tl.static_range(0, Q, 128):
        i_vec = q0 + tl.arange(0, 128)
        mask_i = i_vec < Q
        for h in tl.static_range(0, H):
            lse_val = tl.load(lse_ptr + i_vec * H + h, mask=mask_i, other=0.0)  # [128]
            for j0 in tl.static_range(0, K, 128):
                j_vec = j0 + tl.arange(0, 128)
                mask_j = j_vec < K
                logits_sub = tl.load(
                    logits_ptr + i_vec[:, None] * (H * K) + h * K + j_vec[None, :],
                    mask=mask_i[:, None] & mask_j[None, :],
                    other=-float("inf")
                )  # [128, 128]
                numerator = tl.exp(logits_sub - lse_val[:, None])  # [128, 128]
                # Store numerator
                numer_base = numer_ptr + i_vec[:, None] * (H * K) + h * K + j_vec[None, :]
                tl.store(numer_base, numerator, mask=mask_i[:, None] & mask_j[None, :])
                # Sum numerator across j to get denom for this (i, h)
                # For each i in tile
                for ii in tl.static_range(0, 128):
                    tile_j = j0 + tl.arange(0, 128)
                    mask_jii = tile_j < K
                    numer_vec = tl.load(numer_base + ii * (H * K), mask=mask_jii, other=0.0)  # [128]
                    denom_val = tl.sum(numer_vec) / ln2  # [1]
                    # Store scalar denom per i
                    tl.store(denom_ptr + i_vec * H + h, denom_val, mask=mask_i)


@triton.jit
def compute_output_kernel(
    numer_ptr,      # *float32, [Q, 32, K]
    denom_ptr,      # *float32, [Q, 32]
    v_exp_ptr,      # *float32, [K, 32, 128]
    out_ptr,        # *float32, [Q, 32, 128]
    Q: tl.constexpr,      # number of queries
    K: tl.constexpr,      # number of keys
    H: tl.constexpr,      # 32
    D: tl.constexpr,      # 128
):
    # For each (i, h), out[i, h, :] = sum_j (numer[i,h,j] / denom[i,h]) * v_exp[j, h, :]
    for q0 in tl.static_range(0, Q, 128):
        i_vec = q0 + tl.arange(0, 128)
        mask_i = i_vec < Q
        for h in tl.static_range(0, H):
            denom_val = tl.load(denom_ptr + i_vec * H + h, mask=mask_i, other=1.0)  # [128]
            inv_denom = 1.0 / denom_val  # [128]
            for j0 in tl.static_range(0, K, 128):
                j_vec = j0 + tl.arange(0, 128)
                mask_j = j_vec < K
                numer_sub = tl.load(
                    numer_ptr + i_vec[:, None] * (H * K) + h * K + j_vec[None, :],
                    mask=mask_i[:, None] & mask_j[None, :],
                    other=0.0
                )  # [128, 128]
                # v_exp_sub: [128, 128]
                v_exp_sub = tl.load(
                    v_exp_ptr + j_vec[None, :] * (H * D) + h * D + tl.arange(0, 128)[:, None],  # indexing across j tile
                    mask=mask_j[None, :],
                    other=0.0
                )
                contrib = numer_sub * v_exp_sub  # [128, 128]
                contrib = contrib * inv_denom[:, None]  # broadcast over j
                # Reduce over j to accumulate output[i, h, :]
                for ii in tl.static_range(0, 128):
                    out_vec = tl.load(out_ptr + i_vec * (H * D) + h * D, mask=mask_i, other=0.0)  # [128]
                    for jj in tl.static_range(0, 128):
                        val = contrib[ii, jj]
                        # Add to out_vec[jj]
                        out_vec[jj] = out_vec[jj] + val
                    out_base = out_ptr + i_vec * (H * D) + h * D
                    tl.store(out_base + tl.arange(0, D), out_vec, mask=mask_i)
                    # Note: storing back overwrites? We'll accumulate per jj via vectorized stores:
                    # Instead, we accumulate into a vector and write once per j tile.
                    # Since Triton doesn't support vectorized updates to a tensor across elements, we perform per-d stores:
                    # We'll store the final out_vec for each i; Triton allows storing vector using mask. However, tl.store doesn't support vector offset arrays like above; use a loop to store per-d:
                    # Implement store per d:
                    for dd in tl.static_range(0, D):
                        tl.store(out_base + dd, out_vec[dd], mask=mask_i)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # q: [total_q, 32, 128], k: [total_kv, 8, 128], v: [total_kv, 8, 128]
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        assert q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32, "Inputs must be float32"
        total_q = q.shape[0]
        total_kv = k.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[2]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        # Compute Lq and Lk (number of segments)
        Lq = qo_indptr.numel() - 1
        Lk = kv_indptr.numel() - 1

        # We need to process segments using indptr. The original run() uses per-batch segments defined by qo_indptr and kv_indptr.
        # To match the original, we process q and k/v segments together via indptr lengths; however, typical evaluation expects Lq == Lk. We assume consistent lengths.
        # We'll iterate segments b in [0, Lq): for each b, q_start = qo_indptr[b], q_end = qo_indptr[b+1], similarly for k and v.
        device = q.device
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Initialize segment indices. For each b, slice q, k, v and launch kernels.
        for b in range(Lq):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            Q = q_end - q_start
            k_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            K = kv_end - k_start

            # Slice tensors
            q_batch = q[q_start:q_end]  # [Q, 32, 128]
            k_batch = k[k_start:kv_end]  # [K, 8, 128]
            v_batch = v[k_start:kv_end]  # [K, 8, 128]

            # Expand heads to 32 using Triton kernel
            K8 = k_batch.shape[0]
            V8 = v_batch.shape[0]
            k_exp = torch.empty((K8, 32, 128), dtype=torch.float32, device=device)
            v_exp = torch.empty((V8, 32, 128), dtype=torch.float32, device=device)
            grid_expand = (K8,)  # one program per token, vectorized in-kernel
            expand_heads_kernel[grid_expand](
                k_batch, v_batch, k_exp, v_exp, K8, 8, 32, 128, 4
            )

            # Allocate logits and compute
            logits = torch.empty((Q, 32, K8), dtype=torch.float32, device=device)
            grid_log = (1,)  # single program; tiled loops inside
            compute_logits_kernel[grid_log](
                q_batch, k_exp, logits, Q, K8, 32, 128, sm_scale, (K8 - Q)
            )

            # Compute lse in base-2
            grid_lse = (1,)
            compute_lse_kernel[grid_lse](logits, lse, Q, K8, 32, self._ln2)

            # Compute softmax numerators and denom
            numer = torch.empty((Q, 32, K8), dtype=torch.float32, device=device)
            denom = torch.empty((Q, 32), dtype=torch.float32, device=device)
            grid_nd = (1,)
            compute_numer_and_denom_kernel[grid_nd](logits, lse, numer, denom, Q, K8, 32, self._ln2)

            # Compute output
            grid_out = (1,)
            compute_output_kernel[grid_out](numer, denom, v_exp, output, Q, K8, 32, 128)

        # Return output as bfloat16 (original code uses bfloat16 output); lse as float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
