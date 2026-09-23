import torch
import triton
import triton.language as tl


@triton.jit
def _batch_work_kernel(
    qn_ptr,        # [H, CK], base pointer
    qp_ptr,        # [H, KP], base pointer
    Kc_all_ptr,    # [P, CK], base pointer
    Kp_all_ptr,    # [P, KP], base pointer
    tok_idx_ptr,   # [L_tokens], int32
    out_ptr,       # [H, CK], float32, base pointer (we'll cast outside)
    lse_ptr,       # [H], float32, base pointer
    # sizes
    H: tl.constexpr,          # num_qo_heads (e.g., 16)
    CK: tl.constexpr,         # head_dim_ckv (e.g., 512)
    KP: tl.constexpr,         # head_dim_kpe (e.g., 64)
    L_tokens: tl.constexpr,   # number of tokens for this batch
    sm_scale: tl.constexpr,   # scaling factor
):
    # We process one batch element per launch; H, CK, KP are constexpr.

    # Prepare dim offsets for qn/qp and out rows
    dim_ck = tl.arange(0, CK)
    dim_kp = tl.arange(0, KP)

    # Compute per-head outputs and lse using two passes: first for lse, second for accumulation.
    # We will vectorize across H (constexpr) and loop over tokens.

    # Initialize per-head accumulators for output and lse
    lse_vec = tl.full((H,), -1.0e30, tl.float32)  # per-head lse
    out_accum = tl.zeros((H, CK), tl.float32)     # per-head output accumulator

    # Pass 1: compute lse per head via dynamic loop over tokens (H is constexpr, Triton allows this)
    for h in range(H):
        # Track max for logsumexp
        max_val = tl.full((), -1.0e30, tl.float32)
        sum_exp = tl.zeros((), tl.float32)

        for t in range(L_tokens):
            idx = tl.load(tok_idx_ptr + t)  # int32 index into Kc_all/Kp_all
            # Load Kc_row and Kp_row
            Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)   # [CK]
            Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)   # [KP]

            # Load qn[h, :] and qp[h, :]
            qn_vec = tl.load(qn_ptr + h * CK + dim_ck)         # [CK]
            qp_vec = tl.load(qp_ptr + h * KP + dim_kp)         # [KP]

            # Compute scaled logits for this head and token
            dot_qn = tl.sum(qn_vec * Kc_row)                   # scalar
            dot_qp = tl.sum(qp_vec * Kp_row)                   # scalar
            scaled = (dot_qn + dot_qp) * sm_scale              # scalar

            # Update max and sum for logsumexp
            max_val = tl.maximum(max_val, scaled)
            sum_exp += tl.exp(scaled - max_val)

        # lse = logsumexp(scaled) / ln(2)
        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val / tl.log(2.0)
        tl.store(lse_ptr + h, lse_val)

    # Pass 2: compute output for each head by accumulating softmax-weighted Kc_rows
    for h in range(H):
        # Reinitialize output accumulator for this head
        out_accum[h, :] = tl.zeros((CK,), tl.float32)

        # We recompute scaled per t and accumulate directly: out[h, :] = sum_t exp(scaled[h, t] - lse[h]) * Kc[t, :]
        # We need lse[h]; it was just computed above. We'll recompute scaled in this loop as well.
        # But to avoid recomputing everything again, we recompute lse_vec from scratch here; however,
        # Triton allows reusing the same kernel structure. We'll recompute lse_vec here to have it available.
        # Compute lse again for h (recompute): Store into lse_ptr[h] isn't needed; we only need lse_val scalar.
        # We can compute and store out without recomputing: We computed lse per head above; we don't need to.

        # Instead, we compute out directly: recompute scaled per t and accumulate.
        lse_h = tl.load(lse_ptr + h)  # scalar

        for t in range(L_tokens):
            idx = tl.load(tok_idx_ptr + t)
            Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
            qn_vec = tl.load(qn_ptr + h * CK + dim_ck)        # [CK]
            qp_vec = tl.load(qp_ptr + h * KP + dim_kp)        # [KP]
            dot_qn = tl.sum(qn_vec * Kc_row)                  # scalar
            dot_qp = tl.sum(qp_vec * tl.zeros((KP,), tl.float32))  # placeholder; Kp not needed here

            # Note: We don't have Kp_vec here; we can set dot_qp=0 since it's not needed for out accumulation.
            # This is incorrect; fix below by loading Kp_row and computing dot_qp.
            Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]
            dot_qp = tl.sum(qp_vec * Kp_row)                  # scalar
            scaled = (dot_qn + dot_qp) * sm_scale
            soft = tl.exp(scaled - lse_h)
            out_accum[h, :] += soft * Kc_row

        # Store output for this head
        tl.store(out_ptr + h * CK + dim_ck, out_accum[h, :])

# Note: The above kernel assumes H, CK, KP are constexpr and L_tokens is provided as a constexpr.
# Triton will specialize the kernel for the given shapes. We launch one program per batch element.


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are CUDA and contiguity
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton."
        device = q_nope.device

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages_ckv, ckv_cache_d, _ = ckv_cache.shape
        num_pages_kpe, kpe_cache_d, _ = kpe_cache.shape
        assert ckv_cache_d == head_dim_ckv and kpe_cache_d == head_dim_kpe
        assert num_pages_ckv == num_pages_kpe, "ckv_cache and kpe_cache must have same first dimension."

        # Prepare outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Convert caches to float32 for compute; Triton will read float32
        Kc_all = ckv_cache.to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.to(torch.float32).contiguous()  # [num_pages, 64]

        # Flatten q_nope and q_pe to [B*H, D] for pointer slicing convenience
        qn_flat = q_nope.to(torch.float32).contiguous().view(batch_size * num_qo_heads, head_dim_ckv)  # [B*H, CK]
        qp_flat = q_pe.to(torch.float32).contiguous().view(batch_size * num_qo_heads, head_dim_kpe)  # [B*H, KP]

        for b in range(batch_size):
            # Compute L_tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).contiguous()  # [L_tokens]

            # Base pointers for qn and qp for this batch
            qn_batch_ptr = qn_flat[b * num_qo_heads: (b + 1) * num_qo_heads]  # [H, CK]
            qp_batch_ptr = qp_flat[b * num_qo_heads: (b + 1) * num_qo_heads]  # [H, KP]

            # Launch Triton kernel for this batch element: grid=(1,) — one program handles all H and tokens
            _batch_work_kernel[(1,)](
                qn_batch_ptr, qp_batch_ptr,
                Kc_all, Kp_all,
                tok_idx,
                output[b], lse[b],
                H=num_qo_heads,
                CK=head_dim_ckv,
                KP=head_dim_kpe,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
                num_warps=4,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
