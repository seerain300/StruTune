import math
import torch

# Triton kernels: compute lse and output vectors per (b, h).
# Note: These kernels are invoked from ModelNew.forward; no recursion or "run" function is used.

@triton.jit
def _compute_lse_and_output_kernel(
    qnh_ptr,          # [512] vector for q_nope[b, h, :]
    qph_ptr,          # [64]  vector for q_pe[b, h, :]
    Kc_ptr,           # [L_tokens * 512] contiguous
    Kp_ptr,           # [L_tokens * 64] contiguous
    out_vec_ptr,      # [512] output vector for this head
    lse_ptr,          # scalar lse for this head
    L_TOKENS: tl.constexpr,      # number of tokens, compile-time constant for unrolling
    sm_scale,         # float32 scale
):
    # Local registers
    # Initialize log-sum-exp variables
    m = -1e20  # large negative
    s = 0.0

    # Compute logits and accumulate lse
    for t in tl.static_range(L_TOKENS):
        # Kc row base offset = t * 512
        Kc_row = Kc_ptr + t * 512
        Kp_row = Kp_ptr + t * 64

        # Dot products: qnh @ Kc_selected[t] and qph @ Kp_selected[t]
        dot1 = 0.0
        dot2 = 0.0

        # Reduce over Kc dimension (512)
        for i in tl.static_range(0, 512):
            k = tl.load(Kc_row + i)
            dot1 += qnh_ptr[i] * k

        # Reduce over Kp dimension (64)
        for j in tl.static_range(0, 64):
            kp = tl.load(Kp_row + j)
            dot2 += qph_ptr[j] * kp

        logits = dot1 + dot2
        logits_scaled = logits * sm_scale

        m_new = tl.maximum(m, logits_scaled)
        s = s * tl.exp(m - m_new) + tl.exp(m_scaled - m_new)  # sum exp shifted
        m = m_new
        m_scaled = m * 1.4426950408889634  # log2(e)

    # Now s is sum exp(logits_scaled), m is max, compute lse
    total = tl.log(s) + m_scaled  # logsumexp base 2
    tl.store(lse_ptr, total)

    # Compute and store output vector: out_vec = sum_t attn[t] * Kc_selected[t]
    # attn[t] = exp(logits_scaled[t] - lse) / sum_u exp(logits_scaled[u] - lse)
    total_exp = tl.exp(m_scaled - m)  # not used directly; we can recompute per t

    for t in tl.static_range(L_TOKENS):
        Kc_row = Kc_ptr + t * 512
        Kp_row = Kp_ptr + t * 64

        # Recompute logits_scaled for t (cheap since L_TOKENS is small)
        dot1 = 0.0
        dot2 = 0.0
        for i in tl.static_range(0, 512):
            k = tl.load(Kc_row + i)
            dot1 += qnh_ptr[i] * k
        for j in tl.static_range(0, 64):
            kp = tl.load(Kp_row + j)
            dot2 += qph_ptr[j] * kp
        logits = dot1 + dot2
        logits_scaled = logits * sm_scale

        # attn = exp(logits_scaled - lse) / sum(exp(...))
        num = tl.exp(logits_scaled - total)
        denom = 0.0
        for u in tl.static_range(L_TOKENS):
            Kcu_row = Kc_ptr + u * 512
            Kpu_row = Kp_ptr + u * 64

            dot1u = 0.0
            dot2u = 0.0
            for ii in tl.static_range(0, 512):
                k = tl.load(Kcu_row + ii)
                dot1u += qnh_ptr[ii] * k
            for jj in tl.static_range(0, 64):
                kp = tl.load(Kpu_row + jj)
                dot2u += qph_ptr[jj] * kp
            logitsu = dot1u + dot2u
            logits_scaled_u = logitsu * sm_scale
            denom += tl.exp(logits_scaled_u - total)

        attn_t = num / denom

        # Accumulate into out_vec
        for i in tl.static_range(0, 512):
            ki = tl.load(Kc_row + i)
            out_vec_ptr[i] += attn_t * ki


@triton.jit
def _compute_only_output_kernel(
    qnh_ptr,          # [512]
    qph_ptr,          # [64]
    Kc_ptr,           # [L_tokens * 512]
    Kp_ptr,           # [L_tokens * 64]
    out_vec_ptr,      # [512]
    lse,              # scalar lse (float32)
    L_TOKENS: tl.constexpr,
    sm_scale,         # float32
):
    # Compute attn and accumulate output without recomputing lse
    total = lse * 1.4426950408889634  # convert base-2 lse back to natural units for exponent subtraction
    total_exp = tl.exp(total)  # not directly used

    for t in tl.static_range(L_TOKENS):
        Kc_row = Kc_ptr + t * 512
        Kp_row = Kp_ptr + t * 64

        dot1 = 0.0
        dot2 = 0.0
        for i in tl.static_range(0, 512):
            k = tl.load(Kc_row + i)
            dot1 += qnh_ptr[i] * k
        for j in tl.static_range(0, 64):
            kp = tl.load(Kp_row + j)
            dot2 += qph_ptr[j] * kp
        logits_scaled = (dot1 + dot2) * sm_scale

        num = tl.exp(logits_scaled - lse)
        denom = 0.0
        for u in tl.static_range(L_TOKENS):
            Kcu_row = Kc_ptr + u * 512
            Kpu_row = Kp_ptr + u * 64

            dot1u = 0.0
            dot2u = 0.0
            for ii in tl.static_range(0, 512):
                k = tl.load(Kcu_row + ii)
                dot1u += qnh_ptr[ii] * k
            for jj in tl.static_range(0, 64):
                kp = tl.load(Kpu_row + jj)
                dot2u += qph_ptr[jj] * kp
            logits_scaled_u = (dot1u + dot2u) * sm_scale
            denom += tl.exp(logits_scaled_u - lse)

        attn_t = num / denom

        for i in tl.static_range(0, 512):
            ki = tl.load(Kc_row + i)
            out_vec_ptr[i] += attn_t * ki


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # If inputs are not CUDA, move them to CUDA; Triton requires CUDA tensors
        if not q_nope.is_cuda:
            q_nope = q_nope.to('cuda')
        if not q_pe.is_cuda:
            q_pe = q_pe.to('cuda')
        if not ckv_cache.is_cuda:
            ckv_cache = ckv_cache.to('cuda')
        if not kpe_cache.is_cuda:
            kpe_cache = kpe_cache.to('cuda')
        if not kv_indptr.is_cuda:
            kv_indptr = kv_indptr.to('cuda')
        if not kv_indices.is_cuda:
            kv_indices = kv_indices.to('cuda')

        # Shapes
        B, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == 16, "Expected 16 heads"
        assert head_dim_ckv == 512, "Expected 512 dim for ckv"
        _, _, head_dim_kpe = q_pe.shape
        assert head_dim_kpe == 64, "Expected 64 dim for kpe"
        num_pages = ckv_cache.shape[0]
        assert kpe_cache.shape[0] == num_pages, "kpe_cache and ckv_cache must have same num_pages"

        # Initialize outputs
        output = torch.zeros((B, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)
        lse = torch.full((B, num_qo_heads), -float('inf'), dtype=torch.float32, device=q_nope.device)

        # Prepare working tensors: cast to float32 for compute
        q_nope_f = q_nope.to(torch.float32)
        q_pe_f = q_pe.to(torch.float32)

        # Check safety: avoid out-of-bounds when there are no tokens
        # len_indptr = kv_indptr.numel() should be batch_size + 1
        batch_size = B
        len_indptr = kv_indptr.numel()
        if len_indptr <= 1 or kv_indices.numel() == 0:
            # No tokens to process; return zeros as per original behavior
            output.zero_()
            lse.zero_()
            # Cast output back to bfloat16 to match original interface
            return output.to(torch.bfloat16), lse

        # Compute L_tokens per batch
        L_tokens = int(kv_indptr[batch_size].item() - kv_indptr[0].item())
        if L_tokens == 0:
            # Edge case: L_tokens == 0, but len_indptr > 1 implies kv_indptr[1] == kv_indptr[0], which shouldn't happen here.
            # To be safe, return zeros.
            output.zero_()
            lse.zero_()
            return output.to(torch.bfloat16), lse

        # Gather selected tokens from cache
        # We flatten kv_indices and process per b by using Kc_all = ckv_cache.squeeze(1), Kp_all = kpe_cache.squeeze(1)
        # but since we need tok_idx mapping, do selection explicitly:
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Build flattened Kc_selected and Kp_selected by concatenating rows corresponding to each token index
        # However, Triton kernels expect contiguous 1D pointers. We can construct views and pass base pointers per t.
        # To do so efficiently, create two 1D contiguous buffers for Kc and Kp for the selected tokens:
        # Since L_tokens can be large, allocate and fill on host. Triton will read contiguous chunks per t.

        # Allocate 1D buffers for Kc_selected and Kp_selected (contiguous flattened rows)
        Kc_selected = torch.empty(L_tokens * 512, dtype=torch.float32, device=q_nope.device)
        Kp_selected = torch.empty(L_tokens * 64, dtype=torch.float32, device=q_nope.device)

        # Fill Kc_selected and Kp_selected using kv_indices slice for each batch
        # Note: In general, we need per-b slice. The indptr is per batch. We use b=0 to compute L_tokens; since len_indptr==batch_size+1, we can assume one slice per forward call. The original setup suggests per-b handling; we can compute per b. But here we have one forward call; we'll compute per b.

        # Compute per-batch selected indices
        # Given len_indptr==batch_size+1, per-b tokens are from kv_indptr[b] to kv_indptr[b+1]
        for b in range(B):
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]  # indices for this batch b
            # Load selected rows
            # We need to construct Kc_selected for this b and h is implicit since heads=16; the forward iterates h=0..15.
            # However, our kernel handles one head at a time. We compute qnh and qph per head.

            # For each head h, we run the kernel. To simplify, we run B*16 loops in Python:
            # We'll compute qnh and qph per h in a loop and invoke kernel. But to stay in Triton only, we implement a grid over (B, 16).
            # However Triton requires scalar arguments; we'll pass each b,h via separate invocations in Python.
            pass  # placeholder; we'll implement Triton calls per (b, h) below


        # Since we cannot easily implement a 2D grid with dynamic loops, we will implement per (b, h) in Python,
        # but keep Triton math in the kernels. We launch one kernel per (b, h) combination.

        # Launch Triton per head
        for h in range(num_qo_heads):
            # Prepare pointers for this head: qnh and qph
            qnh = q_nope_f[b, h, :]
            qph = q_pe_f[b, h, :]

            # Prepare Kc_selected and Kp_selected for this batch b and head h
            # We need to gather rows using tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
            # Construct Kc_selected and Kp_selected for this b
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]
            # Gather rows from Kc_all and Kp_all
            Kc_selected_b = Kc_all[tok_idx]  # [L_tokens, 512], contiguous per row
            Kp_selected_b = Kp_all[tok_idx]  # [L_tokens, 64]

            # Flatten to 1D contiguous for kernel
            Kc_selected_b_flat = Kc_selected_b.contiguous().view(-1)
            Kp_selected_b_flat = Kp_selected_b.contiguous().view(-1)

            # Output vector for this head
            out_vec = torch.zeros((head_dim_ckv,), dtype=torch.float32, device=q_nope.device)

            # Launch Triton kernel to compute lse and output
            # We need to pass L_TOKENS as constexpr; Triton requires tl.constexpr. We can use a lambda to set grid and constexpr meta.
            grid = (1,)
            _compute_lse_and_output_kernel[grid](
                qnh, qph, Kc_selected_b_flat, Kp_selected_b_flat, out_vec, lse[b, h],
                L_TOKENS=L_tokens, sm_scale=sm_scale
            )

            # Store output
            output[b, h, :] = out_vec

        # Cast output to bfloat16 to match original interface
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse