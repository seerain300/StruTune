import math
import torch
import triton
import triton.language as tl


@triton.jit
def per_head_kernel(
    qn_ptr,            # *float32, [D]
    qp_ptr,            # *float32, [Dp]
    Kc_ptr,            # *float32, [L, D], row-major (L, D)
    Kp_ptr,            # *float32, [L, Dp], row-major (L, Dp)
    out_ptr,           # *float32, [D]
    lse_ptr,           # *float32, [1] (scalar buffer to store logsumexp)
    L: tl.int32,       # number of tokens
    D: tl.int32,       # head_dim_ckv (512)
    Dp: tl.int32,      # head_dim_kpe (64)
    sm_scale: tl.float32,  # scaling factor for logits (default 1.0)
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # We run one program per (batch, head) and compute:
    # 1) v[i] = sum_d (qn[d] * Kc[i, d]) + sum_dp (qp[dp] * Kp[i, dp]) for i in [0, L)
    # 2) lse = logsumexp_base2(v)
    # 3) attn[i] = exp(v[i] / ln(2) - lse)
    # 4) out = sum_i attn[i] * Kc[i, :]
    # Use tiled reductions over D and L.

    # Vector for tokens index
    i = 0
    # First pass: compute max for numerical stability
    v_max = -float('inf')
    while i < L:
        # Compute logits for this tile of tokens
        logits = tl.zeros((), dtype=tl.float32)
        # Reduce over Kc dimension (D)
        k = 0
        while k < D:
            kk = k + tl.arange(0, BLOCK_K)
            mask_k = kk < D
            # Load qn slice
            qn_k = tl.load(qn_ptr + kk, mask=mask_k, other=0.0)  # [BLOCK_K]
            # Load Kc tile for tokens i
            kc_ptrs = Kc_ptr + i * D + kk
            mask_kc = mask_k
            Kc_tile = tl.load(kc_ptrs, mask=mask_kc, other=0.0)  # [BLOCK_K]
            # Accumulate dot
            logits += tl.sum(qn_k * Kc_tile, axis=0)
            k += BLOCK_K

        # Reduce over Kp dimension (Dp)
        kp = 0
        while kp < Dp:
            kp_off = kp + tl.arange(0, BLOCK_K)
            mask_kp = kp_off < Dp
            qp_k = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)  # [BLOCK_K]
            Kp_tile = tl.load(Kp_ptr + i * Dp + kp_off, mask=mask_kp, other=0.0)  # [BLOCK_K]
            logits += tl.sum(qp_k * Kp_tile, axis=0)
            kp += BLOCK_K

        # Update max
        v_max = tl.maximum(v_max, logits)

        i += 1

    # Second pass: compute sum(exp((logits - max) * sm_scale / ln(2)))
    sum_exp = 0.0
    ln2 = 0.6931471805599453
    scale = sm_scale / ln2
    i = 0
    while i < L:
        logits = tl.zeros((), dtype=tl.float32)
        # Repeat the same dot computations for logits
        k = 0
        while k < D:
            kk = k + tl.arange(0, BLOCK_K)
            mask_k = kk < D
            qn_k = tl.load(qn_ptr + kk, mask=mask_k, other=0.0)
            Kc_tile = tl.load(Kc_ptr + i * D + kk, mask=mask_k, other=0.0)
            logits += tl.sum(qn_k * Kc_tile, axis=0)
            k += BLOCK_K

        kp = 0
        while kp < Dp:
            kp_off = kp + tl.arange(0, BLOCK_K)
            mask_kp = kp_off < Dp
            qp_k = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)
            Kp_tile = tl.load(Kp_ptr + i * Dp + kp_off, mask=mask_kp, other=0.0)
            logits += tl.sum(qp_k * Kp_tile, axis=0)
            kp += BLOCK_K

        exp_val = tl.exp((logits - v_max) * scale)
        sum_exp += exp_val
        i += 1

    # Compute lse = (v_max + log(sum_exp)) * ln(2)
    lse_val = v_max + tl.log(sum_exp)
    lse_val = lse_val * ln2
    # Store lse to buffer [1]
    tl.store(lse_ptr, lse_val)

    # Third pass: compute attn and final output vector
    i = 0
    while i < L:
        logits = tl.zeros((), dtype=tl.float32)
        # Compute logits for this token
        k = 0
        while k < D:
            kk = k + tl.arange(0, BLOCK_K)
            mask_k = kk < D
            qn_k = tl.load(qn_ptr + kk, mask=mask_k, other=0.0)
            Kc_tile = tl.load(Kc_ptr + i * D + kk, mask=mask_k, other=0.0)
            logits += tl.sum(qn_k * Kc_tile, axis=0)
            k += BLOCK_K

        kp = 0
        while kp < Dp:
            kp_off = kp + tl.arange(0, BLOCK_K)
            mask_kp = kp_off < Dp
            qp_k = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)
            Kp_tile = tl.load(Kp_ptr + i * Dp + kp_off, mask=mask_kp, other=0.0)
            logits += tl.sum(qp_k * Kp_tile, axis=0)
            kp += BLOCK_K

        attn_i = tl.exp((logits - v_max) * scale - lse_val)

        # Accumulate out += attn_i * Kc[i, :]
        k = 0
        while k < D:
            kk = k + tl.arange(0, BLOCK_K)
            mask_k = kk < D
            Kc_tile = tl.load(Kc_ptr + i * D + kk, mask=mask_k, other=0.0)
            out_tile = attn_i * Kc_tile
            out_ptrs = out_ptr + kk
            tl.store(out_ptrs, out_tile, mask=mask_k)
            k += BLOCK_K

        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA
        device = q_nope.device
        # Cast to float32 for Triton compute
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        # Extract L and build Kc_all, Kp_all as in original
        # Since indptr[0] = 0, we can safely squeeze the leading dimension (num_pages)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        batch_size, num_qo_heads, head_dim_ckv = q_nope_f32.shape
        head_dim_kpe = q_pe_f32.shape[-1]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare output and lse tensors
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Iterate batches
        for b in range(batch_size):
            # Determine token range
            if kv_indptr.numel() < 2 or kv_indptr.shape[0] != batch_size + 1:
                # Fallback: no kv, output zeros
                output[b].zero_()
                lse[b] = -float('inf')
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())

            L_tokens = max(page_end - page_beg, 0)
            if L_tokens == 0:
                output[b].zero_()
                lse[b] = -float('inf')
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L_tokens]

            # Extract Kc and Kp for this batch
            Kc = Kc_all[tok_idx]  # [L_tokens, 512]
            Kp = Kp_all[tok_idx]  # [L_tokens, 64]

            # Launch Triton kernel per head
            for j in range(num_qo_heads):
                qn = q_nope_f32[b, j]    # [512]
                qp = q_pe_f32[b, j]      # [64]

                grid = (1,)
                per_head_kernel[grid](
                    qn, qp,
                    Kc, Kp,
                    output[b, j], lse[b, j],
                    L_tokens, head_dim_ckv, head_dim_kpe,
                    sm_scale,
                    BLOCK_K=128, BLOCK_L=128,
                    num_warps=4,
                )

        # Cast output to bfloat16 to match original function’s return dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
