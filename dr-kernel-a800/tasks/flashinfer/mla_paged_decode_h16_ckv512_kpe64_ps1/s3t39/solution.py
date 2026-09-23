import torch
import math
import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    qn_ptr,   # *f32 [H, Dq]
    qp_ptr,   # *f32 [H, Dp]
    Kc_ptr,   # *f32 [T, Dq]
    Kp_ptr,   # *f32 [T, Dp]
    logits_ptr,  # *f32 [H, T]
    H: tl.constexpr,  # number of heads (16)
    T: tl.constexpr,  # number of tokens
    Dq: tl.constexpr, # 512
    Dp: tl.constexpr, # 64
    BLOCK_T: tl.constexpr = 128,  # power-of-two
):
    # One program per (head, token-tile)
    h = tl.program_id(0)
    t_block = tl.program_id(1)

    # Accumulator for logits for this head across tokens in this tile
    acc = tl.zeros((), dtype=tl.float32)

    # Feature offsets within head dimension
    # We loop over token offsets in this block and sum contributions across features in tiles.
    offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Compute dot(qn[h, :], Kc[offs_t, :]) and dot(qp[h, :], Kp[offs_t, :]) and accumulate
    # Note: Dq and Dp are 512 and 64 respectively. We tile over features to reduce register pressure.
    # We unroll over feature tiles.
    # First term: qn[h] · Kc[t]
    for d in range(0, Dq, 64):  # feature tiles of 64 (power-of-two)
        offs_d = d + tl.arange(0, 64)  # [64]
        mask_d = offs_d < Dq
        # Load qn[h, d:d+64]
        qn_tile = tl.load(qn_ptr + h * Dq + offs_d, mask=mask_d, other=0.0)  # [64]
        # Load Kc[offs_t, d:d+64], shape [BLOCK_T, 64]
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, 64]
        # Accumulate dot(qn_tile, Kc_tile.sum(axis=1))
        # Broadcast qn_tile to [BLOCK_T, 64] and reduce along features
        contrib = tl.sum(qn_tile[None, :] * Kc_tile, axis=1)  # [BLOCK_T]
        acc += tl.sum(contrib, axis=0)  # scalar

    # Second term: qp[h] · Kp[t]
    for d in range(0, Dp, 64):
        offs_d = d + tl.arange(0, 64)  # [64]
        mask_d = offs_d < Dp
        qp_tile = tl.load(qp_ptr + h * Dp + offs_d, mask=mask_d, other=0.0)  # [64]
        Kp_tile = tl.load(Kp_ptr + offs_t[:, None] * Dp + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, 64]
        contrib = tl.sum(qp_tile[None, :] * Kp_tile, axis=1)  # [BLOCK_T]
        acc += tl.sum(contrib, axis=0)  # scalar

    # Store acc into logits[h, t_block*BLOCK_T:(t_block+1)*BLOCK_T]
    for tt in range(0, BLOCK_T):
        t_idx = t_block * BLOCK_T + tt
        if t_idx < T:
            tl.store(logits_ptr + h * T + t_idx, acc)


@triton.jit
def softmax_row_kernel(
    logits_ptr,  # *f32 [H, T]
    attn_ptr,    # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr = 128,
):
    # One program per head
    h = tl.program_id(0)
    # Load logits row
    acc = tl.zeros((), dtype=tl.float32)
    max_val = -float('inf')
    for t_block in range(0, T, BLOCK_T):
        offs_t = t_block + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))
    # Compute sum of exp(logits - max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for t_block in range(0, T, BLOCK_T):
        offs_t = t_block + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        expv = tl.exp(logits - max_val)
        sum_exp += tl.sum(expv, axis=0)
    inv_sum = 1.0 / sum_exp
    # Store softmax
    for t_block in range(0, T, BLOCK_T):
        offs_t = t_block + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        attn = tl.exp(logits - max_val) * inv_sum
        tl.store(attn_ptr + h * T + offs_t, attn, mask=mask_t)


@triton.jit
def lse_row_kernel(
    logits_ptr,  # *f32 [H, T]
    lse_ptr,     # *f32 [H]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr = 128,
):
    # One program per head
    h = tl.program_id(0)
    max_val = -float('inf')
    for t_block in range(0, T, BLOCK_T):
        offs_t = t_block + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))
    sum_exp = tl.zeros((), dtype=tl.float32)
    for t_block in range(0, T, BLOCK_T):
        offs_t = t_block + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(logits - max_val), axis=0)
    lse_val = tl.log(sum_exp) / math.log(2.0)
    tl.store(lse_ptr + h, lse_val)


# For the per-head GEMV: out[h, :] = attn[h, :] @ Kc[:, :]
# Triton does not provide a built-in GEMV; we implement it using tiles over tokens.
@triton.jit
def gemv_row_kernel(
    attn_ptr,  # *f32 [H, T]
    Kc_ptr,    # *f32 [T, Dq]
    out_ptr,   # *f32 [H, Dq]
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,  # 512
    BLOCK_T: tl.constexpr = 128,  # power-of-two
):
    h = tl.program_id(0)
    out_vec = tl.zeros((Dq,), dtype=tl.float32)
    for t_block in range(0, T, BLOCK_T):
        offs_t = t_block + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        attn_vec = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        # Load Kc[offs_t, :] tile [BLOCK_T, Dq] across feature tiles
        for d in range(0, Dq, 64):
            offs_d = d + tl.arange(0, 64)
            mask_d = offs_d < Dq
            Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :],
                              mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, 64]
            contrib = tl.sum(attn_vec[:, None] * Kc_tile, axis=0)  # [64]
            out_vec += contrib
    tl.store(out_ptr + h * Dq + tl.arange(0, Dq), out_vec, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA device for Triton kernels"

        # Shapes and assertions
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dq = q_nope.shape[2]
        assert H == 16, "num_qo_heads must be 16"
        assert Dq == 512, "head_dim_ckv must be 512"
        Dp = q_pe.shape[2]
        assert Dp == 64, "head_dim_kpe must be 64"

        output = torch.empty((B, H, Dq), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Precompute Dp for Kp load (same as above)
        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens, zero output
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp for this batch element
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]]
            # Kc_b: [L_tokens, Dq], Kp_b: [L_tokens, Dp]
            Kc_b = ckv_cache[tok_idx].squeeze(1).to(torch.float32)
            Kp_b = kpe_cache[tok_idx].squeeze(1).to(torch.float32)

            # qn and qp for this batch
            qn = q_nope[b].to(torch.float32)  # [H, Dq]
            qp = q_pe[b].to(torch.float32)    # [H, Dp]

            # 1) Compute logits_scaled [H, T] = (qn · Kc) + (qp · Kp)
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)
            grid_logits = (H, triton.cdiv(L_tokens, 128))
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Scale logits by sm_scale
            logits_scaled = logits * sm_scale

            # 2) Compute attn[h, :] = softmax(logits_scaled[h, :])
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                logits_scaled, attn,
                H=H, T=L_tokens, BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # 3) Compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
            lse_b = torch.empty((H,), dtype=torch.float32, device=q_nope.device)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                logits_scaled, lse_b,
                H=H, T=L_tokens, BLOCK_T=128,
                num_warps=4, num_stages=2
            )
            lse[b] = lse_b  # shape [H]

            # 4) Compute out[h, :] = attn[h, :] @ Kc[:, :] using Triton GEMV
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=q_nope.device)
            grid_gemv = (H,)
            gemv_row_kernel[grid_gemv](
                attn, Kc_b, out_b,
                H=H, T=L_tokens, Dq=Dq, BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            output[b] = out_b.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
