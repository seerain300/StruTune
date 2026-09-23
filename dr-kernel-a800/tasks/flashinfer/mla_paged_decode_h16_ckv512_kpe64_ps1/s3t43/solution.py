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
    out_ptr,  # *f32 [H, T]
    H: tl.constexpr,       # number of heads
    Dq: tl.constexpr,      # head_dim_ckv
    Dp: tl.constexpr,      # head_dim_kpe
    T: tl.constexpr,       # number of tokens in this batch
    BLOCK_T: tl.constexpr, # tile size for tokens
    BLOCK_K: tl.constexpr, # tile size for reduction dim (Dq)
):
    h = tl.program_id(0)  # head index
    tile_t = tl.program_id(1)  # token tile index

    offs_t = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    mask_t = offs_t < T

    # Accumulate logits for tokens in this tile
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # Loop over reduction dimension in tiles of size BLOCK_K (here BLOCK_K == Dq)
    for k in range(0, Dq, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load qn[h, k:k+BLOCK_K]
        qn_vec = tl.load(qn_ptr + h * Dq + offs_k, mask=(offs_k < Dq), other=0.0)  # [BLOCK_K]
        # Load Kc[offs_t, k:k+BLOCK_K]
        Kc_tile = tl.load(
            Kc_ptr + offs_t[:, None] * Dq + offs_k[None, :],
            mask=mask_t[:, None] & (offs_k[None, :] < Dq),
            other=0.0,
        )  # [BLOCK_T, BLOCK_K]
        # Accumulate dot products for this tile
        acc += tl.sum(Kc_tile * qn_vec[None, :], axis=1)  # [BLOCK_T]

    # Now add the contribution from qp and Kp if needed
    # Note: original code uses Kp only to compute logits; output uses Kc.
    for t in range(0, BLOCK_T):
        # Load Kp[offs_t[t], :]
        Kp_vec = tl.load(Kp_ptr + offs_t[t] * Dp + tl.arange(0, Dp), mask=(offs_t[t] < T), other=0.0)
        # Load qp[h, :]
        qp_vec = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp), mask=True, other=0.0)
        # Accumulate dot
        acc[t] += tl.sum(Kp_vec * qp_vec)

    # Store results
    tl.store(out_ptr + h * T + offs_t, acc, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    x_ptr,   # *f32 [H, T]
    out_ptr, # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)
    tile_t = tl.program_id(1)

    offs_t = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    x = tl.load(x_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
    m = tl.max(x, axis=0)  # scalar
    x = x - m
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)  # scalar
    attn = exp_x / denom
    tl.store(out_ptr + h * T + offs_t, attn, mask=mask_t)


@triton.jit
def lse_row_kernel(
    x_ptr,   # *f32 [H, T]
    out_ptr, # *f32 [H]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)
    tile_t = tl.program_id(1)

    offs_t = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    x = tl.load(x_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
    m = tl.max(x, axis=0)
    # sum(exp(x - m)) over the tile
    sum_tile = tl.sum(tl.exp(x - m), axis=0)
    # combine with previous tiles: use atomic_add to aggregate partial sums per head
    tl.atomic_add(out_ptr + h, sum_tile)


@triton.jit
def gemv_row_kernel(
    attn_ptr,  # *f32 [H, T]
    Kc_ptr,    # *f32 [T, Dq]
    out_ptr,   # *f32 [H, Dq]
    H: tl.constexpr,
    Dq: tl.constexpr,
    T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)  # head index
    tile_d = tl.program_id(1)  # output dim tile index

    offs_d = tile_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    mask_d = offs_d < Dq

    # Accumulator for this tile of Dq
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop over tokens in tiles
    for t in range(0, T, BLOCK_T):
        offs_t = t + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        mask_t = offs_t < T
        # Load attn[h, t:t+BLOCK_T]
        attn_vec = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        # Load Kc[offs_t, offs_d]
        Kc_tile = tl.load(
            Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        )  # [BLOCK_T, BLOCK_D]
        # Accumulate dot for this tile
        acc += tl.sum(Kc_tile * attn_vec[:, None], axis=0)  # [BLOCK_D]

    # Store results
    tl.store(out_ptr + h * Dq + offs_d, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors."

        # Fixed dimensions
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dq = q_nope.shape[2]  # 512
        Dp = q_pe.shape[2]    # 64

        device = q_nope.device

        # Allocate outputs
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # per batch, head, dim
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # Triton block sizes (power-of-two)
        BLOCK_T_logits = 128
        BLOCK_T_softmax = 128
        BLOCK_T_lse = 128
        BLOCK_T_gemv = 128
        BLOCK_K_logits = 128  # we set to 512; Triton will iterate over Dq
        BLOCK_D_gemv = 128

        for b in range(B):
            # Compute number of tokens for this batch
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens for this batch; output zeros, lse stays -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather tokens and corresponding cache rows
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)
            Kc_b = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, 512]
            Kp_b = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, 64]

            # Load q vectors for this batch
            qn = q_nope[b].contiguous().to(torch.float32)  # [16, 512]
            qp = q_pe[b].contiguous().to(torch.float32)    # [16, 64]

            # Allocate intermediates
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            logits_scaled = logits * sm_scale

            # 1) fused_logits_kernel: compute logits[h, t] = qn[h] · Kc[t] + qp[h] · Kp[t]
            grid_logits = (H, triton.cdiv(L_tokens, BLOCK_T_logits))
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, Dq=Dq, Dp=Dp, T=L_tokens,
                BLOCK_T=BLOCK_T_logits, BLOCK_K=128,
                num_warps=4, num_stages=2
            )

            # 2) softmax_row_kernel: compute attn[h, t] = softmax(logits_scaled[h, t])
            grid_softmax = (H, triton.cdiv(L_tokens, BLOCK_T_softmax))
            softmax_row_kernel[grid_softmax](
                logits_scaled, attn,
                H=H, T=L_tokens, BLOCK_T=BLOCK_T_softmax,
                num_warps=4, num_stages=2
            )

            # 3) lse_row_kernel: compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
            # We need full reduction across T. Use a grid with single tile and atomic_add over H.
            grid_lse = (H, 1)
            lse_row_kernel[grid_lse](
                logits_scaled, lse[b],
                H=H, T=L_tokens, BLOCK_T=BLOCK_T_lse,
                num_warps=4, num_stages=2
            )
            # Convert to proper shape (B, H)
            # Note: lse is already [B, H]; done per head.

            # 4) gemv_row_kernel: out[h, :] = attn[h, :] @ Kc[:, :]
            # Output [H, Dq]; we will write into output[b, h, :]
            grid_gemv = (H, triton.cdiv(Dq, BLOCK_D_gemv))
            # Launch for this batch element; output[b, h, :] will be filled
            # We pass output[b] as out_ptr; output is [H, Dq] contiguous
            out_per_batch = output[b]  # float32 [H, Dq], contiguous
            gemv_row_kernel[grid_gemv](
                attn, Kc_b, out_per_batch,
                H=H, Dq=Dq, T=L_tokens,
                BLOCK_D=BLOCK_D_gemv, BLOCK_T=BLOCK_T_gemv,
                num_warps=4, num_stages=2
            )

        # Return as bfloat16 to match original, and lse in float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
