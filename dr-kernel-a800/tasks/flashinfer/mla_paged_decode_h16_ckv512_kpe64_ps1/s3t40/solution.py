import torch
import math
import triton
import triton.language as tl


# Kernel 1: Compute logits[h, t] = qn[h] · Kc[t] + qp[h] · Kp[t]
# Inputs:
#   qn: [H, Dq], float32
#   qp: [H, Dp], float32
#   Kc: [T, Dq], float32
#   Kp: [T, Dp], float32
#   logits: [H, T], float32
@triton.jit
def fused_logits_kernel(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_ptr,
                         H: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr, T: tl.constexpr,
                         BLOCK_T: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Load q vectors for this head (vectors of length Dq and Dp)
    qn_vec = tl.load(qn_ptr + pid_h * Dq + tl.arange(0, Dq), mask=True, other=0.0)  # [Dq]
    qp_vec = tl.load(qp_ptr + pid_h * Dp + tl.arange(0, Dp), mask=True, other=0.0)  # [Dp]

    # Accumulate dot products per token in this tile
    dot_qn = tl.zeros((), dtype=tl.float32)
    dot_qp = tl.zeros((), dtype=tl.float32)

    # Loop over Kc's feature dimension in tiles
    for d in range(0, Dq, BLOCK_T):
        offs_d = d + tl.arange(0, BLOCK_T)
        mask_d = offs_d < Dq
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :],
                          mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, BLOCK_D]
        qn_tile = tl.load(qn_ptr + pid_h * Dq + offs_d, mask=mask_d, other=0.0)  # [BLOCK_D]
        # Reduce along feature axis to get contribution for each token in the tile
        dot_qn += tl.sum(Kc_tile * qn_tile[None, :], axis=1)  # [BLOCK_T]

    # Loop over Kp's feature dimension in tiles
    for d in range(0, Dp, BLOCK_T):
        offs_d = d + tl.arange(0, BLOCK_T)
        mask_d = offs_d < Dp
        Kp_tile = tl.load(Kp_ptr + offs_t[:, None] * Dp + offs_d[None, :],
                          mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, BLOCK_D]
        qp_tile = tl.load(qp_ptr + pid_h * Dp + offs_d, mask=mask_d, other=0.0)  # [BLOCK_D]
        dot_qp += tl.sum(Kp_tile * qp_tile[None, :], axis=1)  # [BLOCK_T]

    logits_row = dot_qn + dot_qp  # [BLOCK_T]
    tl.store(logits_ptr + pid_h * T + offs_t, logits_row, mask=mask_t)


# Kernel 2: Row-wise softmax over tokens for each head
# Inputs:
#   logits_scaled: [H, T], float32
#   attn: [H, T], float32
@triton.jit
def softmax_row_kernel(logits_scaled_ptr, attn_ptr,
                        H: tl.constexpr, T: tl.constexpr,
                        BLOCK_T: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    row = tl.load(logits_scaled_ptr + pid_h * T + offs_t, mask=mask_t, other=-float("inf"))  # [BLOCK_T]
    row_max = tl.max(row, axis=0)
    exp_row = tl.exp(row - row_max)
    sum_exp = tl.sum(exp_row, axis=0)
    attn_row = exp_row / sum_exp
    tl.store(attn_ptr + pid_h * T + offs_t, attn_row, mask=mask_t)


# Kernel 3: Row-wise logsumexp over tokens for each head, then divide by ln(2)
# Inputs:
#   logits_scaled: [H, T], float32
#   lse_ptr: [H], float32
@triton.jit
def lse_row_kernel(logits_scaled_ptr, lse_ptr,
                   H: tl.constexpr, T: tl.constexpr,
                   BLOCK_T: tl.constexpr):
    pid_h = tl.program_id(0)
    # Iterate across tokens in tiles to compute max
    row_max = tl.full((), -float("inf"), dtype=tl.float32)
    for t in range(0, T, BLOCK_T):
        offs = t + tl.arange(0, BLOCK_T)
        mask = offs < T
        row = tl.load(logits_scaled_ptr + pid_h * T + offs, mask=mask, other=-float("inf"))
        tile_max = tl.max(row, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for t in range(0, T, BLOCK_T):
        offs = t + tl.arange(0, BLOCK_T)
        mask = offs < T
        row = tl.load(logits_scaled_ptr + pid_h * T + offs, mask=mask, other=-float("inf"))
        exp_row = tl.exp(row - row_max)
        sum_exp += tl.sum(exp_row, axis=0)

    lse_val = tl.log(sum_exp) + row_max
    ln2 = 0.6931471805599453
    lse_val = lse_val / ln2
    tl.store(lse_ptr + pid_h, lse_val)


# Kernel 4: Per-head GEMV: out[h, :] = attn[h, :] @ Kc[:, :]
# Inputs:
#   attn: [H, T], float32
#   Kc: [T, Dq], float32
#   out: [H, Dq], float32
@triton.jit
def gemv_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                    H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr,
                    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_h = tl.program_id(0)
    out_vec = tl.zeros((Dq,), dtype=tl.float32)

    for t in range(0, T, BLOCK_T):
        offs_t = t + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        attn_row = tl.load(attn_ptr + pid_h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + tl.arange(0, BLOCK_D)[None, :],
                          mask=mask_t[:, None], other=0.0)  # [BLOCK_T, BLOCK_D]
        partial = tl.sum(Kc_tile * attn_row[:, None], axis=0)  # [BLOCK_D]
        out_vec += partial

    tl.store(out_ptr + pid_h * Dq + tl.arange(0, Dq), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda \
            and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton"

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dq = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Expect fixed dimensions as per original code
        assert Dq == 512 and Dp == 64, "Expected head_dim_ckv=512 and head_dim_kpe=64"

        # Output buffers
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b] = 0.0
                lse[b] = -float("inf")
                continue

            # Gather tokens and corresponding cache rows
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)
            Kc_b = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, 512]
            Kp_b = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, 64]

            # Load q vectors for this batch
            qn = q_nope[b].contiguous().to(torch.float32)  # [16, 512]
            qp = q_pe[b].contiguous().to(torch.float32)    # [16, 64]

            # Allocate intermediates
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)

            # Launch fused logits kernel: grid = (H, ceil_div(T, BLOCK_T))
            BLOCK_T_logits = 128
            grid_logits = (H, triton.cdiv(L_tokens, BLOCK_T_logits))
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, Dq=Dq, Dp=Dp, T=L_tokens,
                BLOCK_T=BLOCK_T_logits,
                num_warps=4, num_stages=2
            )

            # Scale logits by sm_scale
            logits_scaled = logits * sm_scale

            # Launch softmax row kernel: grid = (H, ceil_div(T, BLOCK_T))
            grid_softmax = (H, triton.cdiv(L_tokens, BLOCK_T_logits))
            softmax_row_kernel[grid_softmax](
                logits_scaled, attn,
                H=H, T=L_tokens,
                BLOCK_T=BLOCK_T_logits,
                num_warps=4, num_stages=2
            )

            # Launch LSE row kernel: grid = (H,)
            grid_lse = (H,)
            lse[b] = torch.empty((H,), dtype=torch.float32, device=q_nope.device)
            lse_row_kernel[grid_lse](
                logits_scaled, lse[b],
                H=H, T=L_tokens,
                BLOCK_T=BLOCK_T_logits,
                num_warps=4, num_stages=2
            )

            # Per-head GEMV: out[h, :] = attn[h, :] @ Kc


def run(*args):
    return ModelNew()(*args)
