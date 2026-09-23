import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits[h, t] = dot(qn[h, :], Kc[t, :]) + dot(qp[h, :], Kp[t, :])
@triton.jit
def fused_logits_kernel(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_ptr,
                         H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr,
                         BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    # Preload q vectors for this head using compile-time Dq/Dp
    qn = tl.load(qn_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp))  # [Dp]

    # Loop over tokens in tiles
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T

        # Load Kc_sub [BLOCK_T, Dq] and Kp_sub [BLOCK_T, Dp]
        Kc_sub = tl.load(Kc_ptr + offs_t[:, None] * Dq + tl.arange(0, Dq)[None, :], mask=mask_t[:, None], other=0.0)
        Kp_sub = tl.load(Kp_ptr + offs_t[:, None] * Dp + tl.arange(0, Dp)[None, :], mask=mask_t[:, None], other=0.0)

        # Compute dot-products for this tile
        # prod1: [BLOCK_T], sum over feature dimension Dq
        prod1 = tl.sum(Kc_sub * qn[None, :], axis=1)  # sum over axis=1 (features)
        # prod2: [BLOCK_T], sum over feature dimension Dp
        prod2 = tl.sum(Kp_sub * qp[None, :], axis=1)

        logits_row = prod1 + prod2  # [BLOCK_T]

        # Store logits[h, t_start : t_start+BLOCK_T]
        tl.store(logits_ptr + h * T + offs_t, logits_row, mask=mask_t)


# Triton kernel: row-wise softmax: attn[h, :] = softmax(logits[h, :] * sm_scale)
@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, sm_scale,
                        T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head

    # Load logits row and compute softmax
    offs = tl.arange(0, BLOCK_T)
    # We need to iterate over the entire row; T is constexpr, so we can use a loop in blocks
    # For simplicity, we implement a single pass with a vector length equal to T. BLOCK_T should be >= T.
    # If T is larger than BLOCK_T, we can loop, but here we set BLOCK_T=T to avoid extra loops.
    logits = tl.load(logits_ptr + h * T + offs, mask=offs < T, other=-float('inf'))

    # Scale
    scaled = logits * sm_scale

    # Numerical stability: subtract max
    max_val = tl.max(scaled, axis=0)
    exp_scaled = tl.exp(scaled - max_val)
    sum_exp = tl.sum(exp_scaled, axis=0)
    attn = exp_scaled / sum_exp

    tl.store(attn_ptr + h * T + offs, attn, mask=offs < T)


# Triton kernel: row-wise logsumexp: lse[h] = logsumexp(logits[h, :] * sm_scale) / ln(2)
@triton.jit
def lse_row_kernel(logits_ptr, lse_ptr, sm_scale,
                   T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head

    offs = tl.arange(0, BLOCK_T)
    logits = tl.load(logits_ptr + h * T + offs, mask=offs < T, other=-float('inf'))
    scaled = logits * sm_scale

    max_val = tl.max(scaled, axis=0)
    sum_exp = tl.sum(tl.exp(scaled - max_val), axis=0)
    lse = tl.log(sum_exp) + max_val  # logsumexp
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse = lse / ln2
    tl.store(lse_ptr + h, lse)


# Triton kernel: per-head matmul out[h, :] = attn[h, :] @ Kc[:, :]
@triton.jit
def matmul_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                      Dq: tl.constexpr, T: tl.constexpr,
                      BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head

    # Load attn[h, :] in blocks and compute out[h, :]
    # out[h, d] = sum_{t=0..T-1} attn[h, t] * Kc[t, d]
    for d_start in range(0, Dq, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < Dq

        # Accumulator for out[h, d_start : d_start+BLOCK_D]
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for t_start in range(0, T, BLOCK_T):
            offs_t = t_start + tl.arange(0, BLOCK_T)
            mask_t = offs_t < T

            attn_sub = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
            Kc_sub = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :],
                             mask=mask_t[:, None] & mask_d[None, :],
                             other=0.0)  # [BLOCK_T, BLOCK_D]

            # acc += sum_t (attn_sub[t] * Kc_sub[t, :])
            acc += tl.sum(Kc_sub * attn_sub[:, None], axis=0)

        # Store acc to out[h, d_start : d_start+BLOCK_D]
        tl.store(out_ptr + h * Dq + offs_d, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self, use_triton=True):
        super().__init__()
        self.use_triton = use_triton  # extra arg for compatibility; not used in forward

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton not available, fall back to PyTorch compute (but this environment requires Triton kernels).
        device = q_nope.device
        B = q_nope.shape[0]
        H = 16
        Dq = 512
        Dp = 64

        # Output tensors
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # we'll cast to bfloat16 at end
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute L_tokens and gather Kc/Kp for this batch
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV cache for this batch element: output zeros and continue
                output[b].zero_()
                lse[b].zero_()
                continue

            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int64)  # indices
            # Gather from caches
            Kc_b = ckv_cache.index_select(0, tok_idx).squeeze(1).contiguous().to(torch.float32)  # [L_tokens, Dq]
            Kp_b = kpe_cache.index_select(0, tok_idx).squeeze(1).contiguous().to(torch.float32)  # [L_tokens, Dp]

            # qn and qp for this batch
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, Dq]
            qp = q_pe[b].to(torch.float32).contiguous()    # [H, Dp]

            # Allocate logits [H, T]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel: one program per head
            grid_logits = (H,)
            # Choose BLOCK_T as the next power of two up to 1024; for large T, it handles in tiles.
            BLOCK_T = 256 if L_tokens >= 256 else (128 if L_tokens >= 128 else 64)
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Compute lse per head
            grid_lse = (H,)
            BLOCK_T_lse = max(L_tokens, 64)  # ensure we cover the whole row; set BLOCK_T >= T for simplicity
            lse_row_kernel[grid_lse](
                logits, lse[b],
                sm_scale=float(sm_scale),
                T=L_tokens, BLOCK_T=BLOCK_T_lse,
                num_warps=4, num_stages=2
            )

            # Softmax row-wise: attn[h, :] = softmax(logits_scaled[h, :])
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            # Use Triton softmax kernel; since BLOCK_T should be >= T, we set BLOCK_T = L_tokens (compile-time)
            # Note: Triton requires constexpr; we pass T as constexpr via launch args.
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                logits, attn,
                sm_scale=float(sm_scale),
                T=L_tokens, BLOCK_T=L_tokens,
                num_warps=4, num_stages=2
            )

            # Compute per-head output via Triton matmul: out[h, :] = attn[h, :] @ Kc_b[:, :]
            out_row = torch.empty((H, Dq), dtype=torch.float32, device=device)
            grid_matmul = (H,)
            BLOCK_D = 128 if Dq >= 128 else 64
            matmul_row_kernel[grid_matmul](
                attn, Kc_b, out_row,
                Dq=Dq, T=L_tokens,
                BLOCK_D=BLOCK_D, BLOCK_T=L_tokens,
                num_warps=4, num_stages=2
            )

            # Store outputs
            output[b] = out_row
        # Cast to bfloat16 to match original return type
        output = output.to(torch.bfloat16)

        # Return output and lse; original run returns (output, lse)
        return output, lse


def run(*args):
    return ModelNew()(*args)
