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
    H: tl.constexpr,   # number of heads (e.g., 16)
    T: tl.constexpr,   # number of tokens in this batch element
    Dq: tl.constexpr,  # feature dim of Kc (e.g., 512)
    Dp: tl.constexpr,  # feature dim of Kp (e.g., 64)
    BLOCK_T: tl.constexpr = 128,
):
    h = tl.program_id(0)  # head id
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t = t_start + offs_t
        mask_t = t < T

        # Accumulator for logits[h, t]
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # Tile over feature dimension for qn (Dq) and Kc
        offs_dq = tl.arange(0, 128)
        # Loop over Dq in chunks of 128
        for d_start in range(0, Dq, 128):
            d = d_start + offs_dq
            mask_d = d < Dq

            # Load qn[h, d]
            qn_vec = tl.load(qn_ptr + h * Dq + d, mask=mask_d, other=0.0)  # [128]
            # Load Kc[t, d] for all t in this tile
            Kc_tile = tl.load(Kc_ptr + t[:, None] * Dq + d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, 128]
            # Accumulate dot products over d
            acc += tl.sum(Kc_tile * qn_vec[None, :], axis=1)  # [BLOCK_T]

        # Tile over feature dimension for qp (Dp) and Kp
        offs_dp = tl.arange(0, 128)
        for d_start in range(0, Dp, 128):
            dp = d_start + offs_dp
            mask_dp = dp < Dp
            qp_vec = tl.load(qp_ptr + h * Dp + dp, mask=mask_dp, other=0.0)  # [128]
            Kp_tile = tl.load(Kp_ptr + t[:, None] * Dp + dp[None, :], mask=mask_t[:, None] & mask_dp[None, :], other=0.0)  # [BLOCK_T, 128]
            acc += tl.sum(Kp_tile * qp_vec[None, :], axis=1)  # [BLOCK_T]

        # Store acc into logits[h, t]
        tl.store(logits_ptr + h * T + t, acc, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_scaled_ptr,  # *f32 [H, T]
    attn_ptr,           # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr = 128,
):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t = t_start + offs_t
        mask_t = t < T
        x = tl.load(logits_scaled_ptr + h * T + t, mask=mask_t, other=-float("inf"))  # [BLOCK_T]
        m = tl.max(x, axis=0)  # scalar max
        x_shift = x - m
        exp_x = tl.exp(x_shift)
        sum_exp = tl.sum(exp_x, axis=0)
        attn_vals = exp_x / sum_exp
        tl.store(attn_ptr + h * T + t, attn_vals, mask=mask_t)


@triton.jit
def lse_row_kernel(
    logits_scaled_ptr,  # *f32 [H, T]
    lse_ptr,            # *f32 [H]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr = 128,
):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    max_val = -float("inf")
    for t_start in range(0, T, BLOCK_T):
        t = t_start + offs_t
        mask_t = t < T
        x = tl.load(logits_scaled_ptr + h * T + t, mask=mask_t, other=-float("inf"))
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_val = 0.0
    for t_start in range(0, T, BLOCK_T):
        t = t_start + offs_t
        mask_t = t < T
        x = tl.load(logits_scaled_ptr + h * T + t, mask=mask_t, other=-float("inf"))
        sum_val += tl.sum(tl.exp(x - max_val), axis=0)

    lse = max_val + tl.log(sum_val)  # logsumexp
    # Divide by ln(2) to match original behavior
    ln2 = 0.6931471805599453
    lse = lse / ln2
    tl.store(lse_ptr + h, lse)


@triton.jit
def gemv_row_kernel(
    attn_ptr,   # *f32 [H, T]
    Kc_ptr,     # *f32 [T, Dq]
    out_ptr,    # *f32 [H, Dq]
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,
    BLOCK_T: tl.constexpr = 128,
):
    h = tl.program_id(0)
    offs_d = tl.arange(0, 128)
    for d_start in range(0, Dq, 128):
        d = d_start + offs_d
        mask_d = d < Dq
        acc = tl.zeros((128,), dtype=tl.float32)
        for t_start in range(0, T, BLOCK_T):
            t = t_start + tl.arange(0, BLOCK_T)
            mask_t = t < T
            attn_vals = tl.load(attn_ptr + h * T + t, mask=mask_t, other=0.0)  # [BLOCK_T]
            Kc_tile = tl.load(Kc_ptr + t[:, None] * Dq + d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, 128]
            # acc += sum_t(attn_vals[t] * Kc_tile[t, :]) for each chunk of t
            # Here BLOCK_T is 128; we sum over t dimension
            acc += tl.sum(Kc_tile * attn_vals[:, None], axis=0)  # [128]
        tl.store(out_ptr + h * Dq + d, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels"

        B = q_nope.shape[0]
        H = q_nope.shape[1]  # number of heads, expected 16
        Dq = q_nope.shape[2]  # feature dim of q_nope, expected 512
        Dp = q_pe.shape[2]    # feature dim of q_pe, expected 64

        output = torch.empty((B, H, Dq), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b+1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV cache for this batch element; zero output
                output[b].zero_()
                lse[b].zero_()
                continue

            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b+1]].to(torch.long)

            # Load Kc and Kp for this batch element (float32 for compute)
            Kc_b = ckv_cache[tok_idx].squeeze(1).float()  # [L_tokens, Dq]
            Kp_b = kpe_cache[tok_idx].squeeze(1).float()  # [L_tokens, Dp]

            # Prepare qn and qp for this batch element (float32 for compute)
            qn = q_nope[b].float()  # [H, Dq]
            qp = q_pe[b].float()    # [H, Dp]

            # Allocate logits_scaled and attn
            logits_scaled = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)

            # 1) Compute logits_scaled[h, t] = qn[h]·Kc[t] + qp[h]·Kp[t]
            grid_log = (H,)
            fused_logits_kernel[grid_log](
                qn, qp, Kc_b, Kp_b, logits_scaled,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # 2) Softmax per head along tokens
            grid_sm = (H,)
            softmax_row_kernel[grid_sm](
                logits_scaled, attn,
                H=H, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # 3) LogSumExp per head divided by ln(2)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                logits_scaled, lse[b],
                H=H, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # 4) Per-head GEMV: out[h, :] = attn[h, :] @ Kc[:, :]
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=q_nope.device)
            grid_gemv = (H,)
            gemv_row_kernel[grid_gemv](
                attn, Kc_b, out_b,
                H=H, T=L_tokens, Dq=Dq,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Store out[b, h, :] to output
            output[b] = out_b.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
