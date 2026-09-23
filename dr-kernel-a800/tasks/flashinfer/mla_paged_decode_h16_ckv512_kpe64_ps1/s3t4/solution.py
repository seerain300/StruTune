import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr, logits_ptr,
    H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # One Triton program per head
    h = tl.program_id(0)
    # Preload q vectors for this head using compile-time lengths
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq), mask=True)  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp), mask=True)   # [Dp]
    offs_t = tl.arange(0, BLOCK_T)

    # Loop over tokens in tiles
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T

        # Accumulate dot-products for this tile
        acc_qn = tl.zeros([BLOCK_T], dtype=tl.float32)
        acc_qp = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Dot over Dq: qn[h, :] @ Kc[t, :]
        for d in range(0, Dq):
            kd = tl.load(Kc_ptr + t_idx * Dq + d, mask=mask_t, other=0.0)  # [BLOCK_T]
            acc_qn += qn[d] * kd
        # Dot over Dp: qp[h, :] @ Kp[t, :]
        for d in range(0, Dp):
            kp = tl.load(Kp_ptr + t_idx * Dp + d, mask=mask_t, other=0.0)  # [BLOCK_T]
            acc_qp += qp[d] * kp

        logits_tile = acc_qn + acc_qp  # [BLOCK_T]
        # Store to logits[h, t_start : t_start+BLOCK_T]
        tl.store(logits_ptr + h * T + t_idx, logits_tile, mask=mask_t)


@triton.jit
def softmax_row_kernel(scaled_logits_ptr, attn_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    # One Triton program per row (head)
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)

    # Pass 1: compute max for numerical stability
    max_val = tl.full([1], -1e20, dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_logits_ptr + h * T + t_idx, mask=mask_t, other=-1e20)
        # Reduce max over this tile
        for i in range(0, BLOCK_T):
            mi = t_start + i
            valid = mi < T
            xi = tl.where(valid, x[i], -1e20)
            max_val = tl.maximum(max_val, xi)

    # Pass 2: compute sum of exp(scaled - max)
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_logits_ptr + h * T + t_idx, mask=mask_t, other=-1e20)
        for i in range(0, BLOCK_T):
            mi = t_start + i
            valid = mi < T
            xi = tl.where(valid, x[i], -1e20)
            sum_exp += tl.where(valid, tl.exp(xi - max_val), 0.0)

    # Pass 3: write normalized attn
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_logits_ptr + h * T + t_idx, mask=mask_t, other=-1e20)
        for i in range(0, BLOCK_T):
            mi = t_start + i
            valid = mi < T
            xi = tl.where(valid, x[i], -1e20)
            attn_val = tl.exp(xi - max_val) / sum_exp
            tl.store(attn_ptr + h * T + t_idx + i, attn_val, mask=(t_idx + i) < T)


@triton.jit
def lse_row_kernel(scaled_logits_ptr, lse_ptr, T: tl.constexpr, ln2: tl.constexpr, BLOCK_T: tl.constexpr):
    # One Triton program per row (head)
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)

    max_val = tl.full([1], -1e20, dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_logits_ptr + h * T + t_idx, mask=mask_t, other=-1e20)
        for i in range(0, BLOCK_T):
            mi = t_start + i
            valid = mi < T
            xi = tl.where(valid, x[i], -1e20)
            max_val = tl.maximum(max_val, xi)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_logits_ptr + h * T + t_idx, mask=mask_t, other=-1e20)
        for i in range(0, BLOCK_T):
            mi = t_start + i
            valid = mi < T
            xi = tl.where(valid, x[i], -1e20)
            sum_exp += tl.where(valid, tl.exp(xi - max_val), 0.0)

    lse_val = tl.log(sum_exp) + max_val  # logsumexp in natural log
    lse_val = lse_val / ln2  # convert to base-2
    # Store lse for this head
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def attn_matmul_kernel(attn_ptr, Kc_ptr, out_ptr,
                       H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr,
                       BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    # One Triton program per head
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Accumulator for [BLOCK_D]
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Loop over tokens in tiles, accumulate out[h, d_start : d_start + BLOCK_D]
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        # Load attn chunk [BLOCK_T]
        attn_chunk = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
        # For each d in tile
        for d in range(0, BLOCK_D):
            d_idx = d + offs_d
            # Kc has shape [T, Dq], we want Kc[t, d] for t in tile
            Kc_d_vec = tl.load(Kc_ptr + t_idx[:, None] * Dq + d_idx[None, :],  # [BLOCK_T, 1]
                               mask=mask_t[:, None], other=0.0).to(tl.float32)
            prod = attn_chunk * Kc_d_vec[:, 0]  # [BLOCK_T]
            acc[d] += tl.sum(prod, axis=0)

    # Store acc to out[h, :]
    tl.store(out_ptr + h * Dq + offs_d, acc, mask=offs_d < Dq)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Move to CUDA and ensure contiguous
        device = q_nope.device
        if device.type != 'cuda':
            # Fallback for non-CUDA (evaluation uses CUDA)
            return self._fallback_forward(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)

        # Cast to float32 for compute
        q_nope = q_nope.contiguous().to(torch.float32)  # [B, H, Dq]
        q_pe = q_pe.contiguous().to(torch.float32)     # [B, H, Dp]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [N, 64]

        B = q_nope.shape[0]
        H = q_nope.shape[1]  # 16
        Dq = q_nope.shape[2] # 512
        Dp = q_pe.shape[2]   # 64

        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute number of tokens used for this batch element
            if kv_indptr.shape[0] <= b + 1:
                output[b].zero_()
                lse[b].zero_()
                continue
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())

            if L_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc, Kp for these tokens
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)  # [L_tokens]
            Kc = Kc_all[tok_idx]  # [L_tokens, 512]
            Kp = Kp_all[tok_idx]  # [L_tokens, 64]

            # Prepare logits buffer [H, T]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel: one program per head
            BLOCK_T = 256
            grid = (H,)
            fused_logits_kernel[grid](
                q_nope[b], q_pe[b], Kc, Kp, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=BLOCK_T,
                num_warps=4
            )

            # Scale logits
            scaled = logits * sm_scale

            # Compute attn per head via softmax kernel
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                scaled, attn,
                T=L_tokens, BLOCK_T=BLOCK_T,
                num_warps=4
            )

            # Compute lse per head via logsumexp kernel (base-2)
            ln2 = math.log(2.0)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                scaled, lse[b],
                T=L_tokens, ln2=ln2, BLOCK_T=BLOCK_T,
                num_warps=4
            )

            # Compute output[h, :] = attn[h, :] @ Kc[:, :]
            out = torch.empty((H, Dq), dtype=torch.float32, device=device)
            grid_matmul = (H,)
            attn_matmul_kernel[grid_matmul](
                attn, Kc, out,
                H=H, T=L_tokens, Dq=Dq, Dp=0,  # Dp unused here
                BLOCK_T=BLOCK_T, BLOCK_D=128,
                num_warps=4
            )
            output[b] = out

        # Return output


def run(*args):
    return ModelNew()(*args)
