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
def fused_logits_kernel(q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr, logits_ptr,
                         H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr,
                         BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    # Preload q vectors for this head
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq), mask=True)  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp), mask=True)   # [Dp]

    # Tile over tokens
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T

        # Accumulate logits for this tile
        lo = tl.zeros((), dtype=tl.float32)
        # Dot over Dq: sum_i qn[i] * Kc[t_idx, i]
        for di in range(0, Dq, BLOCK_T):  # Dq is constexpr, this loop is unrolled
            d_offsets = di + tl.arange(0, BLOCK_T)  # shape [BLOCK_T]
            mask_d = d_offsets < Dq
            # Kc_sub: [BLOCK_T, BLOCK_T] gather
            Kc_sub = tl.load(
                Kc_ptr + t_idx[:, None] * Dq + d_offsets[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0
            )
            q_sub = qn[d_offsets]  # [BLOCK_T]
            # Multiply elementwise and reduce along feature axis
            # Make shapes broadcastable: [BLOCK_T, 1] * [BLOCK_T, BLOCK_T]
            lo += tl.sum(Kc_sub * q_sub[:, None], axis=1)  # [BLOCK_T]

        hi = tl.zeros((), dtype=tl.float32)
        for di in range(0, Dp, BLOCK_T):  # Dp is small (64), unrolled
            d_offsets = di + tl.arange(0, BLOCK_T)
            mask_d = d_offsets < Dp
            Kp_sub = tl.load(
                Kp_ptr + t_idx[:, None] * Dp + d_offsets[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0
            )
            qp_sub = qp[d_offsets]  # [BLOCK_T]
            hi += tl.sum(Kp_sub * qp_sub[:, None], axis=1)  # [BLOCK_T]

        logits_tile = lo + hi  # [BLOCK_T]
        # Store logits[h, t_start : t_start + BLOCK_T]
        tl.store(logits_ptr + h * T + t_idx, logits_tile, mask=mask_t)


# Triton kernel: row-wise softmax over logits_ptr[h, :]
@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, H: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    row_ptr = logits_ptr + h * T
    # Pass 1: compute max
    max_val = -1e20
    for t_start in range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        x = tl.load(row_ptr + offs, mask=mask, other=-1e20)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    # Pass 2: compute sum of exp
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        x = tl.load(row_ptr + offs, mask=mask, other=-1e20)
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)
    # Pass 3: write normalized attn
    attn_row_ptr = attn_ptr + h * T
    for t_start in range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        x = tl.load(row_ptr + offs, mask=mask, other=-1e20)
        attn = tl.exp(x - max_val) / sum_exp
        tl.store(attn_row_ptr + offs, attn, mask=mask)


# Triton kernel: row-wise logsumexp over logits_ptr[h, :], store to out_ptr[h]
@triton.jit
def lse_row_kernel(logits_ptr, out_ptr, H: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    row_ptr = logits_ptr + h * T
    max_val = -1e20
    for t_start in range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        x = tl.load(row_ptr + offs, mask=mask, other=-1e20)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        x = tl.load(row_ptr + offs, mask=mask, other=-1e20)
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)
    lse_val = max_val + tl.log(sum_exp)  # logsumexp
    # Divide by ln(2)
    lse_val = lse_val / 1.4426950408889634  # 1 / ln(2)
    tl.store(out_ptr + h, lse_val)


# Triton kernel: per-head matmul out[h, :] = attn[h, :] @ Kc[:, :]
@triton.jit
def matmul_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                      H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr,
                      BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    attn_row_ptr = attn_ptr + h * T
    # Preload attn for this head
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        attn_sub = tl.load(attn_row_ptr + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]

        # Accumulator for output vector (length Dq)
        acc = tl.zeros((Dq,), dtype=tl.float32)

        # Loop over feature tiles
        for d_start in range(0, Dq, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask_d = d_offsets < Dq
            Kc_sub = tl.load(
                Kc_ptr + t_idx[:, None] * Dq + d_offsets[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0
            )  # [BLOCK_T, BLOCK_D]
            # acc += sum_{t in tile} attn_sub[t] * Kc_sub[t, :]
            acc += tl.sum(Kc_sub * attn_sub[:, None], axis=0)  # [BLOCK_D]
        # Write acc to out[h, :]
        out_row_ptr = out_ptr + h * Dq
        tl.store(out_row_ptr + d_offsets, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_t=256, block_d=128):
        super().__init__()
        self.sm_scale = float(sm_scale)
        self.block_t = int(block_t)
        self.block_d = int(block_d)
        self.Dq = 512
        self.Dp = 64
        self.H = 16

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
        # Ensure on CUDA and contiguous
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernels."
        assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
        assert ckv_cache.dtype == torch.bfloat16 and kpe_cache.dtype == torch.bfloat16

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dq = 512
        Dp = 64

        output = torch.empty((B, H, Dq), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.bfloat16, device=device)

        # For each batch element
        for b in range(B):
            # Compute L_tokens
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp for this batch element
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]]
            Kc_b = ckv_cache[tok_idx]  # [L_tokens, 512], bfloat16
            Kp_b = kpe_cache[tok_idx]  # [L_tokens, 64], bfloat16

            # Convert to float32 for Triton compute
            qn = q_nope[b].to(torch.float32)  # [16, 512]
            qp = q_pe[b].to(torch.float32)    # [16, 64]

            # Allocate logits [H, T] in float32
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel: one program per head
            grid_logits = (H,)
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b.to(torch.float32), Kp_b.to(torch.float32), logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=self.block_t,
                num_warps=4, num_stages=2
            )

            # Launch lse_row_kernel: compute lse per head
            lse_b = torch.empty((H,), dtype=torch.float32, device=device)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                logits, lse_b,
                H=H, T=L_tokens,
                BLOCK_T=self.block_t,
                num_warps=4, num_stages=2
            )
            lse[b] = lse_b.to(torch.bfloat16)

            # Compute attn[h, :] = softmax(logits_scaled[h, :]) using Triton softmax_row_kernel
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                logits, attn,
                H=H, T=L_tokens,
                BLOCK_T=self.block_t,
                num_warps=4, num_stages=2
            )

            # Compute output[h, :] = attn[h, :] @ Kc_b[:, :] using Triton matmul_row_kernel
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=device)
            grid_matmul = (H,)
            matmul_row_kernel[grid_matmul](
                attn, Kc_b.to(torch.float32), out_b,
                H=H, T=L_tokens, Dq=Dq,
                BLOCK_T=self.block_t, BLOCK_D=self.block_d,
                num_warps=4, num_stages=2
            )

            # Store output for this batch element
            # Ensure H matches q_nope shape[1]
            # Note: out_b is [H, Dq], output[b] expects [H, Dq]
            output[b] = out_b.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
