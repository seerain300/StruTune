import math
import torch

import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    q_nope_ptr,       # *f32 [H, Dq]
    q_pe_ptr,         # *f32 [H, Dp]
    Kc_ptr,           # *f32 [T, Dq]
    Kp_ptr,           # *f32 [T, Dp]
    logits_ptr,       # *f32 [H, T]
    H: tl.constexpr,
    Dq: tl.constexpr,
    Dp: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head
    # Preload q vectors once
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))    # [Dp]
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        sum_vec = tl.zeros([BLOCK_T], dtype=tl.float32)
        # dot over Dq for Kc
        for k in range(0, Dq):
            k_vec = tl.load(Kc_ptr + k * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            sum_vec += qn[k] * k_vec
        # dot over Dp for Kp
        for k in range(0, Dp):
            p_vec = tl.load(Kp_ptr + k * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            sum_vec += qp[k] * p_vec
        tl.store(logits_ptr + h * T + t_idx, sum_vec, mask=mask_t)


@triton.jit
def attn_matmul_kernel(
    attn_ptr,   # *f32 [H, T]
    Kc_ptr,     # *f32 [T, D]
    out_ptr,    # *f32 [H, D]
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + offs_d
        mask_d = d_idx < D
        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + offs_t
            mask_t = t_idx < T
            # Load attn chunk [BLOCK_T]
            attn_chunk = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            # Load Kc chunk [BLOCK_T, BLOCK_D]
            Kc_chunk = tl.load(
                Kc_ptr + t_idx[:, None] * D + d_idx[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            )
            # Accumulate over T tile
            acc += tl.sum(attn_chunk[:, None] * Kc_chunk, axis=0)
        tl.store(out_ptr + h * D + d_idx, acc, mask=mask_d)


@triton.jit
def softmax_row_kernel(
    x_ptr,            # *f32 [T]
    y_ptr,            # *f32 [T]
    T: tl.constexpr,  # length of row
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head
    for t_start in range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
        m = tl.max(x, axis=0)
        x = x - m
        e = tl.exp(x)
        denom = tl.sum(e, axis=0)
        y = e / denom
        tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def lse_row_kernel(
    x_ptr,            # *f32 [T]
    lse_ptr,          # *f32 [H]
    T: tl.constexpr,  # length of row
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head
    max_val = -float("inf")
    for t_start in range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, tile_max)
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)
    lse_val = tl.log(sum_exp) + max_val  # logsumexp
    # Divide by ln(2) (original code divides by log(2))
    lse_val = lse_val / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Move to CUDA for Triton execution
        original_device = q_nope.device
        if original_device.type != 'cuda':
            device = torch.device('cuda')
        else:
            device = original_device

        # Move inputs to CUDA and make contiguous
        q_nope = q_nope.to(device).contiguous()
        q_pe = q_pe.to(device).contiguous()
        ckv_cache = ckv_cache.to(device).contiguous()
        kpe_cache = kpe_cache.to(device).contiguous()
        kv_indptr = kv_indptr.to(device).contiguous()
        kv_indices = kv_indices.to(device).contiguous()

        # Cast to float32 for compute
        q_nope_f32 = q_nope.to(torch.float32)  # [B, 16, 512]
        q_pe_f32 = q_pe.to(torch.float32)      # [B, 16, 64]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        batch_size = q_nope_f32.shape[0]
        num_qo_heads = q_nope_f32.shape[1]
        head_dim_ckv = q_nope_f32.shape[2]
        head_dim_kpe = q_pe_f32.shape[2]

        # Output tensors
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute range from kv_indptr
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg
            if L_tokens <= 0:
                # No KV entries for this batch element: output zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices and corresponding K vectors
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()
            Kc = Kc_all[tok_idx]  # [L_tokens, 512]
            Kp = Kp_all[tok_idx]  # [L_tokens, 64]

            # Queries for this batch element
            qn = q_nope_f32[b]  # [16, 512]
            qp = q_pe_f32[b]    # [16, 64]

            # Allocate logits [H, T]
            H = num_qo_heads
            T = L_tokens
            logits = torch.empty((H, T), dtype=torch.float32, device=device)

            # Launch fused logits kernel: one program per head
            BLOCK_T = 256
            grid_logits = (H,)
            fused_logits_kernel[grid_logits](
                qn,               # q_nope_ptr
                qp,               # q_pe_ptr
                Kc,               # Kc_ptr
                Kp,               # Kp_ptr
                logits,           # logits_ptr
                H=H,
                Dq=head_dim_ckv,
                Dp=head_dim_kpe,
                T=T,
                BLOCK_T=BLOCK_T,
                num_warps=4,
            )

            # Scale logits
            scaled_logits = logits * sm_scale

            # Compute softmax per row using Triton kernel
            attn = torch.empty((H, T), dtype=torch.float32, device=device)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                scaled_logits,     # x_ptr
                attn,              # y_ptr
                T=T,
                BLOCK_T=256,
                num_warps=4,
            )

            # Compute lse per row using Triton kernel
            lse[b] = torch.zeros((H,), dtype=torch.float32, device=device)  # will be overwritten
            lse_row_kernel[grid_softmax](
                scaled_logits,     # x_ptr
                lse[b],            # lse_ptr
                T=T,
                BLOCK_T=256,
                num_warps=4,
            )

            # Compute output: out[h, :] = attn[h, :] @ Kc[:, :]
            out = torch.empty((H, head_dim_ckv), dtype=torch.float32, device=device)

            # Launch attn matmul kernel: one program per head
            BLOCK_T_mm = 256
            BLOCK_D_mm = 128
            grid_mm = (H,)
            attn_matmul_kernel[grid_mm](
                attn,             # attn_ptr
                Kc,               # Kc_ptr
                out,              # out_ptr
                H=H,
                T=T,
                D=head_dim_ckv,
                BLOCK_T=BLOCK_T_mm,
                BLOCK_D=BLOCK_D_mm,
                num_warps=4,
            )

            # Store to output tensor
            output[b] = out

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
