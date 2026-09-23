import math
import torch
import triton
import triton.language as tl

# Triton kernels

@triton.jit
def fused_logits_kernel(
    q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr, logits_ptr,
    B, H, T, Dq: tl.constexpr, Dp: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # One program per head h
    h = tl.program_id(0)
    # Preload q vectors once (compile-time known lengths)
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))   # [Dp]

    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T

        # Accumulate logits for this tile [BLOCK_T]
        logits_tile = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Dot over Dq: qn @ Kc_sub
        Kc_sub = tl.load(Kc_ptr + t_idx * Dq + tl.arange(0, Dq), mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dq]
        dot1 = tl.sum(qn[None, :] * Kc_sub, axis=1)  # [BLOCK_T]

        # Dot over Dp: qp @ Kp_sub
        Kp_sub = tl.load(Kp_ptr + t_idx * Dp + tl.arange(0, Dp), mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dp]
        dot2 = tl.sum(qp[None, :] * Kp_sub, axis=1)  # [BLOCK_T]

        logits_tile = dot1 + dot2

        # Store logits tile
        tl.store(logits_ptr + h * T + t_idx, logits_tile, mask=mask_t)


@triton.jit
def softmax_row_kernel(scaled_ptr, attn_ptr, H, T, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    # Compute row-wise softmax for row h
    row_start = h * T

    # First pass: max
    row_max = -float("inf")
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask = offs_t < T
        x = tl.load(scaled_ptr + row_start + offs_t, mask=mask, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    # Second pass: sum of exp
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask = offs_t < T
        x = tl.load(scaled_ptr + row_start + offs_t, mask=mask, other=-float("inf"))
        e = tl.exp(x - row_max)
        sum_exp += tl.sum(e, axis=0)

    # Third pass: write normalized attn
    inv_sum = 1.0 / sum_exp
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask = offs_t < T
        x = tl.load(scaled_ptr + row_start + offs_t, mask=mask, other=-float("inf"))
        attn = tl.exp(x - row_max) * inv_sum
        tl.store(attn_ptr + row_start + offs_t, attn, mask=mask)


@triton.jit
def lse_row_kernel(scaled_ptr, lse_ptr, H, T, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    row_start = h * T

    # Accumulate logsumexp in float32
    acc = 0.0
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask = offs_t < T
        x = tl.load(scaled_ptr + row_start + offs_t, mask=mask, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        sum_exp = tl.sum(tl.exp(x - tile_max), axis=0)
        acc += sum_exp * tl.exp(tile_max)

    logsumexp = tl.log(acc)
    # Store lse[h] = logsumexp / ln(2)
    tl.store(lse_ptr + h, logsumexp / 0.6931471805599453)  # 1 / ln(2)


@triton.jit
def attn_matmul_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H, T, Dq: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # One program per head h
    h = tl.program_id(0)
    row_attn = h * T
    for d_start in range(0, Dq, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)          # [BLOCK_D]
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Accumulate over tokens in tiles
        for t_start in range(0, T, BLOCK_T):
            offs_t = t_start + tl.arange(0, BLOCK_T)    # [BLOCK_T]
            mask_t = offs_t < T
            # Load attn chunk [BLOCK_T]
            attn_chunk = tl.load(attn_ptr + row_attn + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
            # Load Kc chunk [BLOCK_T, BLOCK_D]
            Kc_chunk = tl.load(
                Kc_ptr + offs_t[:, None] * Dq + d_idx[None, :],
                mask=mask_t[:, None],
                other=0.0,
            )  # [BLOCK_T, BLOCK_D]
            # Reduce over tokens t: acc += sum_t (attn_chunk[t] * Kc_chunk[t, :])
            acc += tl.sum(attn_chunk[:, None] * Kc_chunk, axis=0)
        # Store results
        tl.store(out_ptr + h * Dq + d_idx, acc, mask=d_idx < Dq)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguous, cast to float32 for compute
        device = torch.device('cuda')
        q_nope = q_nope.to(device).contiguous().to(torch.float32)
        q_pe = q_pe.to(device).contiguous().to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).to(device).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(device).contiguous().to(torch.float32)  # [num_pages, 64]

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        Dq = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Output tensors (float32 for compute, cast later)
        output = torch.empty((batch_size, num_qo_heads, Dq), dtype=torch.float32, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute range from kv_indptr
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg

            if L_tokens <= 0:
                # No KV cache entries for this batch element; output zeros and lse -inf
                output[b] = 0.0
                lse[b] = -float("inf")
                continue

            # Gather used token indices and corresponding Kc/Kp
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L_tokens]
            Kc = Kc_all[tok_idx]  # [L_tokens, Dq]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Prepare q vectors for this batch element
            qn = q_nope[b]  # [num_qo_heads, Dq]
            qp = q_pe[b]    # [num_qo_heads, Dp]

            # Allocate tensors for logits, scaled, attn, and output (per head)
            # Triton will write per head via grid=(H,)
            logits = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device)  # we won't read it back
            scaled = torch.empty_like(logits)
            attn = torch.empty_like(scaled)

            # Launch fused logits kernel: grid=(H,)
            H = num_qo_heads
            T = L_tokens
            Dq_const = Dq  # compile-time constants passed as constexpr
            Dp_const = Dp
            BLOCK_T = 256

            fused_logits_kernel[(H,)](
                qn, qp, Kc, Kp, logits, B=0, H=H, T=T,
                Dq=Dq_const, Dp=Dp_const, BLOCK_T=BLOCK_T,
            )

            # Scale logits
            scaled = logits * sm_scale

            # Launch softmax per row (head)
            softmax_row_kernel[(H,)](
                scaled, attn, H=H, T=T, BLOCK_T=BLOCK_T,
            )

            # Launch lse per row (head)
            lse_row_kernel[(H,)](
                scaled, lse[b], H=H, T=T, BLOCK_T=BLOCK_T,
            )

            # Launch attn @ Kc per head
            BLOCK_D = 128
            out = torch.empty((H, Dq), dtype=torch.float32, device=device)

            attn_matmul_kernel[(H,)](
                attn, Kc, out, H=H, T=T, Dq=Dq_const, BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T,
            )

            # Assign per-head outputs to output tensor
            output[b] = out

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
