import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    q_nope_ptr,      # *f32, shape [H, Dq]
    q_pe_ptr,        # *f32, shape [H, Dp]
    Kc_ptr,          # *f32, shape [T, Dq]
    Kp_ptr,          # *f32, shape [T, Dp]
    logits_ptr,      # *f32, shape [H, T] (stored linearized as [H*T])
    H: tl.constexpr, # num heads (compile-time for grid scheduling)
    Dq: tl.constexpr, # 512
    Dp: tl.constexpr, # 64
    T,               # int, runtime number of tokens
    sm_scale,        # f32, runtime
    BLOCK_T: tl.constexpr = 256,
):
    h = tl.program_id(0)
    # Preload q vectors using compile-time aranges
    offs_dq = tl.arange(0, Dq)  # [Dq]
    qn = tl.load(q_nope_ptr + h * Dq + offs_dq)  # [Dq]
    offs_dp = tl.arange(0, Dp)  # [Dp]
    qp = tl.load(q_pe_ptr + h * Dp + offs_dp)    # [Dp]

    offs_t = tl.arange(0, BLOCK_T)  # [BLOCK_T]
    # Write to logits_ptr[h * T + t_idx]
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        sum_vec = tl.zeros([BLOCK_T], dtype=tl.float32)
        # dot over Dq and Dp
        for d in range(0, Dq):
            sum_vec += tl.load(Kc_ptr + t_idx * Dq + d) * qn[d]
        for d in range(0, Dp):
            sum_vec += tl.load(Kp_ptr + t_idx * Dp + d) * qp[d]
        sum_vec = sum_vec * sm_scale
        tl.store(logits_ptr + h * T + t_idx, sum_vec, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_ptr,      # *f32, shape [H, T] (linearized as [H*T])
    attn_ptr,        # *f32, shape [H, T] (linearized as [H*T])
    H: tl.constexpr,
    T,               # int
    BLOCK_T: tl.constexpr = 256,
):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    # Compute row max for stability
    max_val = tl.full([], -float("inf"), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, tile_max)
    # Compute denominator
    denom = tl.zeros([], dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        e = tl.exp(x - max_val)
        denom += tl.sum(e, axis=0)
    # Write normalized attention
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        e = tl.exp(x - max_val)
        attn = e / denom
        tl.store(attn_ptr + h * T + t_idx, attn, mask=mask_t)


@triton.jit
def lse_row_kernel(
    logits_ptr,      # *f32, shape [H, T] (linearized as [H*T])
    lse_ptr,         # *f32, shape [H]
    H: tl.constexpr,
    T,               # int
    ln2,             # f32, runtime constant ln(2)
    BLOCK_T: tl.constexpr = 256,
):
    h = tl.program_id(0)
    # First pass: max
    max_val = tl.full([], -float("inf"), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        x = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, tile_max)
    # Second pass: sum exp(x - max)
    sum_exp = tl.zeros([], dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_idx < T
        x = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)
    lse_h = tl.log(sum_exp) + max_val  # logsumexp
    lse_h = lse_h / ln2                # divide by ln(2)
    tl.store(lse_ptr + h, lse_h)


@triton.jit
def attn_matmul_kernel(
    attn_ptr,        # *f32, shape [H, T] (linearized as [H*T])
    Kc_ptr,          # *f32, shape [T, D]
    out_ptr,         # *f32, shape [H, D]
    H: tl.constexpr,
    T,               # int
    D: tl.constexpr, # 512
    BLOCK_T: tl.constexpr = 256,
    BLOCK_D: tl.constexpr = 128,
):
    h = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + offs_d
        mask_d = d_idx < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + tl.arange(0, BLOCK_T)
            mask_t = t_idx < T
            attn_chunk = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            Kc_chunk = tl.load(
                Kc_ptr + t_idx[:, None] * D + d_idx[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            )  # [BLOCK_T, BLOCK_D]
            acc += tl.sum(Kc_chunk * attn_chunk[:, None], axis=0)
        tl.store(out_ptr + h * D + d_idx, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device for Triton
        device = q_nope.device
        if device.type != 'cuda':
            device = torch.device('cuda')
        # Move inputs to CUDA and cast to float32 for compute
        q_nope_f32 = q_nope.to(device).contiguous().to(torch.float32)   # [B, H, Dq]
        q_pe_f32 = q_pe.to(device).contiguous().to(torch.float32)      # [B, H, Dp]
        Kc_all = ckv_cache.squeeze(1).to(device).contiguous().to(torch.float32)  # [N, Dq]
        Kp_all = kpe_cache.squeeze(1).to(device).contiguous().to(torch.float32)  # [N, Dp]

        B = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        Dq = q_nope_f32.shape[2]
        Dp = q_pe_f32.shape[2]

        # Output tensors
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # [B, H, Dq]
        lse = torch.empty((B, H), dtype=torch.float32, device=device)         # [B, H]

        for b in range(B):
            # Sparse range for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg

            if L_tokens <= 0:
                # No tokens used; output zeros and lse -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather Kc and Kp for used tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L_tokens]
            Kc = Kc_all[tok_idx]  # [L_tokens, Dq]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Allocate per-batch buffers
            logits_flat = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)  # [H*L_tokens]
            attn_flat = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)    # [H*L_tokens]

            # q vectors per head for this batch
            q_nope_b = q_nope_f32[b]  # [H, Dq]
            q_pe_b = q_pe_f32[b]      # [H, Dp]

            # Launch fused logits kernel: one program per head
            grid = (H,)
            fused_logits_kernel[grid](
                q_nope_b, q_pe_b, Kc, Kp, logits_flat, H, Dq, Dp, L_tokens, sm_scale,
                BLOCK_T=256,
            )

            # Reshape logits to [H, L_tokens]
            logits = logits_flat.view(H, L_tokens)

            # Compute LSE per head: logsumexp(scaled_logits) / ln(2)
            ln2 = math.log(2.0)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](logits, lse[b], H, L_tokens, ln2, BLOCK_T=256)

            # Compute attention per head: softmax(scaled_logits)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](logits, attn_flat, H, L_tokens, BLOCK_T=256)

            # Reshape attn to [H, L_tokens]
            attn = attn_flat.view(H, L_tokens)

            # Compute output per head: attn @ Kc -> [H, Dq]
            out_batch = torch.empty((H, Dq), dtype=torch.float32, device=device)  # [H, Dq]
            attn_matmul_kernel[(H,)](
                attn, Kc, out_batch, H, L_tokens, Dq, BLOCK_T=256, BLOCK_D=128
            )

            # Store to output
            output[b] = out_batch

        # Return outputs matching original types: output in bfloat16, lse in float32
        output_bf16 = output.to(torch.bfloat16)  # [B, H, Dq] bfloat16
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
