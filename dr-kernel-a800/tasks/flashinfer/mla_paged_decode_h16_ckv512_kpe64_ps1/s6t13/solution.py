import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    H,              # int32
    D: tl.constexpr,          # 512
    Dp: tl.constexpr,         # 64
    L_tokens,       # int32
    b,              # int32
):
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Load qn row for head h: [D]
    offs = tl.arange(0, D)
    qn_row = tl.load(qn_ptr + h * D + offs)

    # Load qp row for head h: [Dp]
    offs2 = tl.arange(0, Dp)
    qp_row = tl.load(qp_ptr + h * Dp + offs2)

    # Load Kc row for token t: [D]
    Kc_row = tl.load(Kc_ptr + t * D + offs)

    # Load Kp row for token t: [Dp]
    Kp_row = tl.load(Kp_ptr + t * Dp + offs2)

    # Compute dot products
    acc1 = 0.0
    for kk in range(0, D):
        acc1 += qn_row[kk] * Kc_row[kk]
    acc2 = 0.0
    for kk in range(0, Dp):
        acc2 += qp_row[kk] * Kp_row[kk]

    logit = acc1 + acc2
    idx = (b * H + h) * L_tokens + t
    tl.store(logits_ptr + idx, logit)


@triton.jit
def compute_lse_per_head_kernel(
    logits_ptr,     # *fp32, [B*H*L_tokens]
    lse_ptr,        # *fp32, [B*H]
    H,              # int32
    L_tokens,       # int32
    b,              # int32
    sm_scale,       # fp32
):
    h = tl.program_id(0)
    # Load logits[b, h, :] into a vector
    offs = tl.arange(0, L_tokens)
    base = (b * H + h) * L_tokens
    logits_row = tl.load(logits_ptr + base + offs)
    # Scale
    logits_scaled = logits_row * sm_scale
    # Numerical stability: max
    m = tl.max(logits_scaled, axis=0)
    # sum exp(logits_scaled - m)
    expv = tl.exp(logits_scaled - m)
    sum_exp = tl.sum(expv, axis=0)
    # lse = log(sum_exp) / log(2)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / log(2)
    tl.store(lse_ptr + b * H + h, lse_val)


@triton.jit
def compute_softmax_per_head_kernel(
    logits_ptr,     # *fp32, [B*H*L_tokens]
    softmax_ptr,    # *fp32, [B*H*L_tokens]
    H,              # int32
    L_tokens,       # int32
    b,              # int32
    sm_scale,       # fp32
):
    h = tl.program_id(0)
    offs = tl.arange(0, L_tokens)
    base = (b * H + h) * L_tokens
    logits_row = tl.load(logits_ptr + base + offs)
    logits_scaled = logits_row * sm_scale
    m = tl.max(logits_scaled, axis=0)
    expv = tl.exp(logits_scaled - m)
    sum_exp = tl.sum(expv, axis=0)
    softmax_row = expv / sum_exp
    tl.store(softmax_ptr + base + offs, softmax_row)


@triton.jit
def triton_matmul_softmax_kc_kernel(
    softmax_ptr,    # *fp32, [B*H*L_tokens]
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    output_ptr,     # *fp32, [B*H*D] (we'll cast to bfloat16 on host)
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
    b,              # int32
):
    # Grid (h in [0..H-1], d in [0..D-1])
    h = tl.program_id(0)
    d = tl.program_id(1)

    acc = 0.0
    # Loop over tokens t
    for t in range(0, L_tokens):
        # softmax[b, h, t]
        offs = tl.arange(0, L_tokens)
        base = (b * H + h) * L_tokens
        s = tl.load(softmax_ptr + base + offs)[t]
        # Kc[b, t, d]
        kc = tl.load(Kc_ptr + t * D + d)
        acc += s * kc

    # Store as float32; host will cast to bfloat16
    tl.store(output_ptr + (b * H + h) * D + d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and dtype
        device = q_nope.device
        assert device.type == 'cuda', "Inputs must be on CUDA device for Triton."

        # Cast to fp32 for computation
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        ckv_cache_f32 = ckv_cache.to(torch.float32).squeeze(1).contiguous()  # [N_total, D]
        kpe_cache_f32 = kpe_cache.to(torch.float32).squeeze(1).contiguous()  # [N_total, Dp]

        B = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]  # 16
        D = q_nope_f32.shape[2]  # 512
        Dp = q_pe_f32.shape[2]    # 64

        # Compute per-batch L_tokens from kv_indptr
        # kv_indptr[b] and kv_indptr[b+1] mark the start and end of this batch's tokens.
        start = kv_indptr[:B].to(torch.int32)
        end = kv_indptr[1:(B + 1)].to(torch.int32)
        L_tokens = (end - start).tolist()  # list of int for each batch
        assert all(isinstance(x, int) for x in L_tokens), "L_tokens must be a list of ints per batch."

        # For each batch, slice kv_indices according to L_tokens and compute per-batch Kc/Kp
        # Prepare buffers for logits, softmax, and output
        logits = torch.empty((B, H, D), dtype=torch.float32, device=device)
        # Flatten for pointer arithmetic
        logits_flat = logits.view(-1)  # [B*H*L_tokens]

        # Allocate lse
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Allocate softmax buffer
        softmax = torch.empty((B, H, D), dtype=torch.float32, device=device)
        softmax_flat = softmax.view(-1)

        # Allocate output (fp32 for compute, cast later)
        output_fp32 = torch.empty((B, H, D), dtype=torch.float32, device=device)

        # Launch Triton kernels for each batch b
        for b in range(B):
            lb = L_tokens[b]
            # Slice kv_indices for this batch
            tok_idx = kv_indices[start[b]:start[b] + lb].to(torch.int32).to(device)  # [lb]
            # Gather Kc and Kp for this batch
            # Kc_all[tok_idx] -> [lb, D]; Kp_all[tok_idx] -> [lb, Dp]
            Kc_b = ckv_cache_f32[tok_idx]  # [lb, D], contiguous
            Kp_b = kpe_cache_f32[tok_idx]  # [lb, Dp], contiguous

            # Launch logits kernel: grid (H, lb)
            grid_logits = (H, lb)
            compute_logits_per_batch_kernel[grid_logits](
                q_nope_f32[b], q_pe_f32[b], Kc_b, Kp_b, logits_flat,
                H=H, D=512, Dp=64, L_tokens=lb, b=b
            )

            # Launch lse kernel: grid (1,) but per (b, h) => use B,H grid if needed; here one call per b,h => loop over h
            # We will launch one per head h
            for h in range(H):
                compute_lse_per_head_kernel[(1,)](
                    logits_flat, lse[b, h], H=H, L_tokens=lb, b=b, sm_scale=float(sm_scale)
                )

            # Compute softmax per head h (we'll reuse lse to recompute? For correctness, recompute softmax from logits)
            # We need logits[b, :, :] row; we'll recompute softmax from logits
            # Launch softmax kernel: grid (1,) per head h
            for h in range(H):
                compute_softmax_per_head_kernel[(1,)](
                    logits_flat, softmax_flat, H=H, L_tokens=lb, b=b, sm_scale=float(sm_scale)
                )

            # Compute output per head h via matmul kernel: grid (H, D)
            for h in range(H):
                triton_matmul_softmax_kc_kernel[(1, 512)](
                    softmax_flat, Kc_b, output_fp32[b, h], H=H, D=512, L_tokens=lb, b=b
                )

        # Return output in bfloat16 and lse
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
