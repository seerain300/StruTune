import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [N_total, D], contiguous
    Kp_ptr,         # *fp32, shape [N_total, Dp], contiguous
    tok_idx_ptr,    # *int32, shape [L_tokens]
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    H: tl.constexpr,
    D: tl.constexpr,
    Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    b: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # 2D grid over heads and tokens
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Load qn row for head h: [D]
    qn_row = tl.load(qn_ptr + h * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    # Load qp row for head h: [Dp]
    k = tl.arange(0, Dp)
    qp_row = tl.load(qp_ptr + h * Dp + k, mask=k < Dp, other=0.0)

    # Gather token indices
    tok_idx_t = tl.load(tok_idx_ptr + t)

    # Load Kc row for token t: [D]
    Kc_row = tl.load(Kc_ptr + tok_idx_t * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    # Load Kp row for token t: [Dp]
    Kp_row = tl.load(Kp_ptr + tok_idx_t * Dp + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)

    # Compute dot products
    acc1 = 0.0
    for kk in range(0, D):
        acc1 += qn_row[kk] * Kc_row[kk]
    acc2 = 0.0
    for kk in range(0, Dp):
        acc2 += qp_row[kk] * Kp_row[kk]
    logits_val = acc1 + acc2

    # Store logits[b, h, t] as float32
    offset = b * (H * L_tokens) + h * L_tokens + t
    tl.store(logits_ptr + offset, logits_val)


@triton.jit
def lse_per_head_kernel(
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    lse_ptr,        # *fp32, shape [B*H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
    b: tl.constexpr,
    sm_scale: tl.constexpr,
):
    h = tl.program_id(0)
    offset = b * (H * L_tokens) + h * L_tokens

    # Compute max over scaled logits
    max_scaled = -float('inf')
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + offset + t)
        scaled = val * sm_scale
        if scaled > max_scaled:
            max_scaled = scaled

    # Compute sumexp of scaled logits
    sumexp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + offset + t)
        scaled = val * sm_scale
        sumexp += tl.exp(scaled - max_scaled)

    lse_val = max_scaled + tl.log(sumexp) / tl.log(2.0)
    tl.store(lse_ptr + b * H + h, lse_val)


@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    Kc_ptr,         # *fp32, shape [N_total, D], contiguous
    tok_idx_ptr,    # *int32, shape [L_tokens]
    output_ptr,     # *bf16, shape [B*H*D], contiguous
    H: tl.constexpr,
    D: tl.constexpr,
    L_tokens: tl.constexpr,
    b: tl.constexpr,
    sm_scale: tl.constexpr,
):
    h = tl.program_id(0)
    offset = b * (H * L_tokens) + h * L_tokens

    # Compute max over scaled logits for numerical stability
    max_scaled = -float('inf')
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + offset + t)
        scaled = val * sm_scale
        if scaled > max_scaled:
            max_scaled = scaled

    # Compute denominator sumexp
    sumexp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + offset + t)
        scaled = val * sm_scale
        sumexp += tl.exp(scaled - max_scaled)
    inv_sumexp = 1.0 / sumexp

    # Accumulate output[h, :]
    acc = tl.zeros((D,), dtype=tl.float32)
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + offset + t)
        scaled = val * sm_scale
        attn_t = tl.exp(scaled - max_scaled) * inv_sumexp
        tok_idx_t = tl.load(tok_idx_ptr + t)
        Kc_row = tl.load(Kc_ptr + tok_idx_t * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
        acc += Kc_row * attn_t

    # Store acc as bfloat16 to output[b, h, :]
    out_offset = b * (H * D) + h * D
    acc_bf16 = acc.to(tl.bfloat16)
    for kk in range(0, D):
        tl.store(output_ptr + out_offset + kk, acc_bf16[kk])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be CUDA tensors"

        B, H, D = q_nope.shape
        Dp = q_pe.shape[-1]
        assert H == 16 and D == 512 and Dp == 64, "Expected head configurations: H=16, D=512, Dp=64"

        # Cast to float32 for compute; make contiguous
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [N_total, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [N_total, Dp]
        # tok_idx per batch element
        start = kv_indptr[:B].to(torch.int32)
        end = kv_indptr[B:].to(torch.int32)
        L_tokens = (end - start).to(torch.int32).tolist()  # per-batch L_tokens
        tok_idx_list = [kv_indices[start[i]:end[i]].to(torch.int32).contiguous() for i in range(B)]

        # Allocate outputs
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # For each batch element, run Triton kernels
        for b in range(B):
            lb = int(L_tokens[b])
            tok_idx_b = tok_idx_list[b]  # [lb], int32

            # Allocate logits_flat buffer [H*lb]
            logits_flat = torch.empty((H * lb), dtype=torch.float32, device=device)

            # Kernel 1: compute logits for this batch
            compute_logits_per_batch_kernel[(H, lb)](
                q_nope_f32[b], q_pe_f32[b], Kc_all, Kp_all, tok_idx_b, logits_flat,
                H=H, D=D, Dp=Dp, L_tokens=lb, b=b, sm_scale=float(sm_scale),
                num_warps=4, num_stages=2
            )

            # Kernel 2: compute lse per head for this batch
            lse_per_head_kernel[(H,)](
                logits_flat, lse[b],
                H=H, L_tokens=lb, b=b, sm_scale=float(sm_scale),
                num_warps=2, num_stages=1
            )

            # Kernel 3: compute output per head for this batch
            compute_output_per_batch_kernel[(H,)](
                logits_flat, Kc_all, tok_idx_b, output[b],
                H=H, D=D, L_tokens=lb, b=b, sm_scale=float(sm_scale),
                num_warps=4, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
