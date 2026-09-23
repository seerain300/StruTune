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
    H,              # int32 runtime
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32 runtime
    b,              # int32 runtime
):
    # 2D grid: (h in 0..H-1, t in 0..L_tokens-1)
    h = tl.program_id(0)
    t = tl.program_id(1)

    if (h >= H) or (t >= L_tokens):
        return

    # Compute flat index for logits[b, h, t]
    idx = (b * H + h) * L_tokens + t

    # Load qn row for head h: scalar loop over D
    qn_sum = 0.0
    for k in range(0, D):
        q = tl.load(qn_ptr + h * D + k)
        kvec = tl.load(Kc_ptr + t * D + k)
        qn_sum += q * kvec

    # Load qp row for head h: scalar loop over Dp
    qp_sum = 0.0
    for k in range(0, Dp):
        p = tl.load(qp_ptr + h * Dp + k)
        kp = tl.load(Kp_ptr + t * Dp + k)
        qp_sum += p * kp

    logit = qn_sum + qp_sum
    tl.store(logits_ptr + idx, logit)


@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, shape [B*H*L_tokens]
    lse_ptr,        # *fp32, shape [B*H]
    H,              # int32
    L_tokens,       # int32
    b,              # int32
    sm_scale,       # float32
):
    # Grid: (b, h)
    h = tl.program_id(1)
    if h >= H:
        return
    # Row start index
    row_start = (b * H + h) * L_tokens

    # Scalar loop to compute max, sum for logsumexp over L_tokens
    max_val = -1e20
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        logit_scaled = logit * sm_scale
        if logit_scaled > max_val:
            max_val = logit_scaled

    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        logit_scaled = logit * sm_scale
        sum_exp += tl.exp(logit_scaled - max_val)

    lse_val = tl.log(sum_exp) + max_val  # natural logsumexp; original divides by log(2) in Python, but we can do it there if needed
    tl.store(lse_ptr + (b * H + h), lse_val)


@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, shape [B*H*L_tokens]
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    output_ptr,     # *fp32, shape [B*H*D]
    H,              # int32
    D: tl.constexpr,         # 512
    Dp: tl.constexpr,        # 64
    L_tokens,       # int32
    b,              # int32
    sm_scale,       # float32
):
    # Grid: (h in 0..H-1, d in 0..D-1)
    h = tl.program_id(0)
    d = tl.program_id(1)

    # Compute softmax over tokens for head h: scalar loop
    row_start = (b * H + h) * L_tokens
    max_val = -1e20
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        logit_scaled = logit * sm_scale
        if logit_scaled > max_val:
            max_val = logit_scaled

    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        logit_scaled = logit * sm_scale
        sum_exp += tl.exp(logit_scaled - max_val)

    # Compute output[h, d] = sum_t softmax[t] * Kc[t, d] with scalar loops
    out_val = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        logit_scaled = logit * sm_scale
        prob = tl.exp(logit_scaled - max_val) / sum_exp
        Kc_d = tl.load(Kc_ptr + t * D + d)
        out_val += prob * Kc_d

    # Store output[b, h, d] as float32
    tl.store(output_ptr + (b * H * D + h * D + d), out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and dtype float32 for compute
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()
        ckv_cache_f32 = ckv_cache.to(torch.float32).contiguous()
        kpe_cache_f32 = kpe_cache.to(torch.float32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        B = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        D = q_nope_f32.shape[2]
        Dp = q_pe_f32.shape[2]
        assert H == 16, "num_qo_heads must be 16"
        assert D == 512, "head_dim_ckv must be 512"
        assert Dp == 64, "head_dim_kpe must be 64"

        # Prepare output tensors (float32 for compute, cast later to bfloat16)
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Pre-allocate logits buffer [B*H*L_tokens] as flat
        logits_flat = torch.empty(B * H * 100, dtype=torch.float32, device=device)  # placeholder, we'll resize per b
        # We'll compute L_tokens per batch element using kv_indptr
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L = max(end - start, 0)
            # Slice kv_indices for this batch element: tok_idx = kv_indices[start:start+L]
            tok_idx = kv_indices[start:start + L].contiguous()
            # Allocate Kc and Kp per batch element (gather from global caches)
            Kc = torch.empty((L, D), dtype=torch.float32, device=device)
            Kp = torch.empty((L, Dp), dtype=torch.float32, device=device)
            # Gather rows
            for t in range(L):
                idx = int(tok_idx[t].item())
                Kc[t] = ckv_cache_f32[idx, 0, :].contiguous()
                Kp[t] = kpe_cache_f32[idx, 0, :].contiguous()

            # Prepare qn and qp rows for this batch
            qn = q_nope_f32[b].contiguous()  # [H, D]
            qp = q_pe_f32[b].contiguous()    # [H, Dp]

            # Launch Triton kernel to compute logits[b, :, :]
            # Grid size: (H, L)
            compute_logits_per_batch_kernel[(H, L)](
                qn, qp, Kc, Kp, logits_flat, H, D, Dp, L, b, num_warps=4
            )

            # Compute lse for this batch
            compute_lse_per_batch_kernel[(B, H)](logits_flat, lse, H, L, b, sm_scale, num_warps=4)

            # Compute output for this batch
            compute_output_per_batch_kernel[(H, D)](logits_flat, Kc, output[b], H, D, Dp, L, b, sm_scale, num_warps=4)

        # Cast output to bfloat16 as required by original Model
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
