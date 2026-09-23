import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def compute_logits_scaled_kernel(
    Qn_ptr, Qp_ptr, Kc_all_ptr, Kp_all_ptr, Indices_ptr,
    Logits_ptr,
    num_tokens, num_qo_heads, topk,
    head_dim_ckv, head_dim_kpe,
    sm_scale,
    BLOCK_D: tl.constexpr,
    BLOCK_DP: tl.constexpr,
):
    # Each program computes one (t, h, k) triplet's scaled logit contribution.
    # We structure the grid over (t, h, k). Triton supports 3D grid; we compute t = pid0, h = pid1, k = pid2.
    t = tl.program_id(0)
    h = tl.program_id(1)
    k = tl.program_id(2)

    # Load index for this (t, k)
    idx = tl.load(Indices_ptr + t * topk + k)
    valid = idx != -1

    # Initialize accumulators
    acc1 = 0.0  # fp32
    acc2 = 0.0  # fp32

    # Compute contrib1 = sum_d Qn[t,h,d] * Kc_all[idx,d]
    d = 0
    while d < head_dim_ckv:
        offs_d = d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim_ckv
        qn_vals = tl.load(Qn_ptr + t * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv + offs_d, mask=mask_d, other=0.0)
        kc_vals = tl.load(Kc_all_ptr + idx * head_dim_ckv + offs_d, mask=mask_d, other=0.0)
        acc1 += tl.sum(qn_vals * kc_vals, axis=0)
        d += BLOCK_D

    # Compute contrib2 = sum_dp Qp[t,h,dp] * Kp_all[idx,dp]
    dp = 0
    while dp < head_dim_kpe:
        offs_dp = dp + tl.arange(0, BLOCK_DP)
        mask_dp = offs_dp < head_dim_kpe
        qp_vals = tl.load(Qp_ptr + t * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe + offs_dp, mask=mask_dp, other=0.0)
        kp_vals = tl.load(Kp_all_ptr + idx * head_dim_kpe + offs_dp, mask=mask_dp, other=0.0)
        acc2 += tl.sum(qp_vals * kp_vals, axis=0)
        dp += BLOCK_DP

    contrib = (acc1 + acc2) * sm_scale
    # Store to logits[t, h, k] in a contiguous buffer
    tl.store(Logits_ptr + (t * num_qo_heads + h) * topk + k, contrib)


def _compute_logits_triton(q_nope, q_pe, Kc_all, Kp_all, Indices, sm_scale):
    """
    Compute logits_scaled[t, h, k] = (q_nope[t, h] @ Kc_all[Indices[t, k]]) + (q_pe[t, h] @ Kp_all[Indices[t, k]])
    scaled by sm_scale. Return a 3D tensor [num_tokens, num_qo_heads, topk] in fp32.
    """
    device = q_nope.device
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = Indices.shape[-1]

    # Prepare inputs
    q_nope_f32 = q_nope.to(torch.float32)
    q_pe_f32 = q_pe.to(torch.float32)
    Kc_all_f32 = Kc_all.to(torch.float32)
    Kp_all_f32 = Kp_all.to(torch.float32)
    Indices_i32 = Indices.to(torch.int32)

    logits = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)

    # Launch Triton: 3D grid over (t, h, k)
    grid = (num_tokens, num_qo_heads, topk)
    compute_logits_scaled_kernel[grid](
        q_nope_f32, q_pe_f32, Kc_all_f32, Kp_all_f32, Indices_i32,
        logits,
        num_tokens, num_qo_heads, topk,
        head_dim_ckv, head_dim_kpe,
        float(sm_scale),
        BLOCK_D=128, BLOCK_DP=64,
    )
    return logits


def _run_triton_and_torch(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Full forward that uses Triton for logits and PyTorch for per-group softmax/logsumexp and final output.
    """
    device = q_nope.device

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Flatten paged KV cache to token-level in fp32
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [num_tokens * num_pages, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [num_tokens * num_pages, head_dim_kpe]

    # Indices
    Indices = sparse_indices.to(torch.int32)

    # Compute logits with Triton
    logits = _compute_logits_triton(q_nope, q_pe, Kc_all, Kp_all, Indices, sm_scale)  # [num_tokens, num_qo_heads, topk]

    # Per-group softmax: group_size = 32 since head_dim_ckv == num_qo_heads * 32
    group_size = head_dim_ckv // num_qo_heads  # should be 32
    assert head_dim_ckv % num_qo_heads == 0, "head_dim_ckv must be divisible by num_qo_heads"
    # Reshape logits to [num_tokens, num_qo_heads, num_groups, group_size]
    num_groups = topk // (head_dim_ckv // num_qo_heads)
    logits_reshaped = logits.view(num_tokens, num_qo_heads, num_groups, group_size)  # [num_tokens, num_qo_heads, 64, 32]

    # Per-group logsumexp and then softmax in PyTorch
    # Note: original divides by log(2); keep exact behavior
    ln2 = math.log(2.0)
    # lse per (t, h, g): max over group + log(sum(exp(group - max)))
    # Compute per-group logsumexp
    m = logits_reshaped - logits_reshaped  # initialize
    # We need to compute max and sumexp over last dim (group_size)
    m = logits_reshaped  # placeholder; compute m using max
    m = logits_reshaped.max(dim=3, keepdim=True)[0]
    sumexp = torch.sum(torch.exp(logits_reshaped - m), dim=3, keepdim=True)
    lse = m + torch.log(sumexp) / ln2  # [num_tokens, num_qo_heads, num_groups, 1]

    # attn = softmax(logits_scaled, per group)
    attn = torch.exp(logits_reshaped - lse)  # broadcast subtract

    # Final output per head: sum over its group of attn rows multiplied by Kc rows
    output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)

    for t in range(num_tokens):
        for h in range(num_qo_heads):
            # For each group g, accumulate over 32 tokens
            for g in range(num_groups):
                # attn[t, h, g, :] shape [group_size], Kc_all[ Indices[t, g*group_size + offs], :]
                # Build the 32-element vector by looping offs in 0..31
                group_start = g * group_size
                for offs in range(group_size):
                    k_idx = group_start + offs
                    idx = int(Indices[t, k_idx].item())
                    attn_val = float(attn[t, h, g, offs].item())
                    kc_row = Kc_all[idx]  # [head_dim_ckv], fp32
                    # Add to output[t, h, :]
                    output[t, h] += attn_val * kc_row

    # lse shape: [num_tokens, num_qo_heads]
    lse = lse.squeeze(-1)  # remove last dim

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure inputs are contiguous and on device
        q_nope = q_nope.contiguous().to(torch.float32)
        q_pe = q_pe.contiguous().to(torch.float32)
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        sparse_indices = sparse_indices.contiguous()

        output, lse = _run_triton_and_torch(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

        return output, lse


def run(*args):
    return ModelNew()(*args)
