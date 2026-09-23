import math
import torch
import triton
import triton.language as tl


# Kernel 1: Compute logits_scaled[t, h, k] = (q_nope[t, h] @ Kc_all[indices[k]] + q_pe[t, h] @ Kp_all[indices[k]]) * sm_scale
@triton.jit
def compute_logits_scaled_kernel(
    q_nope_ptr, q_pe_ptr, Kc_all_ptr, Kp_all_ptr, indices_ptr,
    logits_scaled_ptr,
    num_tokens, num_qo_heads,
    sm_scale: tl.float32,
    TOPK: tl.constexpr, HEAD_DIM_CKV: tl.constexpr,
):
    t = tl.program_id(0)  # token
    h = tl.program_id(1)  # head

    # Load query vectors (float32)
    qn = tl.load(q_nope_ptr + t * num_qo_heads + h)  # shape [HEAD_DIM_CKV], float32
    qp = tl.load(q_pe_ptr + t * num_qo_heads + h)    # shape [HEAD_DIM_KPE], float32

    # Initialize logits_scaled buffer
    # logits_scaled layout: [num_tokens, num_qo_heads, TOPK], row-major over last dim
    base = logits_scaled_ptr + t * (num_qo_heads * TOPK) + h * TOPK

    for k in range(TOPK):
        idx = tl.load(indices_ptr + t * TOPK + k)
        valid = idx != -1

        # If invalid, set contribs to 0; otherwise compute dot products
        # Kc_all_ptr is [num_tokens * TOPK, HEAD_DIM_CKV]
        # Kp_all_ptr is [num_tokens * TOPK, HEAD_DIM_KPE]
        # We need to read the row corresponding to (t, k) via indices. The flatten index is tok_idx = t * TOPK + k, but we use indices[k].
        if valid:
            # For valid indices, read Kc row and Kp row at idx
            # idx is the row offset in Kc_all/Kp_all
            Kc_row = tl.load(Kc_all_ptr + idx * HEAD_DIM_CKV)  # [HEAD_DIM_CKV]
            Kp_row = tl.load(Kp_all_ptr + idx * HEAD_DIM_KPE)  # [HEAD_DIM_KPE]

            contrib1 = 0.0
            # Sum qn * Kc_row
            for d in range(HEAD_DIM_CKV):
                contrib1 += qn[d] * Kc_row[d]
            contrib2 = 0.0
            # Sum qp * Kp_row
            for d in range(HEAD_DIM_KPE):
                contrib2 += qp[d] * Kp_row[d]
            scaled = (contrib1 + contrib2) * sm_scale
        else:
            scaled = 0.0

        tl.store(base + k, scaled)


# Kernel 2: Compute lse per (t, h) = logsumexp(logits_scaled[t, h, :]) / ln(2) with two-pass Triton
@triton.jit
def compute_lse_and_attn_kernel(
    logits_scaled_ptr, attn_ptr,
    num_tokens, num_qo_heads, TOPK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    base = logits_scaled_ptr + t * (num_qo_heads * TOPK) + h * TOPK

    # Pass 1: compute max m and sum_exp
    m = -float("inf")
    sum_exp = 0.0
    for k in range(TOPK):
        s = tl.load(base + k)
        m = tl.maximum(m, s)
        # sum_exp += exp(s - m)
        sum_exp += tl.exp(s - m)
    ln2 = 0.6931471805599453  # float64 literal; Triton will promote to float32 as needed
    lse = m + tl.log(sum_exp) / ln2  # float32

    # Pass 2: compute attn_k = exp(logits_scaled - lse) and store
    base_attn = attn_ptr + t * (num_qo_heads * TOPK) + h * TOPK
    for k in range(TOPK):
        s = tl.load(base + k)
        a = tl.exp(s - lse)
        tl.store(base_attn + k, a)


# Kernel 3: Compute output[t, h, :] = sum_k attn[t, h, k] * Kc_all[indices[t, k], :]
@triton.jit
def compute_output_kernel(
    Kc_all_ptr, indices_ptr, attn_ptr, output_ptr,
    num_tokens, num_qo_heads, HEAD_DIM_CKV: tl.constexpr, TOPK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # output_ptr is [num_tokens, num_qo_heads, HEAD_DIM_CKV], row-major over last dim
    out_row = output_ptr + t * (num_qo_heads * HEAD_DIM_CKV) + h * HEAD_DIM_CKV

    # Accumulate in float32 for numerical stability
    out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)

    base_attn = attn_ptr + t * (num_qo_heads * TOPK) + h * TOPK
    for k in range(TOPK):
        idx = tl.load(indices_ptr + t * TOPK + k)
        valid = idx != -1
        if valid:
            attn_k = tl.load(base_attn + k)  # float32
            Kc_row = tl.load(Kc_all_ptr + idx * HEAD_DIM_CKV)  # [HEAD_DIM_CKV], float32
            out_vec += attn_k * Kc_row

    # Store as float32 (ModelNew.forward will cast to bfloat16 if needed)
    for d in range(HEAD_DIM_CKV):
        tl.store(out_row + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; Triton kernels do all work.

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        q_nope: [num_tokens, num_qo_heads, head_dim_ckv] bfloat16
        q_pe:   [num_tokens, num_qo_heads, head_dim_kpe] bfloat16
        ckv_cache: [num_pages, page_size, head_dim_ckv] bfloat16
        kpe_cache: [num_pages, page_size, head_dim_kpe] bfloat16
        sparse_indices: [num_tokens, topk] int32 (can contain -1 as padding)
        sm_scale: float32 scalar
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "Tensors must be on CUDA"
        device = q_nope.device

        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, page_size, _ = ckv_cache.shape
        topk = sparse_indices.shape[-1]

        # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]
        total_kv_tokens = Kc_all.shape[0]
        assert Kp_all.shape[0] == total_kv_tokens, "Kp_all and Kc_all must have same length"

        # Cast queries to float32 for stable math
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)

        # Ensure sparse_indices is int32 and contiguous
        indices = sparse_indices  # already int32
        assert indices.dtype == torch.int32 and indices.device == device

        # Allocate outputs (float32 for numerical fidelity; cast later if needed)
        logits_scaled = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        attn = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

        # Launch Triton kernels
        # Compute logits_scaled
        grid_logits = (num_tokens, num_qo_heads)
        compute_logits_scaled_kernel[grid_logits](
            q_nope_f32, q_pe_f32, Kc_all, Kp_all, indices,
            logits_scaled,
            num_tokens, num_qo_heads,
            float(sm_scale),
            TOPK=topk, HEAD_DIM_CKV=head_dim_ckv,
            num_warps=1, num_stages=1
        )

        # Compute lse and attn
        grid_lse = (num_tokens, num_qo_heads)
        compute_lse_and_attn_kernel[grid_lse](
            logits_scaled, attn,
            num_tokens, num_qo_heads,
            TOPK=topk,
            num_warps=1, num_stages=1
        )

        # Compute final output
        grid_out = (num_tokens, num_qo_heads)
        compute_output_kernel[grid_out](
            Kc_all, indices, attn, output,
            num_tokens, num_qo_heads,
            HEAD_DIM_CKV=head_dim_ckv, TOPK=topk,
            num_warps=1, num_stages=1
        )

        # Cast output to bfloat16 to match original model's output dtype
        output_bf16 = output.to(torch.bfloat16)

        # Return output and lse (compute lse explicitly in PyTorch from attn to match original signature)
        # We need lse as float32 per (t, h): lse[t, h] = logsumexp(logits_scaled[t, h, :]) / ln(2).
        # However, since we already computed attn in Triton, we can derive lse in PyTorch from attn:
        # lse = log(sum(exp(logits_scaled))) / ln(2), but we don't have logits_scaled saved.
        # Instead, recompute lse via attn using the identity: sum_k attn_k * 1 = 1, and lse = log(sum(exp(logits_scaled))) / ln(2) equals
        # the quantity we need. But to be exact, we can compute lse directly via torch.logsumexp on logits_scaled saved earlier.
        # Since we did not save logits_scaled for lse in Triton, we will compute lse from attn:
        # We can derive: let s_k = logits_scaled_k, a_k = exp(s_k - lse). Then sum_k a_k = exp(-lse) * sum_k exp(s_k).
        # We cannot recover lse this way without s_k. Therefore, we compute lse using torch.logsumexp on logits_scaled saved:
        # But we already have logits_scaled in a variable. Compute lse via torch to ensure correctness.
        logits_scaled_cpu = logits_scaled.cpu()  # we can do this on device too: torch.logsumexp(logits_scaled, dim=2)/math.log(2.0)
        # Compute lse in torch on device:
        lse = torch.logsumexp(logits_scaled, dim=2) / math.log(2.0)  # [num_tokens, num_qo_heads], float32

        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
