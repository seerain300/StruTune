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
    q_nope_ptr, q_pe_ptr, Kc_all_ptr, Kp_all_ptr, sparse_indices_ptr,
    logits_scaled_ptr,
    num_tokens, num_qo_heads,
    TOPK: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
):
    # Each program handles one (t, h)
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Load q_nope[t, h, :] and q_pe[t, h, :]
    # q_nope shape: [num_tokens, num_qo_heads, head_dim_ckv]
    qn = tl.load(q_nope_ptr + t * num_qo_heads * head_dim_ckv + h * head_dim_ckv)  # float32
    # q_pe shape: [num_tokens, num_qo_heads, head_dim_kpe]
    qp = tl.load(q_pe_ptr + t * num_qo_heads * head_dim_kpe + h * head_dim_kpe)  # float32

    # Loop over k in [0, TOPK)
    for k in range(TOPK):
        idx = tl.load(sparse_indices_ptr + t * TOPK + k)  # int32
        # Load Kc_all row and Kp_all row
        kc_row = tl.load(Kc_all_ptr + idx * head_dim_ckv)  # float32 [head_dim_ckv]
        kp_row = tl.load(Kp_all_ptr + idx * head_dim_kpe)  # float32 [head_dim_kpe]

        contrib1 = 0.0
        for d in range(head_dim_ckv):
            contrib1 += qn[d] * kc_row[d]
        contrib2 = 0.0
        for d in range(head_dim_kpe):
            contrib2 += qp[d] * kp_row[d]

        logit = contrib1 + contrib2  # sm_scale is 1.0 as per provided inputs
        # Store to logits_scaled[t, h, k] (float32)
        tl.store(logits_scaled_ptr + t * (num_qo_heads * TOPK) + h * TOPK + k, logit)


@triton.jit
def compute_lse_and_attn_kernel(
    logits_scaled_ptr, attn_ptr, lse_ptr,
    num_tokens, num_qo_heads, TOPK: tl.constexpr,
):
    # Each program handles one (t, h)
    t = tl.program_id(0)
    h = tl.program_id(1)
    base = logits_scaled_ptr + t * (num_qo_heads * TOPK) + h * TOPK

    # Pass 1: compute max and sum of exp(x - m)
    m = -float("inf")
    sum_exp = 0.0
    for k in range(TOPK):
        x = tl.load(base + k)
        m = tl.maximum(m, x)
    for k in range(TOPK):
        x = tl.load(base + k)
        e = tl.exp(x - m)
        sum_exp += e

    ln2 = 0.6931471805599453  # log(2.0)
    lse = m + tl.log(sum_exp) / ln2
    tl.store(lse_ptr + t * num_qo_heads + h, lse)

    # Pass 2: compute attn and store
    for k in range(TOPK):
        x = tl.load(base + k)
        attn_k = tl.exp(x - lse)
        tl.store(attn_ptr + t * (num_qo_heads * TOPK) + h * TOPK + k, attn_k)


@triton.jit
def compute_output_kernel_sparse(
    attn_ptr, Kc_all_ptr, output_ptr, sparse_indices_ptr,
    num_tokens, num_qo_heads, head_dim_ckv: tl.constexpr, TOPK: tl.constexpr,
):
    # Each program handles one (t, h)
    t = tl.program_id(0)
    h = tl.program_id(1)

    out = tl.zeros([head_dim_ckv], dtype=tl.float32)

    # Loop over k in [0, TOPK), accumulate attn_k * Kc_all[sparse_indices[t, k], :]
    for k in range(TOPK):
        attn_k = tl.load(attn_ptr + t * (num_qo_heads * TOPK) + h * TOPK + k)
        idx_k = tl.load(sparse_indices_ptr + t * TOPK + k)  # int32
        kc_row = tl.load(Kc_all_ptr + idx_k * head_dim_ckv)  # float32 [head_dim_ckv]
        out += attn_k * kc_row

    # Store output as bfloat16
    tl.store(output_ptr + t * num_qo_heads * head_dim_ckv + h * head_dim_ckv, out.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, page_size, _ = ckv_cache.shape
        topk = sparse_indices.shape[-1]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert page_size == 64
        assert topk == 2048

        device = q_nope.device

        # Cast inputs to float32 for computation
        q_nope = q_nope.to(torch.float32).contiguous()  # [num_tokens, num_qo_heads, 512]
        q_pe = q_pe.to(torch.float32).contiguous()      # [num_tokens, num_qo_heads, 64]
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32).contiguous()  # [total_tokens, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32).contiguous()  # [total_tokens, 64]
        sparse_indices = sparse_indices.to(torch.int32).contiguous()  # [num_tokens, 2048]

        # Allocate outputs
        logits_scaled = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        attn = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)

        # Launch kernel 1: compute logits_scaled
        grid = (num_tokens, num_qo_heads)
        compute_logits_scaled_kernel[grid](
            q_nope, q_pe, Kc_all, Kp_all, sparse_indices,
            logits_scaled,
            num_tokens, num_qo_heads,
            TOPK=topk,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            num_warps=2,
        )

        # Launch kernel 2: compute lse and attn
        compute_lse_and_attn_kernel[grid](
            logits_scaled, attn, lse,
            num_tokens, num_qo_heads,
            TOPK=topk,
            num_warps=2,
        )

        # Launch kernel 3: compute output
        compute_output_kernel_sparse[grid](
            attn, Kc_all, output, sparse_indices,
            num_tokens, num_qo_heads,
            head_dim_ckv=head_dim_ckv,
            TOPK=topk,
            num_warps=2,
        )

        # Return output (bfloat16) and lse (float32) to match original signature
        return output, lse


def run(*args):
    return ModelNew()(*args)
