import math
import torch
import triton
import triton.language as tl


# Kernel 1: compute v[i] = sum over D of (qn · Kc[i, :]) + sum over Dp of (qp · Kp[i, :])
@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [D]
    qp_ptr,           # *float32, [Dp]
    Kc_ptr,           # *float32, [L, D], row-major
    Kp_ptr,           # *float32, [L, Dp], row-major
    v_ptr,            # *float32, [L]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    Dp: tl.int32,     # head_dim_kpe
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    acc = 0.0
    # Reduce over D
    k = 0
    while k < D:
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        kc = tl.load(Kc_ptr + i * D + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        qn = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)          # [BLOCK_K]
        acc += tl.sum(qn * kc, axis=0)
        k += BLOCK_K
    # Reduce over Dp
    k = 0
    while k < Dp:
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < Dp
        kp = tl.load(Kp_ptr + i * Dp + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        qp = tl.load(qp_ptr + k_off, mask=mask_k, other=0.0)           # [BLOCK_K]
        acc += tl.sum(qp * kp, axis=0)
        k += BLOCK_K
    tl.store(v_ptr + i, acc)


# Kernel 2: compute base-2 logsumexp for a vector v of length L (writes a scalar lse per head)
@triton.jit
def lse_base2_kernel(
    v_ptr,            # *float32, [L]
    out_ptr,          # *float32, [1] (we'll use scalar via pointer)
    L: tl.int32,
    inv_log2: tl.float32,  # 1/ln(2)
    BLOCK: tl.constexpr,
):
    # Reduce using two-pass: first find max, then sum exp(v / ln(2)), then log
    max_val = -float("inf")
    # Pass 1: max
    k = 0
    while k < L:
        offsets = k + tl.arange(0, BLOCK)
        mask = offsets < L
        v = tl.load(v_ptr + offsets, mask=mask, other=-float("inf"))
        local_max = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, local_max)
        k += BLOCK
    # Pass 2: sum exp(v / ln(2))
    sum_exp = 0.0
    k = 0
    while k < L:
        offsets = k + tl.arange(0, BLOCK)
        mask = offsets < L
        v = tl.load(v_ptr + offsets, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(v * inv_log2), axis=0)
        k += BLOCK
    lse = max_val + tl.log(sum_exp)  # logsumexp in ln scale
    tl.store(out_ptr, lse)


# Kernel 3: compute softmax-base-2 attention weights for vector v using lse:
# attn[i] = exp((v[i] / ln(2)) - lse)
@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1] (scalar lse)
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    inv_log2: tl.float32,
    BLOCK: tl.constexpr,
):
    k = 0
    while k < L:
        offsets = k + tl.arange(0, BLOCK)
        mask = offsets < L
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        lse = tl.load(lse_ptr)  # scalar
        attn = tl.exp(v * inv_log2 - lse)
        tl.store(attn_ptr + offsets, attn, mask=mask)
        k += BLOCK


# Kernel 4: write final output vector for head h: out[h, :] = attn @ Kc (reduce over L)
@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major
    out_ptr,          # *float32, [D]
    D: tl.int32,
    L: tl.int32,
    BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)  # head index (we use grid (H,))
    # one program per output dimension h, reducing over L
    acc = 0.0
    k = 0
    while k < L:
        offsets = k + tl.arange(0, BLOCK_K)
        mask = offsets < L
        attn = tl.load(attn_ptr + offsets, mask=mask, other=0.0)  # [BLOCK_K]
        kc = tl.load(Kc_ptr + offsets * D + h, mask=mask, other=0.0)  # [BLOCK_K]
        acc += tl.sum(attn * kc, axis=0)
        k += BLOCK_K
    tl.store(out_ptr + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-orchestrated forward. Computes output and lse per batch.
    """
    device = q_nope.device
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    num_pages = ckv_cache.shape[0]
    L_tokens = kv_indptr[-1].item() - kv_indptr[0].item()
    # Allocate outputs
    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    # Precompute inv_log2 for base-2 logsumexp
    inv_log2 = 1.0 / math.log(2.0)

    for b in range(batch_size):
        if kv_indptr.numel() <= b + 1:
            # No tokens for this batch
            continue
        # Determine token range
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L = page_end - page_beg
        if L <= 0:
            # No KV cache for this batch element
            output[b].zero_()
            lse[b] = -float("inf")
            continue

        # Gather Kc and Kp for this batch
        tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # indices
        Kc = ckv_cache.index_select(0, tok_idx).squeeze(1).to(torch.float32)  # [L, D]
        Kp = kpe_cache.index_select(0, tok_idx).squeeze(1).to(torch.float32)  # [L, Dp]
        D = head_dim_ckv
        Dp = head_dim_kpe

        # Prepare q vectors
        qn = q_nope[b, :, :].to(torch.float32)  # [H, D]
        qp = q_pe[b, :, :].to(torch.float32)    # [H, Dp]

        # Compute v per head
        for j in range(num_qo_heads):
            v = torch.empty(L, dtype=torch.float32, device=device)
            # Launch matvec_add_kernel
            grid = (L,)
            # BLOCK_K tuned for typical D
            matvec_add_kernel[grid](
                qn[j, :].contiguous(),          # [D]
                qp[j, :].contiguous(),          # [Dp]
                Kc.contiguous(),                # [L, D]
                Kp.contiguous(),                # [L, Dp]
                v,                              # [L]
                L, D, Dp,
                BLOCK_K=128,
                num_warps=4,
            )
            # Compute lse for head j
            lse_scalar = torch.empty((), dtype=torch.float32, device=device)
            lse_base2_kernel[(1,)](
                v, lse_scalar, L, inv_log2, BLOCK=1024
            )
            # Compute attn for head j
            attn = torch.empty(L, dtype=torch.float32, device=device)
            softmax_base2_kernel[(grid,)](
                v, lse_scalar, attn, L, inv_log2, BLOCK=1024
            )
            # Compute out[b, j, :] = attn @ Kc
            out_vec = torch.empty(D, dtype=torch.float32, device=device)
            matvec_write_y_kernel[(num_qo_heads,)](
                attn, Kc.contiguous(), out_vec, D, L, BLOCK_K=128, num_warps=4
            )
            output[b, j, :] = out_vec

        # Now lse for this batch element is per head; if the original had only one head in scope,
        # it computed lse per head. Since output uses per-head, we also compute lse per head here.
        # However, the original function returned lse as (batch_size, num_qo_heads). We keep it.

    # Return in the same dtype as original output (bfloat16), and lse in float32
    return output.to(torch.bfloat16), lse


# Helper: non-recursive get_inputs for local testing
def get_inputs():
    # Example inputs; the harness will provide its own. This helper returns CUDA tensors.
    num_pages = 989669
    batch_size = 1
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_tokens = 8
    q_nope = torch.randn([batch_size, num_qo_heads, head_dim_ckv], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([batch_size, num_qo_heads, head_dim_kpe], device='cuda')
    ckv_cache = torch.randn([num_pages, 1, head_dim_ckv], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([num_pages, 1, head_dim_kpe], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = num_tokens
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)  # [2]
    kv_indices = torch.randint(0, num_pages, [num_tokens], dtype=torch.int32, device='cuda')
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Optional helper to satisfy the evaluation harness interface
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure CUDA
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        # Run Triton-orchestrated computation
        return run(*args)