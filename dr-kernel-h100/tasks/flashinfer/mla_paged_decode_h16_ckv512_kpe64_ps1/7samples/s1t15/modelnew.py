import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, shape [D] (head vector)
    qp_ptr,           # *float32, shape [Dp] (head vector)
    Kc_ptr,           # *float32, [L, D], row-major
    Kp_ptr,           # *float32, [L, Dp], row-major
    v_ptr,            # *float32, [L], output per-token logits
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    Dp: tl.int32,     # head_dim_kpe
    BLOCK_K: tl.constexpr,
):
    # One program per token index i
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr_row = Kc_ptr + i * D + k_off
        kc_vals = tl.load(kc_ptr_row, mask=mask_k, other=0.0)
        sum1 += tl.sum(qn_slice * kc_vals, axis=0)
    # Reduce over Kp dimension (Dp)
    for k in range(0, Dp, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < Dp
        qp_slice = tl.load(qp_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kp_ptr_row = Kp_ptr + i * Dp + k_off
        kp_vals = tl.load(kp_ptr_row, mask=mask_k, other=0.0)
        sum2 += tl.sum(qp_slice * kp_vals, axis=0)
    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    L: tl.int32,
    BLOCK_L: tl.constexpr,
):
    # One program computes max and sum_exp across v
    m = -float("inf")
    sum_exp = 0.0
    ln2 = 1.4426950408889634  # 1 / log(2)
    # First pass: find max
    for start in range(0, L, BLOCK_L):
        offs = start + tl.arange(0, BLOCK_L)
        mask = offs < L
        vals = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        m = tl.maximum(m, tl.max(vals, axis=0))
    # Second pass: sum exp(v - m)
    for start in range(0, L, BLOCK_L):
        offs = start + tl.arange(0, BLOCK_L)
        mask = offs < L
        vals = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(vals - m), axis=0)
    lse = (m + tl.log(sum_exp)) / ln2
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    BLOCK_L: tl.constexpr,
):
    ln2 = 1.4426950408889634
    lse = tl.load(lse_ptr)
    for start in range(0, L, BLOCK_L):
        offs = start + tl.arange(0, BLOCK_L)
        mask = offs < L
        vals = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        attn = tl.exp(vals - lse)  # base-2 softmax using lse
        tl.store(attn_ptr + offs, attn, mask=mask)


def get_inputs():
    # Non-recursive helper for local testing; harness provides its own inputs.
    axes = {
        "batch_size": 1,
        "num_pages": 989669,
        "len_indptr": 2,
        "num_kv_indices": 8,
    }
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    # Simulate kv_indptr and kv_indices for one batch
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)  # [2]
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        device = q_nope.device
        batch_size = q_nope.shape[0]
        num_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Preprocess caches to float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers
        output = torch.zeros((batch_size, num_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, num_heads), -float("inf"), dtype=torch.float32, device=device)

        # For each batch
        for b in range(batch_size):
            L = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if L <= 0:
                # No tokens for this batch element
                output[b].zero_()
                for j in range(num_heads):
                    lse[b, j] = -float("inf")
                continue

            # Gather token indices
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]]

            # Collect Kc and Kp for these tokens
            Kc_sel = Kc_all[tok_idx]  # [L, 512]
            Kp_sel = Kp_all[tok_idx]  # [L, 64]

            # Prepare v buffer [L] (float32)
            v = torch.empty(L, dtype=torch.float32, device=device)

            # Compute v[j, :] for each head j using Triton kernel
            # Convert head vectors to float32
            for j in range(num_heads):
                qn = q_nope[b, j].to(torch.float32)          # [512]
                qp = q_pe[b, j].to(torch.float32)           # [64]
                grid = (L,)
                matvec_add_kernel[grid](qn, qp, Kc_sel, Kp_sel, v, L, head_dim_ckv, head_dim_kpe, BLOCK_K=128, num_warps=4)
                # Now lse and softmax using Triton
                lse_buf = torch.empty(1, dtype=torch.float32, device=device)
                BLOCK_L = 1024
                lse_kernel[(1,)](v, lse_buf, L, BLOCK_L, num_warps=4)
                lse_j = lse_buf[0]
                # Compute attn per token
                attn = torch.empty(L, dtype=torch.float32, device=device)
                softmax_base2_kernel[(1,)](v, lse_buf, attn, L, BLOCK_L, num_warps=4)
                # Final matvec: out[b, j, :] = attn @ Kc_sel
                # Kc_sel is [L, 512], attn is [L], result is [512]
                out_j = torch.addmm(torch.zeros(1, dtype=torch.float32, device=device), attn.view(L, 1), Kc_sel, beta=0.0, alpha=1.0)
                output[b, j, :] = out_j.to(torch.bfloat16)

        return output, lse