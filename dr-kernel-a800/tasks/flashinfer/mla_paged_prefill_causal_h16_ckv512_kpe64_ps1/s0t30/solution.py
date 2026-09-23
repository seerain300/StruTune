import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_ptr,                  # *float32, shape [Dn] (row of q_nope for head h)
    qp_ptr,                  # *float32, shape [Dp] (row of q_pe for head h)
    Kc_ptr,                  # *float32, shape [KV, Dn]
    Kp_ptr,                  # *float32, shape [KV, Dp]
    logits_ptr,              # *float32, shape [KV] (output logits for head h)
    sm_scale: tl.float32,    # scaling factor
    KV: tl.int32,            # number of KV tokens
    Dn: tl.constexpr,        # head_dim_ckv (e.g., 512)
    Dp: tl.constexpr,        # head_dim_kpe (e.g., 64)
    stride_Kc_K: tl.int32,   # stride for KV dim in Kc (usually 1)
    stride_Kc_D: tl.int32,   # stride for Dn dim in Kc (usually KV)
    stride_Kp_K: tl.int32,   # stride for KV dim in Kp (usually 1)
    stride_Kp_D: tl.int32,   # stride for Dp dim in Kp (usually KV)
):
    # Accumulate two GEMVs: qn @ Kc.T and qp @ Kp.T, then sum and scale
    acc = 0.0
    # qn_ptr is 1D [Dn], Kc_ptr is [KV, Dn]
    for k in range(KV):
        # dot1 = sum_j qn[j] * Kc[k, j]
        dot1 = 0.0
        for j in range(0, Dn, 32):
            cols = j + tl.arange(0, 32)
            mask_d = cols < Dn
            qn_vec = tl.load(qn_ptr + cols, mask=mask_d, other=0.0)  # [32] float32
            kc_vec = tl.load(Kc_ptr + k * stride_Kc_K + cols * stride_Kc_D, mask=mask_d, other=0.0)  # [32]
            dot1 += tl.sum(qn_vec * kc_vec, axis=0)
        # dot2 = sum_d qp[d] * Kp[k, d]
        dot2 = 0.0
        for d in range(0, Dp, 32):
            cols = d + tl.arange(0, 32)
            mask_d = cols < Dp
            qp_vec = tl.load(qp_ptr + cols, mask=mask_d, other=0.0)  # [32] float32
            kp_vec = tl.load(Kp_ptr + k * stride_Kp_K + cols * stride_Kp_D, mask=mask_d, other=0.0)  # [32]
            dot2 += tl.sum(qp_vec * kp_vec, axis=0)
        acc += dot1 + dot2
    # scale
    acc = acc * sm_scale
    # store logits[k] (we loop k=0..KV-1 and store per k; here we compute all KV, but we need per-k storage).
    # To store per k, we recompute acc per k using tl.arange to build vector and store.
    for k in range(KV):
        # Recompute acc for this k using vectorized loads
        dot1 = 0.0
        for j in range(0, Dn, 32):
            cols = j + tl.arange(0, 32)
            mask_d = cols < Dn
            qn_vec = tl.load(qn_ptr + cols, mask=mask_d, other=0.0)
            kc_vec = tl.load(Kc_ptr + k * stride_Kc_K + cols * stride_Kc_D, mask=mask_d, other=0.0)
            dot1 += tl.sum(qn_vec * kc_vec, axis=0)
        dot2 = 0.0
        for d in range(0, Dp, 32):
            cols = d + tl.arange(0, 32)
            mask_d = cols < Dp
            qp_vec = tl.load(qp_ptr + cols, mask=mask_d, other=0.0)
            kp_vec = tl.load(Kp_ptr + k * stride_Kp_K + cols * stride_Kp_D, mask=mask_d, other=0.0)
            dot2 += tl.sum(qp_vec * kp_vec, axis=0)
        val = (dot1 + dot2) * sm_scale
        tl.store(logits_ptr + k, val)


@triton.jit
def apply_mask_kernel(
    logits_ptr,              # *float32, shape [KV]
    mask_ptr,                # *int32, shape [KV]
    KV: tl.int32,
):
    for j in range(KV):
        if tl.load(mask_ptr + j) == 0:
            val = tl.load(logits_ptr + j)
            tl.store(logits_ptr + j, -float('inf'))
        # else keep


@triton.jit
def lse_row_kernel(
    logits_ptr,              # *float32, shape [KV]
    lse_ptr,                 # *float32, scalar output
    KV: tl.int32,
):
    max_val = -float('inf')
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        if val > max_val:
            max_val = val
    sum_exp = 0.0
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        sum_exp += tl.exp(val - max_val)
    lse_val = tl.log(sum_exp) + max_val
    tl.store(lse_ptr, lse_val)


@triton.jit
def softmax_row_kernel(
    logits_ptr,              # *float32, shape [KV]
    attn_ptr,                # *float32, shape [KV]
    KV: tl.int32,
):
    # Two-pass softmax without storing intermediate: compute sum_exp, then normalize and store
    max_val = -float('inf')
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        if val > max_val:
            max_val = val

    sum_exp = 0.0
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        sum_exp += tl.exp(val - max_val)

    for j in range(KV):
        val = tl.load(logits_ptr + j)
        attn_val = tl.exp(val - max_val) / sum_exp
        tl.store(attn_ptr + j, attn_val)


@triton.jit
def compute_out_row_kernel(
    attn_ptr,                # *float32, shape [KV]
    Kc_ptr,                  # *float32, shape [KV, Dn]
    out_ptr,                 # *bfloat16, shape [Dn]
    KV: tl.int32,
    Dn: tl.constexpr,        # head_dim_ckv (e.g., 512)
    stride_Kc_K: tl.int32,   # stride along KV in Kc
    stride_Kc_D: tl.int32,   # stride along Dn in Kc
    stride_out_D: tl.int32,  # stride along Dn in out (should be 1)
):
    # Compute out = attn @ Kc (GEMV)
    out_row = tl.zeros((Dn,), dtype=tl.float32)
    for j in range(Dn):
        acc = 0.0
        for k in range(KV):
            acc += tl.load(attn_ptr + k) * tl.load(Kc_ptr + k * stride_Kc_K + j * stride_Kc_D)
        out_row[j] = acc
    # Store as bfloat16
    for j in range(Dn):
        tl.store(out_ptr + j * stride_out_D, out_row[j].to(tl.bfloat16))


def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # Ensure inputs are on CUDA
    device = q_nope.device
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64

    # Prepare caches (squeeze "1" dimension)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

    output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        if q_start >= q_end:
            continue

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        tok_idx = kv_indices[kv_start:kv_end].to(torch.long)  # [KV]
        KV = kv_end - kv_start
        # Gather Kc and Kp for this batch
        Kc = Kc_all[tok_idx]  # [KV, 512]
        Kp = Kp_all[tok_idx]  # [KV, 64]

        # Loop over queries in this batch
        for i in range(q_start, q_end):
            # Loop over heads
            for h in range(num_qo_heads):
                # Select qn[h, :] and qp[h, :]
                qn = q_nope[i, h, :].to(torch.float32).contiguous()  # [512]
                qp = q_pe[i, h, :].to(torch.float32).contiguous()   # [64]

                # Allocate intermediates
                logits = torch.empty((KV,), dtype=torch.float32, device=device)  # [KV]
                attn = torch.empty((KV,), dtype=torch.float32, device=device)    # [KV]

                # 1) Compute logits for head h using Triton
                compute_logits_kernel[(1,)](
                    qn, qp, Kc, Kp, logits, sm_scale, KV, 512, 64, Kc.stride(0), Kc.stride(1), Kp.stride(0), Kp.stride(1)
                )

                # 2) Apply causal mask: keep j if j > (prefix_len + i), else set to -inf
                prefix_len = KV - (q_end - q_start)
                abs_pos = prefix_len + (i - q_start)  # current query absolute position
                mask = torch.ones((KV,), dtype=torch.int32, device=device)
                mask[:(abs_pos + 1)] = 0
                apply_mask_kernel[(1,)](logits, mask, KV)

                # 3) Compute logsumexp for this head
                lse_val = torch.empty((1,), dtype=torch.float32, device=device)
                lse_row_kernel[(1,)](logits, lse_val, KV)
                lse[i, h] = lse_val[0]

                # 4) Softmax over masked logits (in-kernel)
                softmax_row_kernel[(1,)](logits, attn, KV)

                # 5) Compute output row: out[h, :] = attn @ Kc (Triton GEMV)
                out_row = torch.empty((head_dim_ckv,), dtype=torch.bfloat16, device=device)
                compute_out_row_kernel[(1,)](attn, Kc, out_row, KV, 512, Kc.stride(0), Kc.stride(1), out_row.stride(0))
                output[i, h] = out_row

    return output, lse


# Helpers for evaluation harness
def get_inputs():
    # Ensure CUDA tensors for Triton
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
