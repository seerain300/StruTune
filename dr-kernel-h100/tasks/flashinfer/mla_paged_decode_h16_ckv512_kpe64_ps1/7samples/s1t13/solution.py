import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [1, D] — we pass q_nope[b, j, :]
    qp_ptr,           # *float32, [1, Dp] — we pass q_pe[b, j, :]
    Kc_ptr,           # *float32, [L, D], row-major
    Kp_ptr,           # *float32, [L, Dp], row-major
    v_ptr,            # *float32, [L]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    Dp: tl.int32,     # head_dim_kpe
    BLOCK_K: tl.constexpr,
):
    # One program per token i
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc = tl.load(Kc_ptr + i * D + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc, axis=0)
    # Reduce over Kp dimension (Dp)
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        kp = tl.load(Kp_ptr + i * Dp + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp, axis=0)
    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1] scalar output
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # One program performs stable reduction to compute base-2 logsumexp
    max_v = -float("inf")
    # Pass 1: find max
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        chunk_max = tl.max(v, axis=0)
        max_v = tl.maximum(max_v, chunk_max)
    # Pass 2: sum exp normalized by ln(2)
    sum_exp = 0.0
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        expv = tl.exp(v - max_v) * inv_ln2
        sum_exp += tl.sum(expv, axis=0)
    lse_val = max_v + tl.log(sum_exp)
    tl.store(lse_ptr, lse_val)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    attn_ptr,         # *float32, [L]
    lse_val: tl.float32,
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # One program per token i
    i = tl.program_id(0)
    vi = tl.load(v_ptr + i)
    attn_i = tl.exp(vi * inv_ln2 - lse_val)
    tl.store(attn_ptr + i, attn_i)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D]
    out_ptr,          # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # One program per output dimension h
    h = tl.program_id(0)
    acc = 0.0
    for i in range(0, L, BLOCK_K):
        offs = i + tl.arange(0, BLOCK_K)
        mask = offs < L
        attn = tl.load(attn_ptr + offs, mask=mask, other=0.0)  # [BLOCK_K]
        kc = tl.load(Kc_ptr + offs * D + h, mask=mask, other=0.0)  # [BLOCK_K]
        acc += tl.sum(attn * kc, axis=0)
    tl.store(out_ptr + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]  # 512
    head_dim_kpe = q_pe.shape[2]    # 64

    # Prepare Kc_all and Kp_all from cache; ensure float32 and contiguous
    Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

    # Output and lse
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    inv_ln2 = 1.0 / math.log(2.0)

    # Ensure kv_indptr and kv_indices are on device and int32
    if kv_indptr.device != device:
        kv_indptr = kv_indptr.to(device)
    if kv_indices.device != device:
        kv_indices = kv_indices.to(device)

    for b in range(batch_size):
        # Determine token range
        # kv_indptr[b] and kv_indptr[b+1] are int32, need int64 for indexing
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = max(0, page_end - page_beg)
        if L_tokens == 0:
            lse[b, 0:] = -float("inf")  # fill with -inf to avoid errors
            continue

        # Gather tok_idx
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)

        # Slice Kc and Kp for tokens
        Kc = Kc_all[tok_idx]  # [L_tokens, 512]
        Kp = Kp_all[tok_idx]  # [L_tokens, 64]

        # For each head j, compute v, lse, attn, and output
        for j in range(num_qo_heads):
            # Prepare per-head qn and qp as 1D vectors
            qn = q_nope[b, j, :].to(torch.float32).contiguous()  # [512]
            qp = q_pe[b, j, :].to(torch.float32).contiguous()   # [64]

            # Compute v[j, :] = sum over tokens of (qn · Kc[i, :]) + (qp · Kp[i, :])
            v = torch.empty(L_tokens, dtype=torch.float32, device=device)
            grid_v = (L_tokens,)
            matvec_add_kernel[grid_v](
                qn, qp, Kc, Kp, v, L_tokens, head_dim_ckv, head_dim_kpe,
                BLOCK_K=128, num_warps=4
            )

            # Compute lse_j (base-2 logsumexp)
            lse_j = torch.empty(1, dtype=torch.float32, device=device)
            grid_lse = (1,)
            lse_kernel[grid_lse](
                v, lse_j, L_tokens, inv_ln2,
                BLOCK=128, num_warps=4
            )
            lse[b, j] = lse_j[0]

            # Compute attn[j, :]
            attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
            grid_softmax = (L_tokens,)
            softmax_base2_kernel[grid_softmax](
                v, attn, lse[b, j], L_tokens, inv_ln2,
                BLOCK=128, num_warps=4
            )

            # Final matvec: out[b, j, :] = attn @ Kc[:, :]
            out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            grid_y = (head_dim_ckv,)
            matvec_write_y_kernel[grid_y](
                attn, Kc, out_row, L_tokens, head_dim_ckv,
                BLOCK=128, num_warps=4
            )
            output[b, j, :] = out_row

    # Cast output to bfloat16 to match original
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def get_inputs():
    # Helper: the harness will supply inputs to ModelNew.forward; do not call run here.
    pass


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA
        for t in (q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
            if isinstance(t, torch.Tensor) and t.device.type != 'cuda':
                t = t.to('cuda')
        # Triton-orchestrated computation
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
