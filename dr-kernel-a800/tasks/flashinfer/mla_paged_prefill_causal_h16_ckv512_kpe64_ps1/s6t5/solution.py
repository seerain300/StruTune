import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    Qn_ptr, Qp_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    H: tl.constexpr, L: tl.constexpr, D_ckv: tl.constexpr, D_kpe: tl.constexpr,
    Qn_stride0, Qn_stride1,  # strides for Qn: [H, D_ckv]
    Qp_stride0, Qp_stride1,  # strides for Qp: [H, D_kpe]
    Kc_stride0, Kc_stride1,  # strides for Kc: [L, D_ckv]
    Kp_stride0, Kp_stride1,  # strides for Kp: [L, D_kpe]
    Logits_stride0, Logits_stride1,
    sm_scale: tl.float32,
    BLOCK_K: tl.constexpr,
):
    # Grid: (H,)
    h = tl.program_id(0)
    # Accumulator for this head's logits
    acc = tl.zeros((L,), dtype=tl.float32)

    # Iterate over K dimension in chunks (combine D_ckv and D_kpe)
    for k0 in range(0, D_ckv + D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_ks = ks < (D_ckv + D_kpe)

        # Load q vectors for this head h
        Qn_ptrs = Qn_ptr + h * Qn_stride0 + ks * Qn_stride1
        Qp_ptrs = Qp_ptr + h * Qp_stride0 + ks * Qp_stride1
        qn_chunk = tl.load(Qn_ptrs, mask=mask_ks, other=0.0)
        qp_chunk = tl.load(Qp_ptrs, mask=mask_ks, other=0.0)

        # Accumulate contributions
        for kk in range(BLOCK_K):
            if mask_ks[kk]:
                k_idx = ks[kk]
                if k_idx < D_ckv:
                    Kc_vals = tl.load(Kc_ptr + tl.arange(0, L) * Kc_stride0 + k_idx * Kc_stride1)
                    acc += qn_chunk[kk] * Kc_vals
                else:
                    Kp_vals = tl.load(Kp_ptr + tl.arange(0, L) * Kp_stride0 + (k_idx - D_ckv) * Kp_stride1)
                    acc += qp_chunk[kk] * Kp_vals

    # Scale and store to Logits[h, :]
    acc = acc * sm_scale
    Logits_row_ptr = Logits_ptr + h * Logits_stride0 + tl.arange(0, L) * Logits_stride1
    tl.store(Logits_row_ptr, acc, mask=tl.arange(0, L) < L)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, mask_ptr, LSE_ptr,
    H: tl.constexpr, L: tl.constexpr,
    Logits_stride0, Logits_stride1,
    inv_ln2: tl.float32,
    BLOCK_L: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)
    # Compute row-wise max over masked logits
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)  # 0 means keep, 1 means ignore
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, vals, -float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp over masked logits
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, vals, -float('inf'))
        e = tl.exp(vals - max_val)
        e = tl.where(m == 0, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    ln_sum = tl.log2(sum_exp) * inv_ln2  # ln(sumexp) = log2(sumexp) * (1/ln(2))
    tl.store(LSE_ptr + h, ln_sum)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, mask_ptr, Kc_ptr, Out_ptr,
    H: tl.constexpr, L: tl.constexpr, D_ckv: tl.constexpr,
    Logits_stride0, Logits_stride1,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    inv_ln2: tl.float32,  # not used here; for symmetry
    BLOCK_L: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute row-wise max
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, vals, -float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, vals, -float('inf'))
        e = tl.exp(vals - max_val)
        e = tl.where(m == 0, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Output vector out[h, :] in float32
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for k0 in range(0, D_ckv, BLOCK_L):
        ks = k0 + tl.arange(0, BLOCK_L)
        mask_k = ks < D_ckv
        # out_vec += sum_l softmax[l] * Kc[l, ks]
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            vals = tl.where(m == 0, vals, -float('inf'))
            e = tl.exp(vals - max_val) * inv_sum  # non-causal positions were masked to 0 above
            for kk in range(BLOCK_L):
                if mask_k[kk] and (k0 + kk) < D_ckv:
                    Kc_col = tl.load(Kc_ptr + ls * Kc_stride0 + (k0 + kk) * Kc_stride1, mask=mask_l, other=0.0)
                    out_vec[k0 + kk] += tl.sum(e * Kc_col, axis=0)

    Out_row_ptr = Out_ptr + h * Out_stride0 + tl.arange(0, D_ckv) * Out_stride1
    tl.store(Out_row_ptr, out_vec, mask=tl.arange(0, D_ckv) < D_ckv)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Prepare output tensors
        # We don't have explicit H from q_nope; the original code asserts num_qo_heads=16.
        H = 16
        D_ckv = 512
        D_kpe = 64

        total_q = int(qo_indptr[-1].item())
        batch_size = qo_indptr.shape[0] - 1

        output = torch.empty((total_q, H, D_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        # Precompute Kc_all and Kp_all: ckv_cache is [num_pages, 1, D_ckv], squeeze dim=1
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D_ckv]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, D_kpe]

        inv_ln2 = 1.0 / math.log(2.0)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            kv_len = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            tok_idx = kv_indices[b * kv_indptr[b].item() : (b + 1) * kv_indptr[b + 1].item()].to(torch.int32)
            Kc_batch = Kc_all[tok_idx]  # [kv_len, D_ckv]
            Kp_batch = Kp_all


def run(*args):
    return ModelNew()(*args)
