import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    Qn_ptr, Qp_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    H: tl.constexpr, L: tl.constexpr, D_ckv: tl.constexpr, D_kpe: tl.constexpr,
    Qn_stride0, Qn_stride1,
    Qp_stride0, Qp_stride1,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    Logits_stride0, Logits_stride1,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles one (h, tile along L)
    h = tl.program_id(0)
    pid_n = tl.program_id(1)
    ls = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_l = ls < L

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K dimension in chunks; include both Kc and Kp
    for k0 in range(0, D_ckv + D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < (D_ckv + D_kpe)

        # Load q vectors for ks; Q has shape [H, D_kq] where D_kq is either D_ckv or D_kpe depending on side
        # We need to load q_nope and q_pe separately: q_nope[h, ks if ks<D_ckv else 0], q_pe[h, ks-D_ckv if ks>=D_ckv else 0]
        # Here, we decode q_nope and q_pe contributions:
        # For ks < D_ckv: q_nope[h, ks]
        # For ks >= D_ckv: q_pe[h, ks - D_ckv]
        qn_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
        qp_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)

        # First half for q_nope
        for kk in range(BLOCK_K):
            if mask_k[kk]:
                k_idx = ks[kk]
                if k_idx < D_ckv:
                    qn_vals[kk] = tl.load(Qn_ptr + h * Qn_stride0 + k_idx * Qn_stride1)
                # else do nothing for q_nope part

        # Second half for q_pe
        for kk in range(BLOCK_K):
            if mask_k[kk]:
                k_idx = ks[kk]
                if k_idx >= D_ckv:
                    kp_idx = k_idx - D_ckv
                    qp_vals[kk] = tl.load(Qp_ptr + h * Qp_stride0 + kp_idx * Qp_stride1)
                # else do nothing for q_pe part

        # Now accumulate acc += sum_j (qn_vals[j] * Kc[ls, j]) + (qp_vals[j] * Kp[ls, j])
        # We loop over ks (BLOCK_K) to multiply with Kc/Kp and reduce
        for jj in range(BLOCK_K):
            if mask_k[jj]:
                k_idx = ks[jj]
                # Contribution from q_nope
                if k_idx < D_ckv:
                    Kc_vals = tl.load(Kc_ptr + ls * Kc_stride0 + k_idx * Kc_stride1, mask=mask_l, other=0.0)
                    acc += qn_vals[jj] * Kc_vals
                # Contribution from q_pe
                if k_idx >= D_ckv:
                    kp_idx = k_idx - D_ckv
                    Kp_vals = tl.load(Kp_ptr + ls * Kp_stride0 + kp_idx * Kp_stride1, mask=mask_l, other=0.0)
                    acc += qp_vals[jj] * Kp_vals

    # Store acc to Logits[h, ls]
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def apply_mask_lse_kernel(
    Logits_ptr, mask_ptr, LSE_ptr,
    H: tl.constexpr, L: tl.constexpr,
    Logits_stride0, Logits_stride1,
    mask_stride: tl.constexpr,  # typically 1 since mask is 1D contiguous
):
    h = tl.program_id(0)
    # Compute row max
    max_val = -float('inf')
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        m = tl.load(mask_ptr + ls * mask_stride, mask=mask_l, other=1)
        vals = tl.where(m == 0, -float('inf'), vals)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp
    sum_exp = 0.0
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        m = tl.load(mask_ptr + ls * mask_stride, mask=mask_l, other=1)
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val)
        e = tl.where(m == 0, 0.0, e)
        sum_exp += tl.sum(e, axis=0)

    ln2 = 0.6931471805599453  # 1/ln(2)
    lse_scaled = tl.log2(sum_exp) * ln2  # natural log via log2(x) * ln(2)
    tl.store(LSE_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Kc_ptr, Out_ptr,
    H: tl.constexpr, L: tl.constexpr, D_ckv: tl.constexpr,
    Logits_stride0, Logits_stride1,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)

    # Compute row max
    max_val = -float('inf')
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.where(m == 0, -float('inf'), vals)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp
    sum_exp = 0.0
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val)
        e = tl.where(m == 0, 0.0, e)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Accumulate out = softmax * Kc
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for k0 in range(0, D_ckv, 128):
        ks = k0 + tl.arange(0, 128)
        mask_k = ks < D_ckv

        # For each l, compute softmax[l] and accumulate out_vec += softmax[l] * Kc[l, ks]
        for l0 in range(0, L, 128):
            ls = l0 + tl.arange(0, 128)
            mask_l = ls < L
            m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            vals = tl.where(m == 0, -float('inf'), vals)
            e = tl.exp(vals - max_val) * inv_sum
            e = tl.where(m == 0, 0.0, e)

            Kc_vals = tl.load(Kc_ptr + ls * Kc_stride0 + ks * Kc_stride1, mask=(mask_l & mask_k), other=0.0)
            out_vec += tl.sum(e[:, None] * Kc_vals[None, :], axis=0)

    out_ptrs = Out_ptr + h * Out_stride0 + tl.arange(0, D_ckv) * Out_stride1
    tl.store(out_ptrs, out_vec, mask=(tl.arange(0, D_ckv) < D_ckv))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # device setup
        device = q_nope.device
        assert ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        # Prepare sizes
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Decode batch count: len_indptr - 1
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        assert kv_indptr.shape[0] == batch_size

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute mask for each batch: abs_pos threshold depends on b and query index
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # KV block
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # [L]
            # Gather Kc and Kp for this batch element
            Kc_batch = ckv_cache[tok_idx].to(torch.float32)  # [L, 512]
            Kp_batch = kpe_cache[tok_idx].to(torch.float32)  # [L, 64]

            # For each query i in [q_start, q_end)
            for i in range(q_start, q_end):
                # Prepare Logits and mask
                Logits = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                mask = torch.ones((kv_len,), dtype=torch.int32, device=device)  # we'll fill per-row below

                # Launch compute_logits_kernel: grid over heads and tiles of L
                grid = (num_qo_heads, triton.cdiv(kv_len, 128))
                compute_logits_kernel[grid](
                    q_nope, q_pe, Kc_batch, Kp_batch, Logits,
                    H=num_qo_heads, L=kv_len, D_ckv=512, D_kpe=64,
                    Qn_stride0=0, Qn_stride1=1,  # q_nope is [H, D], strides in elements
                    Qp_stride0=0, Qp_stride1=1,  # q_pe is [H, D]
                    Kc_stride0=Kc_batch.stride(0), Kc_stride1=Kc_batch.stride(1),
                    Kp_stride0=Kp_batch.stride(0), Kp_stride1=Kp_batch.stride(1),
                    Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                    BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2,
                )

                # Build causal mask for this row: positions l > (kv_len - (q_end - q_start) + i) -> 0
                # Note: for each batch b, mask depends on b, i, q_end. We use absolute row index in L to decide causal.
                # mask is 1 for causal, 0 for non-causal (set to -inf later).
                # Compute threshold as absolute position query_abs_pos
                # From original: causal if l <= query_abs_pos; here query_abs_pos = (q_end - q_start) + i - 1 (we subtract 1 to account for inclusive end).
                # To keep mask 1D: we reuse the same logic by passing threshold via host-side mask. However Triton kernel reads mask we create.
                # Let's compute mask on host (Triton doesn't have side-effect to a global tensor here).
                # For simplicity, we compute mask and pass as int32 tensor (1 causal, 0 non-causal). We need to know threshold:
                # threshold = (q_end - q_start) + i - 1
                threshold = (q_end - q_start) + i - 1
                if threshold < 0:
                    threshold = 0
                mask = torch.ones((kv_len,), dtype=torch.int32, device=device)
                mask[(threshold + 1):] = 0  # non-causal positions (indices > threshold)

                # Launch apply_mask_lse_kernel to compute per-head lse_scaled
                lse[i] = apply_mask_lse_kernel[(num_qo_heads,)](
                    Logits, mask, lse[i],
                    H=num_qo_heads, L=kv_len,
                    Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                    mask_stride=1,
                    num_warps=1, num_stages=1,
                )

                # Launch softmax_matmul_kernel to compute output for this i
                Out = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                softmax_matmul_kernel[(num_qo_heads,)](
                    Logits, Kc_batch, Out,
                    H=num_qo_heads, L=kv_len, D_ckv=512,
                    Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                    Kc_stride0=Kc_batch.stride(0), Kc_stride1=Kc_batch.stride(1),
                    Out_stride0=Out.stride(0), Out_stride1=Out.stride(1),
                    BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2,
                )
                output[i] = Out

        # Return bfloat16 output and float32 lse
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
