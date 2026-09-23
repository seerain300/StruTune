import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_qn_kernel(
    Qn_ptr, Kc_ptr, Logits_ptr,
    H, L, D_ckv,
    Qn_stride0, Qn_stride1,
    Kc_stride0, Kc_stride1,
    Logits_stride0, Logits_stride1,
    inv_sm_scale,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, ceil_div(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv

        # Load q_nope[h, ks] as a vector of length BLOCK_K
        q_ptrs = Qn_ptr + h * Qn_stride0 + ks * Qn_stride1
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load Kc[ls, ks] tile
        kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
        mask_kc = mask_l[:, None] & mask_k[None, :]
        kc_tile = tl.load(kc_ptrs, mask=mask_kc, other=0.0)  # [BLOCK_L, BLOCK_K]

        # Accumulate: acc += sum_k q_vec[k] * kc_tile[:, k]
        # We can use tl.dot for reduction
        acc += tl.sum(kc_tile * q_vec[None, :], axis=1)

    # Apply scaling
    acc *= inv_sm_scale

    # Store
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def compute_logits_qp_kernel(
    Qp_ptr, Kp_ptr, Logits_ptr,
    H, L, D_kpe,
    Qp_stride0, Qp_stride1,
    Kp_stride0, Kp_stride1,
    Logits_stride0, Logits_stride1,
    inv_sm_scale,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, ceil_div(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe

        # Load q_pe[h, ks]
        q_ptrs = Qp_ptr + h * Qp_stride0 + ks * Qp_stride1
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0)

        # Load Kp[ls, ks]
        kp_ptrs = Kp_ptr + ls[:, None] * Kp_stride0 + ks[None, :] * Kp_stride1
        mask_kp = mask_l[:, None] & mask_k[None, :]
        kp_tile = tl.load(kp_ptrs, mask=mask_kp, other=0.0)

        acc += tl.sum(kp_tile * q_vec[None, :], axis=1)

    acc *= inv_sm_scale

    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, L_ptr, L_dim, H,
    Logits_stride0, Logits_stride1,
    inv_ln2,
    BLOCK_L: tl.constexpr,
):
    # One program per (h)
    h = tl.program_id(0)

    # Compute max
    max_val = -float('inf')
    for l0 in range(0, L_dim, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L_dim
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum exp
    sum_exp = 0.0
    for l0 in range(0, L_dim, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L_dim
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2  # logsumexp / ln(2)
    tl.store(L_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Kc_ptr, Out_ptr,
    H, L_dim, D_ckv,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute softmax over L_dim
    max_val = -float('inf')
    for l0 in range(0, L_dim, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L_dim
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    for l0 in range(0, L_dim, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L_dim
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp = tl.sum(e, axis=0)
        softmax = e / sum_exp  # [BLOCK_L]

        # Accumulate out[h, k] = sum_l softmax[l] * Kc[l, k]
        for k0 in range(0, D_ckv, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            mask_k = ks < D_ckv

            kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
            mask_kc = mask_l[:, None] & mask_k[None, :]
            kc_tile = tl.load(kc_ptrs, mask=mask_kc, other=0.0)  # [BLOCK_L, BLOCK_K]

            # out[h, ks] += sum_l softmax[l] * kc_tile[l, :]
            prod = tl.sum(softmax[:, None] * kc_tile, axis=0)  # [BLOCK_K]
            out_ptrs = Out_ptr + h * Out_stride0 + ks * Out_stride1
            tl.store(out_ptrs, prod, mask=mask_k)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assert fixed shapes
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        # In original run, assert page_size == 1 (but we won't rely on it; use kv_indices directly)

        device = q_nope.device
        assert device.type == 'cuda', "Triton implementation requires CUDA tensors."

        # Output and lse as float32 buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        batch_size = kv_indptr.shape[0] - 1
        sm_scale_inv = 1.0 / sm_scale

        # Constants for kernels
        BLOCK_L = 64
        BLOCK_K = 32
        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            num_q = q_end - q_start

            # For each query in this batch element
            for i in range(num_q):
                # Build q_nope[i] and q_pe[i] as device-compatible 1D contiguous tensors
                qn_row = q_nope[q_start + i].contiguous()  # [16, 512] -> flatten to 1D: H*D_ckv = 8192
                qp_row = q_pe[q_start + i].contiguous()   # [16, 64]  -> flatten to 1D: H*D_kpe = 1024

                # Gather KV indices for this batch
                tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32)  # [M], int32 on host; we will pass as Python list of integers to Triton (not torch.to)
                M = tok_idx.numel()

                # Allocate per-head Logits
                Logits = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)

                # Launch compute_logits_qn and compute_logits_qp kernels
                grid = (num_qo_heads, triton.cdiv(M, BLOCK_L))
                # Pass qn_row and qp_row as 1D device tensors:
                # We create them on device without torch.to on device (torch.tensor on host is fine, but here q_nope/q_pe are already device tensors, and .contiguous() returns device tensors). Ensure they are 1D and contiguous:
                qn_flat = qn_row.view(-1).contiguous()  # [8192], device
                qp_flat = qp_row.view(-1).contiguous()  # [1024], device

                compute_logits_qn_kernel[grid](
                    qn_flat, ckv_cache, Logits,
                    num_qo_heads, M, head_dim_ckv,
                    qn_flat.stride(0), qn_flat.stride(1),
                    ckv_cache.stride(0), ckv_cache.stride(1),
                    Logits.stride(0), Logits.stride(1),
                    sm_scale_inv,
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K
                )

                compute_logits_qp_kernel[grid](
                    qp_flat, kpe_cache, Logits,
                    num_qo_heads, M, head_dim_kpe,
                    qp_flat.stride(0), qp_flat.stride(1),
                    kpe_cache.stride(0), kpe_cache.stride(1),
                    Logits.stride(0), Logits.stride(1),
                    sm_scale_inv,
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K
                )

                # Apply causal mask and compute lse per head
                lse_b = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                grid_lse = (num_qo_heads,)
                lse_mask_kernel[grid_lse](
                    Logits, lse_b, M, num_qo_heads,
                    Logits.stride(0), Logits.stride(1),
                    inv_ln2,
                    BLOCK_L=BLOCK_L
                )
                # lse_b is per-head, update lse tensor at (q_start+i, :)
                lse[q_start + i, :] = lse_b

                # Compute output for this (b, i) per head
                Out = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                grid_out = (num_qo_heads,)
                softmax_matmul_kernel[grid_out](
                    Logits, ckv_cache, Out,
                    num_qo_heads, M, head_dim_ckv,
                    ckv_cache.stride(0), ckv_cache.stride(1),
                    Out.stride(0), Out.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K
                )
                # Store to output[q_start+i, :, :]
                output[q_start + i, :, :] = Out

        return output, lse


def run(*args):
    return ModelNew()(*args)
