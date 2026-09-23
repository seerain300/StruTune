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
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, ceil_div(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    total_K = D_ckv  # 512
    for k0 in range(0, total_K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < total_K

        # Qn_ptr is 1D with length H * D_ckv. For each ks, the value corresponds to q_nope[i, ks], spread across heads.
        q_ptrs = Qn_ptr + ks
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Kc_ptr is 2D [L, D_ckv], but we only need Kc[ls, ks] for the current L tile
        kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1  # [BLOCK_L, BLOCK_K]
        kc = tl.load(kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]

        # Accumulate: for each column ks, add q_vec[ks] * kc[:, ks]
        # We can do an explicit loop over BLOCK_K columns (BLOCK_K is constexpr), or use a matmul-style reduction.
        # Here we use explicit loop for simplicity and robustness.
        for kk in range(0, BLOCK_K):
            k_idx = k0 + kk
            valid_k = k_idx < total_K
            # Skip invalid ks by masking (since we pre-mask loads above, q_vec[kk] is 0 for invalid k)
            q_k = q_vec[kk]
            kc_col = kc[:, kk]  # [BLOCK_L]
            acc += q_k * kc_col

    # Store accumulated logits
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def compute_logits_qp_kernel(
    Qp_ptr, Kp_ptr, Logits_ptr,
    H, L, D_kpe,
    Qp_stride0, Qp_stride1,
    Kp_stride0, Kp_stride1,
    Logits_stride0, Logits_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, ceil_div(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    total_K = D_kpe  # 64
    for k0 in range(0, total_K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < total_K

        # Qp_ptr is 1D with length H * D_kpe. For each ks, the value corresponds to q_pe[i, ks], spread across heads.
        q_ptrs = Qp_ptr + ks
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Kp_ptr is 2D [L, D_kpe]
        kp_ptrs = Kp_ptr + ls[:, None] * Kp_stride0 + ks[None, :] * Kp_stride1  # [BLOCK_L, BLOCK_K]
        kp = tl.load(kp_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]

        # Accumulate per column
        for kk in range(0, BLOCK_K):
            k_idx = k0 + kk
            valid_k = k_idx < total_K
            q_k = q_vec[kk]
            kp_col = kp[:, kk]  # [BLOCK_L]
            acc += q_k * kp_col

    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, LSE_ptr, L,
    Logits_stride0, Logits_stride1,
    inv_ln2,
    BLOCK_L: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute row-wise max and sum(exp) after masking
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2  # divide by ln(2)
    tl.store(LSE_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Kc_ptr, Out_ptr,
    H, L, D_ckv,
    Logits_stride0, Logits_stride1,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute softmax over L for this head
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    # Now write output[h, :] = softmax @ Kc[:, :]
    out_row = tl.zeros([D_ckv], dtype=tl.float32)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv

        # Reload logits for softmax and compute soft vector for this tile
        soft_tile = tl.zeros([BLOCK_L], dtype=tl.float32)
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            e = tl.exp(vals - max_val)
            soft_tile += e  # accumulate sums; we’ll divide after
        # Divide by total sum_exp to get softmax per position
        soft_tile = soft_tile / sum_exp

        # Load Kc columns for this ks: Kc[:, ks] is [L, BLOCK_K]
        kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1  # need to define L for kc_ptrs; better: iterate ls and accumulate
        # Simpler: iterate over ls in the same way as soft_tile and accumulate
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
            kc = tl.load(kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
            soft_vec = soft_tile  # [BLOCK_L], reloaded for this tile. Better: compute per position by reloading Logits? We already have soft per tile; reload Logits per position? To avoid recomputation, we should compute soft once for whole row.

        # Since we cannot reuse soft_tile across all ks without storing, we compute soft per position by reloading Logits. To keep simple, we recompute soft per k0 by reloading Logits for that k0 window? This is fine: Triton supports looping over blocks.

        # Redo softmax per ks by reloading Logits for this k0
        max_val_k0 = -float('inf')
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            block_max = tl.max(vals, axis=0)
            max_val_k0 = tl.maximum(max_val_k0, block_max)

        sum_exp_k0 = 0.0
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            e = tl.exp(vals - max_val_k0)
            sum_exp_k0 += tl.sum(e, axis=0)

        soft_vec_k0 = tl.zeros([BLOCK_L], dtype=tl.float32)
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            e = tl.exp(vals - max_val_k0)
            soft_vec_k0 += e
        soft_vec_k0 = soft_vec_k0 / sum_exp_k0

        # Now accumulate out_row for these ks using Kc[:, ks]
        # Note: We need kc of shape [L, BLOCK_K] for current ks
        kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1  # same ls as above
        kc = tl.load(kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        # For each position l in this tile, we have soft_vec_k0[l] * kc[l, :]
        for ll in range(0, BLOCK_L):
            l_idx = l0 + ll
            if l_idx < L:
                soft_val = soft_vec_k0[ll]
                # Accumulate across BLOCK_K: out_row += soft_val * kc[l_idx, :]
                # But kc is 2D; we need per column. We can loop over kk:
                for kk in range(0, BLOCK_K):
                    k_idx = k0 + kk
                    if k_idx < D_ckv:
                        out_row[k_idx] += soft_val * kc[l_idx, kk]

    # Store out_row[h, :] to Out_ptr
    out_ptrs = Out_ptr + h * Out_stride0 + tl.arange(0, D_ckv) * Out_stride1
    tl.store(out_ptrs, out_row, mask=tl.arange(0, D_ckv) < D_ckv)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # num_qo_heads is fixed at 16 by the original code; head_dim_ckv=512, head_dim_kpe=64, page_size=1
        total_q = q_nope.shape[0]
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64
        device = q_nope.device

        # Prepare output and lse buffers (float32 for compute)
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Batch size
        batch_size = kv_indptr.shape[0] - 1

        inv_sm_scale = 1.0 / sm_scale  # Triton kernel uses this to apply scaling

        # Fixed blocks
        BLOCK_L = 128
        BLOCK_K_qn = 64   # for D_ckv=512, 512/64=8 tiles, but we loop; constexpr
        BLOCK_K_qp = 32   # for D_kpe=64, 64/32=2 tiles

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # Gather KV indices for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            M = int(kv_indices[page_end - 1].item() - kv_indices[page_beg].item() + 1)
            tok_idx = kv_indices[page_beg:page_end].to(device)  # int32 tensor, but we pass only values into Triton via kernel arguments

            # Gather Kc and Kp for these tokens
            Kc = ckv_cache[tok_idx].contiguous()  # [M, 512], float32
            Kp = kpe_cache[tok_idx].contiguous()  # [M, 64], float32

            # Number of queries in this batch element
            num_q = q_end - q_start

            # Loop over each query i
            for i in range(num_q):
                # Current query rows (no torch ops here)
                qn_row = q_nope[q_start + i].contiguous()  # [16, 512]
                qp_row = q_pe[q_start + i].contiguous()   # [16, 64]

                # Flatten query vectors into 1D for Triton kernels
                # We need q_nope[i, :] laid out as H*D_ckv = 16*512, and q_pe[i, :] as H*D_kpe = 16*64
                qn_flat = qn_row.reshape(-1).contiguous()  # [8192]
                qp_flat = qp_row.reshape(-1).contiguous()  # [1024]

                # Logits buffer for this (b, i): shape [num_qo_heads, M]
                Logits = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)

                # Compute q_nope contributions
                grid_qn = (num_qo_heads, triton.cdiv(M, BLOCK_L))
                compute_logits_qn_kernel[grid_qn](
                    qn_flat, Kc, Logits,
                    num_qo_heads, M, head_dim_ckv,
                    qn_flat.stride(0), 1,  # Qn is 1D, stride on 0th dim
                    Kc.stride(0), Kc.stride(1),
                    Logits.stride(0), Logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K_qn,
                )

                # Compute q_pe contributions and add to Logits
                grid_qp = (num_qo_heads, triton.cdiv(M, BLOCK_L))
                compute_logits_qp_kernel[grid_qp](
                    qp_flat, Kp, Logits,
                    num_qo_heads, M, head_dim_kpe,
                    qp_flat.stride(0), 1,  # 1D tensor
                    Kp.stride(0), Kp.stride(1),
                    Logits.stride(0), Logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K_qp,
                )

                # Apply causal mask and compute lse per head
                lse_mask_kernel[(num_qo_heads,)](
                    Logits, lse[q_start + i], M,
                    Logits.stride(0), Logits.stride(1),
                    1.0 / math.log(2.0),  # inv_ln2
                    BLOCK_L=BLOCK_L,
                )

                # Compute output[h, :] = softmax(Logits[h, :]) @ Kc[:, :] and store to output
                grid_out = (num_qo_heads, 1)
                softmax_matmul_kernel[grid_out](
                    Logits, Kc, output[q_start + i],
                    num_qo_heads, M, head_dim_ckv,
                    Logits.stride(0), Logits.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    output[q_start + i].stride(0), output[q_start + i].stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K_qn,
                )

        # Return outputs in the original format: output bfloat16 and lse float32
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
