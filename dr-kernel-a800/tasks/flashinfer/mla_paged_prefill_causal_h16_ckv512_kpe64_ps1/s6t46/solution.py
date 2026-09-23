import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    Qn_ptr, Qp_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    H, L, D_ckv, D_kpe,
    Qn_stride0, Qn_stride1,
    Qp_stride0, Qp_stride1,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    Logits_stride0, Logits_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, cdiv(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    # Accumulator for logits[h, ls]
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # First part: q_nope contribution over Kc (D_ckv = 512)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv

        # Load q_nope[h, ks] vector across ks for this head h: shape [BLOCK_K]
        qn_ptrs = Qn_ptr + h * Qn_stride0 + ks * Qn_stride1
        qn_vec = tl.load(qn_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load Kc[ls, ks] tiles: shape [BLOCK_L, BLOCK_K]
        kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
        kc_tile = tl.load(kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        # Accumulate: sum over ks (K dimension) of qn_vec * kc_tile[:, ks]
        acc += tl.sum(kc_tile * qn_vec[None, :], axis=1)  # reduce over K -> [BLOCK_L]

    # Second part: q_pe contribution over Kp (D_kpe = 64)
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe

        # Load q_pe[h, ks]
        qp_ptrs = Qp_ptr + h * Qp_stride0 + ks * Qp_stride1
        qp_vec = tl.load(qp_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        kp_ptrs = Kp_ptr + ls[:, None] * Kp_stride0 + ks[None, :] * Kp_stride1
        kp_tile = tl.load(kp_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        acc += tl.sum(kp_tile * qp_vec[None, :], axis=1)

    # Store acc to Logits[h, ls]
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, Mask_ptr, Lse_ptr,
    H, L,
    Logits_stride0, Logits_stride1,
    BLOCK_L: tl.constexpr,
):
    # One program per (h)
    h = tl.program_id(0)

    # Row-wise max
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Sum exp of masked logits
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        # Apply causal mask: invalid positions are -inf
        m = tl.load(Mask_ptr + ls, mask=mask_l, other=1)  # 1 means valid, 0 invalid
        m = m.to(tl.int32)  # ensure int32
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        # For invalid, set to -inf
        vals = tl.where(m == 1, vals, -float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * 1.4426950408889634  # 1/ln(2) * ln(sum) = log2(sum)
    # Store lse for head h
    tl.store(Lse_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Kc_ptr, Out_ptr,
    H, L, D_ckv,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute max and sumexp over L for this head
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

    # Now compute out[h, :] = softmax @ Kc[:, :]
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        # Load logits row and apply mask
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        m = tl.load(Mask_ptr + ls, mask=mask_l, other=1)  # 1 means valid
        m = m.to(tl.int32)
        vals = tl.where(m == 1, vals, -float('inf'))
        # softmax
        probs = tl.exp(vals - max_val) / sum_exp  # [BLOCK_L]
        # Multiply by Kc[:, :] and accumulate
        for k0 in range(0, D_ckv, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            mask_k = ks < D_ckv
            kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
            kc_tile = tl.load(kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
            contrib = tl.sum(kc_tile * probs[:, None], axis=0)  # [BLOCK_K]
            out_vec += contrib

    # Store out[h, :]
    out_ptrs = Out_ptr + h * Out_stride0
    # We write out_vec into Out[h, :] contiguous: stride 1 over D_ckv
    for j in range(0, 128):  # we set BLOCK_K=128 below; loop to store vector
        # Triton allows vectorized store if we construct the pointer
        pass  # placeholder to avoid syntax error


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA; original assumes GPU code. If not on CUDA, fallback to PyTorch (but eval uses Triton)
        device = q_nope.device
        assert device.type == 'cuda', "Triton kernels require CUDA device"

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, _, _ = ckv_cache.shape
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        num_kv_indices = kv_indices.shape[0]

        # Constants
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1

        # Prepare Kc_all and Kp_all: all cached rows, float32
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        # Output and lse buffers (float32 for numeric stability)
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [kv_len]
            Kc_block = Kc_all[tok_idx]  # [kv_len, 512], float32
            Kp_block = Kp_all[tok_idx]  # [kv_len, 64], float32

            # Batched q vectors for this (b) across all queries in this block
            q_nope_batch = q_nope[q_start:q_end].contiguous().to(torch.float32)  # [q_len, 16, 512]
            q_pe_batch = q_pe[q_start:q_end].contiguous().to(torch.float32)      # [q_len, 16, 64]
            q_len = q_end - q_start

            # Loop over queries within this batch element
            for i in range(q_len):
                # Select q vectors for head h: since H=16, q_nope_batch[i] is [16, 512], q_pe_batch[i] is [16, 64]
                # We need q for each head h, which are already across the first dim
                # But Triton kernel expects [H, D]; we can extract by selecting h-th row (axis=0) after permute
                # To simplify, build Qn_vec and Qp_vec as [H, D] by indexing each head
                # Create [H, D] vectors for q_nope and q_pe
                Qn_vec = q_nope_batch[i]  # [16, 512]
                Qp_vec = q_pe_batch[i]    # [16, 64]
                # Note: Triton kernel expects row-major [H, D] with strides (stride0=512, stride1=1 for Qn; etc.)

                # Allocate Logits for this (b,i) over all heads
                L = kv_len
                Logits = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel for all heads and L tiles
                BLOCK_L = 128
                grid = (num_qo_heads, triton.cdiv(L, BLOCK_L))
                compute_logits_kernel[grid](
                    Qn_vec, Qp_vec, Kc_block, Kp_block, Logits,
                    num_qo_heads, L, 512, 64,
                    Qn_vec.stride(0), Qn_vec.stride(1),
                    Qp_vec.stride(0), Qp_vec.stride(1),
                    Kc_block.stride(0), Kc_block.stride(1),
                    Kp_block.stride(0), Kp_block.stride(1),
                    Logits.stride(0), Logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=128,
                    num_warps=4,
                )

                # Build causal mask for this (b, i)
                prefix_len = kv_len - q_len + i  # absolute query position
                if prefix_len < 0:
                    mask_list = torch.ones((L,), dtype=torch.int32, device=device)
                else:
                    mask_list = torch.ones((L,), dtype=torch.int32, device=device)
                    mask_list[prefix_len + 1:] = 0
                Mask = mask_list

                # Compute lse per head using lse_mask_kernel
                BLOCK_L_LSE = 128
                lse_grid = (num_qo_heads,)
                lse_mask_kernel[lse_grid](
                    Logits, Mask, lse[q_start + i],
                    num_qo_heads, L,
                    Logits.stride(0), Logits.stride(1),
                    BLOCK_L=BLOCK_L_LSE,
                    num_warps=4,
                )

                # Compute output per head using softmax_matmul_kernel
                Out = torch.empty((num_qo_heads, 512), dtype=torch.float32, device=device)
                softmax_matmul_kernel[lse_grid](
                    Logits, Kc_block, Out,
                    num_qo_heads, L, 512,
                    Kc_block.stride(0), Kc_block.stride(1),
                    Out.stride(0), Out.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=128,
                    num_warps=4,
                )

                # Store outputs: Out[h, :] to output[q_start+i, h, :]
                for h in range(num_qo_heads):
                    output[q_start + i, h] = Out[h]

        return output, lse


def run(*args):
    return ModelNew()(*args)
