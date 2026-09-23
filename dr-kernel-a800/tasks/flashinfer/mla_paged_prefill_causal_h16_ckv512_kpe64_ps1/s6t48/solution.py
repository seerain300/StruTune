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

    # Loop over K dimension in chunks: first D_ckv for Kc, then remaining for Kp
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv
        # Load q vectors for head h at ks (from Qn_ptr as [H, D_ckv])
        q_ptrs = Qn_ptr + h * Qn_stride0 + ks * Qn_stride1
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Load Kc rows at positions ls for ks
        Kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
        Kc_mat = tl.load(Kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        # Accumulate dot: (BLOCK_K x 1) @ (1 x BLOCK_K) -> (BLOCK_L x 1)
        acc += tl.sum(Kc_mat * q_vec[None, :], axis=1)

    # Kp contributions
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe
        q_ptrs = Qp_ptr + h * Qp_stride0 + ks * Qp_stride1
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]
        Kp_ptrs = Kp_ptr + ls[:, None] * Kp_stride0 + ks[None, :] * Kp_stride1
        Kp_mat = tl.load(Kp_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        acc += tl.sum(Kp_mat * q_vec[None, :], axis=1)

    # Store acc to Logits[h, ls]
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def build_mask_kernel(
    Prefix_ptr, Mask_ptr, L, prefix_len,
    Mask_stride0: tl.constexpr,
):
    # Grid: (L,)
    j = tl.program_id(0)
    if j < L:
        # prefix_len can be negative; if so, all positions are causal => mask=1
        mask_val = 1
        if prefix_len >= 0:
            mask_val = 1 if j <= prefix_len else 0
        tl.store(Mask_ptr + j * Mask_stride0, mask_val)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, Mask_ptr, Lse_ptr, H, L,
    Logits_stride0, Logits_stride1,
    inv_ln2: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute row-wise max
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        mask_l = mask_l & tl.load(Mask_ptr + ls, mask=mask_l, other=1)  # load mask as int
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp(vals - max_val)
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        mask_l = mask_l & tl.load(Mask_ptr + ls, mask=mask_l, other=1)  # load mask as int
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2  # divide by ln(2)
    tl.store(Lse_ptr + h, lse_scaled)


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

    # Compute softmax over L using tiled reduction
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

    # Accumulate out[h, :] = sum_l softmax[h, l] * Kc[l, :]
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)  # [BLOCK_L]
        soft = e / sum_exp
        # Load Kc rows and accumulate
        Kc_ptrs = Kc_ptr + ls * Kc_stride0 + tl.arange(0, BLOCK_K) * Kc_stride1  # BLOCK_K=128
        for k0 in range(0, D_ckv, 128):
            ks = k0 + tl.arange(0, 128)
            mask_k = ks < D_ckv
            Kc_mat = tl.load(Kc_ptrs + k0, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, 128]
            out_vec += tl.sum(soft[:, None] * Kc_mat, axis=0)

    # Store out[h, :]
    out_ptrs = Out_ptr + h * Out_stride0  # Out_stride0 = D_ckv, Out_stride1 = 1
    tl.store(out_ptrs, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
        device = q_nope.device
        # The function expects q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale
        # Cast caches to float32 for computation
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Process batch elements
        q_len_total = int(qo_indptr[-1].item())
        assert q_len_total == total_q

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Prepare q_nope and q_pe for each batch element: shape [num_q, H, D]
        q_nope_reshaped = q_nope.view(total_q, num_qo_heads, head_dim_ckv).contiguous().to(torch.float32)  # [Q, H, D_ckv]
        q_pe_reshaped = q_pe.view(total_q, num_qo_heads, head_dim_kpe).contiguous().to(torch.float32)   # [Q, H, D_kpe]

        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            # If no queries in this batch element, skip
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # Compute tok_idx from kv_indptr and kv_indices for this batch
            # Note: page_size=1 by assumption
            if b >= kv_indptr.shape[0] - 1:
                # Edge case: no kv for this batch element
                # Fill output and lse with zeros for this batch element
                # We still need to set lse for each i in [q_start, q_end-1] = -inf
                for i in range(q_len):
                    lse[q_start + i] = -float('inf')
                # Output remains empty as there’s no computation; but shape already allocated
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [kv_len]
            # Gather Kc and Kp for this block
            Kc_block = Kc_all[tok_idx]  # [kv_len, 512]
            Kp_block = Kp_all[tok_idx]  # [kv_len, 64]

            # Loop over queries in this batch element
            for i in range(q_len):
                # Prepare Qn_vec and Qp_vec: each is [H, D]
                Qn_vec = q_nope_reshaped[q_start + i]  # [H, D_ckv]
                Qp_vec = q_pe_reshaped[q_start + i]   # [H, D_kpe]
                # Allocate Logits for this head over L
                Logits = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel for all heads and L tiles
                BLOCK_L = 128
                grid = (num_qo_heads, triton.cdiv(kv_len, BLOCK_L))
                compute_logits_kernel[grid](
                    Qn_vec, Qp_vec, Kc_block, Kp_block, Logits,
                    num_qo_heads, kv_len, 512, 64,
                    Qn_vec.stride(0), Qn_vec.stride(1),
                    Qp_vec.stride(0), Qp_vec.stride(1),
                    Kc_block.stride(0), Kc_block.stride(1),
                    Kp_block.stride(0), Kp_block.stride(1),
                    Logits.stride(0), Logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=128,
                    num_warps=4,
                )

                # Compute prefix_len for causal mask: absolute query position in this block
                # Note: q_end is exclusive
                prefix_len = kv_len - q_len + i  # absolute position when considering only this block
                # Build mask tensor on device using Triton kernel
                mask = torch.empty((kv_len,), dtype=torch.int32, device=device)
                build_mask_kernel[(kv_len,)](
                    torch.tensor(prefix_len, dtype=torch.int32, device=device),
                    mask, kv_len, prefix_len,
                    1,  # Mask_stride0 is 1
                    num_warps=1,
                )

                # Launch lse_mask_kernel for each head to compute lse
                lse_scaled = torch.empty((1,), dtype=torch.float32, device=device)
                lse_mask_kernel[(num_qo_heads,)](
                    Logits, mask, lse_scaled,
                    num_qo_heads, kv_len,
                    Logits.stride(0), Logits.stride(1),
                    1.0 / math.log(2.0),
                    BLOCK_L=128,
                    num_warps=1,
                )
                lse[q_start + i] = lse_scaled[0]  # store scalar per head

                # Launch softmax_matmul_kernel for each head to compute output
                out_vec = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                softmax_matmul_kernel[(num_qo_heads,)](
                    Logits, Kc_block, out_vec,
                    num_qo_heads, kv_len, 512,
                    Logits.stride(0), Logits.stride(1),
                    Kc_block.stride(0), Kc_block.stride(1),
                    head_dim_ckv,  # Out_stride0
                    1,              # Out_stride1 is 1 since contiguous
                    BLOCK_L=128, BLOCK_K=128,
                    num_warps=4,
                )
                # Store into output[q_start + i, :, :]
                output[q_start + i] = out_vec

        return output, lse


def run(*args):
    return ModelNew()(*args)
