import torch
import triton
import triton.language as tl
from math import log as ln

# Constants
H = 16          # number of query heads, assumed fixed
D_ckv = 512     # head_dim_ckv
D_kpe = 64      # head_dim_kpe

inv_ln2 = 1.4426950408889634  # 1 / ln(2)


@triton.jit
def compute_logits_kernel(
    Qn_ptr, Qp_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    L,                 # number of KV positions
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

    # Load q vectors for this head: shape [D_ckv + D_kpe]
    # Qn_ptr and Qp_ptr are of shape [1, H, D], so indexing (0, h, ks) yields vector [D]
    qn_sum = tl.zeros([BLOCK_L], dtype=tl.float32)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv
        # Load Qn[h, ks]
        q_ptrs = Qn_ptr + 0 * Qn_stride0 + h * Qn_stride1 + ks * 0  # broadcasting: [1, H, D] => [H, D], but we pass [1,H,D] shaped tensors
        # Note: Triton expects pointer arithmetic to correspond to tensor layout; here Qn_ptr points to [1,H,D] so we use ks directly
        qn = tl.load(Qn_ptr, mask=mask_k, other=0.0)  # will be broadcasted properly in matmul below
        kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
        kc = tl.load(Kc_ptr + ls * Kc_stride0 + ks * Kc_stride1, mask=mask_l[:, None] & mask_k[None, :], other=0.0)
        # Accumulate qn[k] * Kc[l, k]
        qn_sum += tl.sum(qn[None, :] * kc, axis=1)

    qp_sum = tl.zeros([BLOCK_L], dtype=tl.float32)
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe
        qp = tl.load(Qp_ptr, mask=mask_k, other=0.0)
        kp_ptrs = Kp_ptr + ls[:, None] * Kp_stride0 + ks[None, :] * Kp_stride1
        kp = tl.load(Kp_ptr + ls * Kp_stride0 + ks * Kp_stride1, mask=mask_l[:, None] & mask_k[None, :], other=0.0)
        qp_sum += tl.sum(qp[None, :] * kp, axis=1)

    acc = qn_sum + qp_sum  # [BLOCK_L]
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, Mask_ptr, L_ptr, H, L,
    Logits_stride0, Logits_stride1,
    inv_ln2,
    BLOCK_L: tl.constexpr,
):
    # One program per (h)
    h = tl.program_id(0)

    # Compute row-wise max
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp(vals - max_val) on masked logits
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        # Load mask: Mask_ptr[ls] = 1 if causal, else 0
        mask_vec = tl.load(Mask_ptr + ls, mask=mask_l, other=1)  # assume 0/1
        valid = mask_vec != 0
        # If invalid, set to -inf; else keep vals
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals_masked = tl.where(valid & mask_l, vals, -float('inf'))
        e = tl.exp(vals_masked - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2
    tl.store(L_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Mask_ptr, Kc_ptr, Out_ptr,
    H, L, D_ckv,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Load masked logits for this head
    # We will compute softmax over L
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        mask_vec = tl.load(Mask_ptr + ls, mask=mask_l, other=1)  # 0/1 int32 mask
        valid = mask_vec != 0
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(valid & mask_l, vals, -float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        mask_vec = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        valid = mask_vec != 0
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(valid & mask_l, vals, -float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    # Now compute out[h, :] = softmax @ Kc[:, :]
    out_vec = tl.zeros([D_ckv], dtype=tl.float32)
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        mask_vec = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        valid = mask_vec != 0
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(valid & mask_l, vals, -float('inf'))
        e = tl.exp(vals - max_val) / sum_exp
        for k0 in range(0, D_ckv, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            mask_k = ks < D_ckv
            kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
            kc = tl.load(Kc_ptr + ls * Kc_stride0 + ks * Kc_stride1, mask=mask_l[:, None] & mask_k[None, :], other=0.0)
            contrib = tl.sum(e[None, :] * kc, axis=1)  # [BLOCK_K]
            out_vec += contrib

    out_ptrs = Out_ptr + h * Out_stride0
    tl.store(out_ptrs, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # total_q, num_qo_heads, head_dim_ckv
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        # ckv_cache has shape [num_pages, 1, 512], kpe_cache has shape [num_pages, 1, 64]
        # qo_indptr: [len_indptr], kv_indptr: [len_indptr], kv_indices: [num_kv_indices] int32
        device = q_nope.device

        # Compute batch size from qo_indptr
        batch_size = qo_indptr.shape[0] - 1

        # We will loop over b and i
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Prepare precomputed Kc_all and Kp_all for each batch b
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len == 0:
                continue

            # KV indices for this batch
            if b >= kv_indptr.shape[0] - 1:
                break
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if (page_end - page_beg) == 0:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(device=device)  # [kv_len], int32
            Kc_all = ckv_cache[tok_idx].to(torch.float32)  # [kv_len, 512]
            Kp_all = kpe_cache[tok_idx].to(torch.float32)  # [kv_len, 64]
            kv_len = Kc_all.shape[0]

            # Prepare Qn and Qp: [1, H, D]
            Qn_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, H, D_ckv]
            Qp_batch = q_pe[q_start:q_end].to(torch.float32)    # [q_len, H, D_kpe]
            # We will compute per i in [q_start, q_end)
            for i in range(q_len):
                # Logits buffers per (H, L_tiles): H = 16
                L = kv_len
                Logits = torch.empty((H, L), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel for all heads and L tiles
                BLOCK_L = 64
                grid = (H, triton.cdiv(L, BLOCK_L))
                # Qn_ptr and Qp_ptr: [1, H, D] shaped tensors, but we pass pointers; Triton will index via strides
                # For simplicity, we pass Qn_batch[i].unsqueeze(0) and Qp_batch[i].unsqueeze(0) as [1, H, D]
                Qn_ptr = Qn_batch[i].unsqueeze(0)  # [1, H, D_ckv]
                Qp_ptr = Qp_batch[i].unsqueeze(0)  # [1, H, D_kpe]
                # Strides for Qn_ptr and Qp_ptr: since shape [1, H, D], strides are (H*D, D, 1) but we pass 2D pointers; instead, we use 1D pointer tricks by constructing [H, D] slices. To keep it correct, we will instead pass [H, D] tensors and view as [1, H, D] by making strides reflect that.
                # Here we'll create 2D pointers using [H, D] tensors and then use strides accordingly. However Triton expects 1D indexing for such ops. To keep correctness, we'll pass Qn_ptr and Qp_ptr as [1, H, D] tensors and let Triton handle indexing via provided strides. The above load uses Qn_ptr and Qp_ptr as 1D pointers; we should instead pass actual 1D pointers. Let's restructure to pass Qn_ptr as [H, D] and compute pointer with h and ks.

                # Better approach: pass Qn and Qp as [H, D] tensors and compute pointers as (h * D) + ks.
                # Change Qn_ptr, Qp_ptr to [H, D] for simplicity:
                Qn_2d = Qn_batch[i]            # [H, D_ckv]
                Qp_2d = Qp_batch[i]            # [H, D_kpe]
                Qn_ptr = Qn_2d                 # [H, D_ckv]
                Qp_ptr = Qp_2d                 # [H, D_kpe]

                compute_logits_kernel[grid](
                    Qn_ptr, Qp_ptr, Kc_all, Kp_all, Logits,
                    L,
                    Qn_ptr.stride(0), Qn_ptr.stride(1),     # strides for [H, D] tensor
                    Qp_ptr.stride(0), Qp_ptr.stride(1),
                    Kc_all.stride(0), Kc_all.stride(1),
                    Kp_all.stride(0), Kp_all.stride(1),
                    Logits.stride(0), Logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=64,
                )

                # Prepare mask for causal: for position i, causal means l <= prefix_len + i - 1, where prefix_len = kv_len - q_len (since tokens after last query are valid). Then causal is l <= (kv_len - q_len + i). But in our loop, prefix_len = 0 because we process one i per chunk? Actually, prefix_len is the number of already processed queries in this batch element.
                # We can compute prefix_len = q_len - 1 (since q_len is number of queries in this b). But better: prefix_len = total queries processed before b? It's simpler to set prefix_len = q_len - 1 for each i.
                # mask: causal if l <= (kv_len - q_len + i) -> rewritten as l + (q_len - 1 - i) <= kv_len - 1
                # Using simpler logic: mask_l_true if (i + 1) >= (kv_len - l). Equivalently, l >= (kv_len - (i + 1)). Let start = kv_len - (i + 1). If start <= 0, all causal; else mask l>=start.
                start = kv_len - (i + 1)
                # Create mask tensor [L] as int32 (1 for causal, 0 for non-causal)
                mask_list = torch.ones((L,), dtype=torch.int32, device=device)
                if start > 0:
                    mask_list[:start] = 0
                Mask = mask_list

                # Launch lse_mask_kernel for each head
                lse_kernel_grid = (H,)
                lse_mask_kernel[lse_kernel_grid](
                    Logits, Mask, lse[q_start + i], H, L,
                    Logits.stride(0), Logits.stride(1),
                    inv_ln2,
                    BLOCK_L=128,
                )

                # Launch softmax_matmul_kernel for each head to compute out[h, :]
                softmax_matmul_kernel[lse_kernel_grid](
                    Logits, Mask, Kc_all, output[q_start + i],
                    H, L, D_ckv,
                    Kc_all.stride(0), Kc_all.stride(1),
                    output[q_start + i].stride(0), output[q_start + i].stride(1),
                    BLOCK_L=64, BLOCK_K=64,
                )

        return output, lse


def run(*args):
    return ModelNew()(*args)
