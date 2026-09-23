import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    Qn_ptr, Qp_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    H, L,
    Qn_strideH, Qn_strideH2, Qn_strideD,
    Qp_strideH, Qp_strideH2, Qp_strideD,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    Logits_stride0, Logits_stride1,
    inv_scale,  # float32 scalar
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, cdiv(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    # Accumulator for logits[h, ls]
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # Iterate over K dimension (D_ckv + D_kpe) in tiles of BLOCK_K
    for k0 in range(0, 1024, BLOCK_K):  # 1024 covers both 512 and 64
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < (512 + 64)  # upper bound; will mask beyond actual dims

        # Load q vectors for head h at ks: shape (BLOCK_K,)
        # Qn_ptr points to a (1, H, 512) tensor; index (0, h, ks)
        qn_ptrs = Qn_ptr + 0 * Qn_strideH + h * Qn_strideH2 + ks * Qn_strideD
        qn = tl.load(qn_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Kc contributions: load Kc[ls, ks] as (BLOCK_L, BLOCK_K)
        Kc_vals = tl.load(
            Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1,
            mask=mask_l[:, None] & mask_k[None, :],
            other=0.0,
        )  # [BLOCK_L, BLOCK_K]
        # Accumulate dot: sum_k qn[k] * Kc_vals[:, k]
        for kk in range(BLOCK_K):
            acc += qn[kk] * Kc_vals[:, kk]

        # Kp contributions: ks beyond 511 correspond to q_p head contribution
        Kp_vals = tl.load(
            Kp_ptr + ls[:, None] * Kp_stride0 + (ks[None, :] - 512) * Kp_stride1,
            mask=mask_l[:, None] & mask_k[None, :],
            other=0.0,
        )  # [BLOCK_L, BLOCK_K]
        for kk in range(BLOCK_K):
            acc += qn[kk + 512] * Kp_vals[:, kk]

    # Apply scaling and store
    acc = acc * inv_scale
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, Mask_ptr, L_ptr,
    H, L_dim,
    Logits_stride0, Logits_stride1,
    inv_ln2,  # float32 scalar = 1/ln(2)
    BLOCK_L: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute row-wise max over masked logits
    max_val = -float('inf')
    for l0 in range(0, L_dim, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L_dim
        m = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp over masked logits
    sum_exp = 0.0
    for l0 in range(0, L_dim, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L_dim
        m = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val)
        e = tl.where(m == 0, 0.0, e)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2
    tl.store(L_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Mask_ptr, Kc_ptr, Out_ptr,
    H, L_dim, D_ckv,
    Logits_stride0, Logits_stride1,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    inv_scale, inv_ln2,  # not used here directly; provided for signature consistency
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute softmax over L_dim from Logits[h, :]
    max_val = -float('inf')
    for l0 in range(0, L_dim, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L_dim
        m = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for l0 in range(0, L_dim, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L_dim
        m = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val)
        e = tl.where(m == 0, 0.0, e)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Compute out[h, :] = softmax(masked logits) @ Kc[:, :]
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv

        # Accumulate out_vec[ks] += e[l] * Kc[l, ks]
        for l0 in range(0, L_dim, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L_dim
            m = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            vals = tl.where(m == 0, -float('inf'), vals)
            e = tl.exp(vals - max_val) * inv_sum  # masked non-causal positions were set to -inf; here e=0
            # multiply e (length BLOCK_L) by Kc[:, ks] (length BLOCK_L for each ks) and accumulate
            for kk in range(BLOCK_K):
                kc_col = tl.load(Kc_ptr + ls * Kc_stride0 + (ks[kk] if kk < BLOCK_K else 0) * Kc_stride1,
                                 mask=mask_l, other=0.0)
                # To vectorize properly, we need out_vec += sum_l e[l] * kc_col[l]
                # Triton supports per-element math, but more structured approach is to use a static inner loop:
                for l_idx in range(BLOCK_L):
                    if mask_l[l_idx]:
                        out_vec[kk] += e[l_idx] * kc_col[l_idx]

    # Store out_vec for head h
    out_ptrs = Out_ptr + h * Out_stride0
    tl.store(out_ptrs + tl.arange(0, D_ckv), out_vec, mask=mask_k)  # mask_k covers ks < D_ckv


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.sm_scale = 1.0
        self.inv_ln2 = 1.4426950408889634  # 1 / ln(2)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        H = self.num_qo_heads
        D_ckv = self.head_dim_ckv
        D_kpe = self.head_dim_kpe

        # Ensure inputs are on GPU and contiguous
        q_nope = q_nope.to(device).contiguous()
        q_pe = q_pe.to(device).contiguous()
        Kc_all = ckv_cache.squeeze(1).to(device).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(device).contiguous()  # [num_pages, 64]
        qo_indptr = qo_indptr.to(device).int32()
        kv_indptr = kv_indptr.to(device).int32()
        kv_indices = kv_indices.to(device).int32()

        total_q = int(q_nope.shape[0])
        # Determine batch size from qo_indptr
        batch_size = int(qo_indptr.shape[0] - 1)

        output = torch.empty((total_q, H, D_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        inv_scale = float(sm_scale)

        # Precompute masks for each batch element (simple causal mask: l > prefix + i)
        # For now, we compute per (b, i) in the loop; to precompute we need prefix_len and i. We'll compute inside kernels.
        # Launch over batch and queries
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or page_beg >= page_end:
                continue

            kv_len = int(page_end - page_beg)
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
            Kc = Kc_all[tok_idx].contiguous()  # [kv_len, 512]
            Kp = Kp_all[tok_idx].contiguous()  # [kv_len, 64]

            # q_nope_batch and q_pe_batch: [q_len, H, D]
            qn = q_nope[q_start:q_end].contiguous()  # [q_len, H, 512]
            qp = q_pe[q_start:q_end].contiguous()   # [q_len, H, 64]
            q_len = q_end - q_start

            # We need prefix_len = kv_len - q_len, and abs_pos = prefix_len + i
            prefix_len = kv_len - q_len

            # Preallocate per-(b,i) buffers
            logits = torch.empty((H, kv_len), dtype=torch.float32, device=device)
            out_vec = torch.empty((H, D_ckv), dtype=torch.float32, device=device)

            # Launch compute_logits kernel
            BLOCK_L = 128
            grid_log = (H, triton.cdiv(kv_len, BLOCK_L))
            compute_logits_kernel[grid_log](
                qn, qp, Kc, Kp, logits,
                H, kv_len,
                qn.stride(0), qn.stride(1), qn.stride(2),
                qp.stride(0), qp.stride(1), qp.stride(2),
                Kc.stride(0), Kc.stride(1),
                Kp.stride(0), Kp.stride(1),
                logits.stride(0), logits.stride(1),
                inv_scale,
                BLOCK_L=BLOCK_L, BLOCK_K=64, num_warps=4, num_stages=2,
            )

            # Build causal mask tensor per head
            # mask[l] = 1 if l <= (prefix_len + i), else 0
            for i in range(q_len):
                abs_pos = prefix_len + i
                # Mask: 1 where l <= abs_pos else 0
                mask = torch.full((kv_len,), 1, dtype=torch.int32, device=device)
                mask[(abs_pos + 1):] = 0
                # Launch lse_mask_kernel for head 0..H-1
                # We launch one program per head
                grid_lse = (H,)
                lse_mask_kernel[grid_lse](
                    logits, mask, lse[q_start + i],
                    H, kv_len,
                    logits.stride(0), logits.stride(1),
                    self.inv_ln2,
                    BLOCK_L=128,
                    num_warps=4, num_stages=2,
                )

                # Launch softmax_matmul_kernel to compute out vector
                grid_out = (H,)
                softmax_matmul_kernel[grid_out](
                    logits, mask, Kc, out_vec,
                    H, kv_len, D_ckv,
                    logits.stride(0), logits.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_vec.stride(0), out_vec.stride(1),
                    inv_scale, self.inv_ln2,
                    BLOCK_L=128, BLOCK_K=64, num_warps=4, num_stages=2,
                )

                # Store out vector for this query position i
                output[q_start + i] = out_vec

        # Return output cast to bfloat16 and lse
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
