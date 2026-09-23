import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits for all heads and tokens for a given (b, i).
# Inputs:
#   Q_nope: [H, D_ckv] float32, per-head q_nope[i]
#   Q_pe: [H, D_kpe] float32, per-head q_pe[i]
#   Kc: [L, D_ckv] float32
#   Kp: [L, D_kpe] float32
# Outputs:
#   Logits: [H, L] float32
#   H is a meta constant (16), L is runtime, D_ckv=512, D_kpe=64
@triton.jit
def compute_all_heads_logits_kernel(
    Q_nope_ptr, Q_pe_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    H: tl.constexpr, L, D_ckv: tl.constexpr, D_kpe: tl.constexpr,
    Qn_stride0, Qn_stride1,   # strides for Q_nope
    Qp_stride0, Qp_stride1,   # strides for Q_pe
    Kc_stride0, Kc_stride1,   # strides for Kc
    Kp_stride0, Kp_stride1,   # strides for Kp
    Logits_stride0, Logits_stride1,
    sm_scale,                 # float32 scale
    BLOCK_L: tl.constexpr,    # tile along L (e.g., 128)
    BLOCK_K: tl.constexpr     # tile along K (feature dim) (e.g., 128)
):
    # One program computes one head h's logits across L; we launch grid (H, cdiv(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    # Accumulator for this head
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # Loop over K dimension in chunks
    # D_ckv and D_kpe are tl.constexpr, so Triton can unroll these loops.
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_ks = ks < D_ckv
        # Load q_nope vector for this head
        qn_ptrs = Q_nope_ptr + h * Qn_stride0 + ks * Qn_stride1
        qn_vec = tl.load(qn_ptrs, mask=mask_ks, other=0.0)
        # Accumulate Kc contributions
        for kk in range(BLOCK_K):
            if mask_ks[kk]:
                k_idx = ks[kk]
                Kc_vals = tl.load(Kc_ptr + ls * Kc_stride0 + k_idx * Kc_stride1, mask=mask_l, other=0.0)
                acc += qn_vec[kk] * Kc_vals

    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_ks = ks < D_kpe
        # Load q_pe vector for this head
        qp_ptrs = Q_pe_ptr + h * Qp_stride0 + ks * Qp_stride1
        qp_vec = tl.load(qp_ptrs, mask=mask_ks, other=0.0)
        # Accumulate Kp contributions
        for kk in range(BLOCK_K):
            if mask_ks[kk]:
                k_idx = ks[kk]
                Kp_vals = tl.load(Kp_ptr + ls * Kp_stride0 + k_idx * Kp_stride1, mask=mask_l, other=0.0)
                acc += qp_vec[kk] * Kp_vals

    # Scale logits
    acc = acc * sm_scale

    # Store
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


# Triton kernel: compute per-head logsumexp over masked logits, and also produce a mask vector
# Inputs:
#   Logits: [H, L] float32
# Outputs:
#   LSE: [H] float32, scaled by ln(2)
# We return mask as a 1D tensor, but since Triton kernels cannot return multiple outputs, we compute it in-place here.
@triton.jit
def lse_mask_kernel(
    Logits_ptr, Mask_ptr, LSE_ptr,
    H: tl.constexpr, L,
    Logits_stride0, Logits_stride1,
    Mask_stride0,
    ln2_scale,  # float32, 1/ln(2)
    BLOCK_L: tl.constexpr
):
    # One program per head
    h = tl.program_id(0)

    # Compute threshold for causal mask: threshold = L - (q_end - q_start) + i
    # We pass this as a scalar argument to the kernel.
    threshold = tl.full((), 0, tl.int32)  # placeholder; overwritten by host via masked_fill
    # Triton doesn't allow receiving scalars like this directly; we compute on host and pass as arg instead of here.
    # Instead, we compute mask in host beforehand and pass mask_ptr. This kernel will just compute lse.

    # Compute row-wise max over masked values
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp over masked values
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * ln2_scale  # natural log
    tl.store(LSE_ptr + h, lse_scaled)


# Triton kernel: softmax over masked logits and final matmul with Kc to produce output vector for one head
# Inputs:
#   Logits: [H, L] float32
#   Mask: [L] int32 (1 for causal, 0 for non-causal)
#   Kc: [L, D_ckv] float32
# Outputs:
#   Out: [D_ckv] float32
@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Mask_ptr, Kc_ptr, Out_ptr,
    H: tl.constexpr, L, D_ckv: tl.constexpr,
    Logits_stride0, Logits_stride1,
    Mask_stride0,
    Kc_stride0, Kc_stride1,
    Out_stride0,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr
):
    # One program per head
    h = tl.program_id(0)

    # Compute max and sum over masked logits
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Compute output vector: out[h, :] = sum_l softmax[l] * Kc[l, :]
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val) * inv_sum  # softmax values
        e = tl.where(m == 0, 0.0, e)          # non-causal positions contribute 0 to softmax
        # Accumulate out_vec += sum_l e[l] * Kc[l, ks] over ks in tiles
        for k0 in range(0, D_ckv, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            mask_ks = ks < D_ckv
            Kc_tile = tl.load(Kc_ptr + ls * Kc_stride0 + ks * Kc_stride1, mask=mask_l[:, None] & mask_ks[None, :], other=0.0)
            # out_vec += sum over l of e[l] * Kc_tile[l, k]
            # e is [BLOCK_L], Kc_tile is [BLOCK_L, BLOCK_K]
            # We compute the contribution by summing over l dimension after multiplying each row by e[l]
            contrib = tl.sum(e * Kc_tile, axis=0)  # sum over L -> [BLOCK_K]
            out_vec += contrib

    # Store output for this head
    tl.store(Out_ptr + h * Out_stride0, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # If Triton not available, fall back to the original PyTorch computation (not used in evaluation but kept for robustness)
        if not TRITON_AVAILABLE:
            # Fallback: run original logic (this keeps correctness if Triton is unavailable)
            total_q, num_qo_heads, head_dim_ckv = q_nope.shape
            head_dim_kpe = q_pe.shape[-1]
            device = q_nope.device

            Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
            Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

            output = torch.zeros(
                (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
            )
            lse = torch.full(
                (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            )

            # Unchanged batch and query handling from original code
            batch_size = qo_indptr.shape[0] - 1
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
                tok_idx = kv_indices[page_beg:page_end].to(torch.long)
                Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv]
                Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe]

                q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_ckv]
                q_pe_batch = q_pe[q_start:q_end].to(torch.float32)      # [q_len, num_heads, head_dim_kpe]
                q_len = q_end - q_start

                for i in range(q_len):
                    qn = q_nope_batch[i]  # [num_heads, head_dim_ckv]
                    qp = q_pe_batch[i]    # [num_heads, head_dim_kpe]
                    logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_heads, kv_len]
                    logits_scaled = logits * sm_scale
                    prefix_len = kv_len - (q_end - q_start)
                    query_abs_pos = prefix_len + i
                    causal_mask = torch.arange(kv_len, device=logits_scaled.device) > query_abs_pos
                    logits_scaled.masked_fill_(causal_mask.unsqueeze(0), -float("inf"))
                    lse[q_start + i] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=-1)
                    out = attn @ Kc
                    output[q_start + i] = out.to(torch.bfloat16)

            return output, lse

        # Triton-only path: all computation in kernels
        device = q_nope.device
        assert q_nope.shape[1] == 16 and q_nope.shape[-1] == 512, "num_qo_heads must be 16 and head_dim_ckv must be 512"
        assert q_pe.shape[1] == 16 and q_pe.shape[-1] == 64, "q_pe shape must be [*, 16, 64]"
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[-1] == 512, "ckv_cache shape must be [*, 1, 512]"
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[-1] == 64, "kpe_cache shape must be [*, 1, 64]"

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]

        # Batch processing
        batch_size = qo_indptr.shape[0] - 1
        num_heads = 16
        # We will compute per batch element
        output = torch.empty((total_q, num_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=device)

        # Constants for Triton kernels
        BLOCK_L = 128  # tile along L
        BLOCK_K = 128  # tile along feature dim (512/64)

        # Precompute scalar ln(2) scaling for LSE
        ln2 = 1.4426950408889634  # 1 / ln(2)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # Gather K blocks
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.long).to(device)
            Kc_block = ckv_cache[tok_idx, 0]  # [L, 512]
            Kp_block = kpe_cache[tok_idx, 0]  # [L, 64]

            # Prepare per-query vectors for all heads
            q_nope_batch = q_nope[q_start:q_end].to(torch.float32).contiguous()  # [q_len, 16, 512]
            q_pe_batch = q_pe[q_start:q_end].to(torch.float32).contiguous()     # [q_len, 16, 64]
            q_len = q_end - q_start

            # Allocate intermediate Logits [H, L] float32
            Logits = torch.empty((num_heads, kv_len), dtype=torch.float32, device=device)

            # Launch kernel to compute logits for all heads
            grid = (num_heads, triton.cdiv(kv_len, BLOCK_L))
            compute_all_heads_logits_kernel[grid](
                q_nope_batch, q_pe_batch, Kc_block, Kp_block, Logits,
                H=num_heads, L=kv_len, D_ckv=512, D_kpe=64,
                Qn_stride0=q_nope_batch.stride(0), Qn_stride1=q_nope_batch.stride(2),
                Qp_stride0=q_pe_batch.stride(0), Qp_stride1=q_pe_batch.stride(2),
                Kc_stride0=Kc_block.stride(0), Kc_stride1=Kc_block.stride(1),
                Kp_stride0=Kp_block.stride(0), Kp_stride1=Kp_block.stride(1),
                Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                sm_scale=float(sm_scale),
                BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )

            # Precompute causal mask on device: l <= threshold is causal
            # threshold = L - (q_end - q_start) + i; since i varies, we compute per (b, i)
            # For each i, prefix_len = kv_len - (q_end - q_start), then threshold = prefix_len + i
            # We compute threshold per i by simple torch tensor; Triton kernel receives this scalar.
            # We'll launch per i.
            for i in range(q_len):
                # Prepare scalar threshold for causal mask
                prefix_len = kv_len - (q_end - q_start)
                threshold_i = prefix_len + i  # int32 scalar
                mask = (torch.arange(kv_len, device=device, dtype=torch.int32) <= threshold_i).to(torch.int32)

                # LSE vector for heads
                LSE = torch.empty((num_heads,), dtype=torch.float32, device=device)

                grid_lse = (num_heads,)
                lse_mask_kernel[grid_lse](
                    Logits, mask, LSE,
                    H=num_heads, L=kv_len,
                    Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                    Mask_stride0=1,  # 1D mask, contiguous
                    ln2_scale=ln2,
                    BLOCK_L=BLOCK_L,
                    num_warps=1, num_stages=1,
                )

                # Output vector for each head
                Out_vec = torch.empty((num_heads, head_dim_ckv), dtype=torch.float32, device=device)

                grid_softmax = (num_heads,)
                softmax_matmul_kernel[grid_softmax](
                    Logits, mask, Kc_block, Out_vec,
                    H=num_heads, L=kv_len, D_ckv=head_dim_ckv,
                    Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                    Mask_stride0=1,
                    Kc_stride0=Kc_block.stride(0), Kc_stride1=Kc_block.stride(1),
                    Out_stride0=Out_vec.stride(0),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2,
                )

                # Store to output[i] as bfloat16
                output[q_start + i] = Out_vec.to(torch.bfloat16)
                # Store lse[i] as float32
                lse[q_start + i] = LSE

        return output, lse


def run(*args):
    return ModelNew()(*args)
