import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits = (qn @ Kc.T) + (qp @ Kp.T) for a single head h
# Inputs:
#   qn_ptr: [Hc] float32
#   qp_ptr: [Hp] float32
#   Kc_ptr: [L, Hc] float32
#   Kp_ptr: [L, Hp] float32
#   logits_ptr: [L] float32
#   sm_scale: float32
#   Hc: int (head_dim_ckv)
#   Hp: int (head_dim_kpe)
#   L: int (num_tokens for this batch element)
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_ptr,
    sm_scale, Hc: tl.int32, Hp: tl.int32, L: tl.int32,
    qn_stride, qp_stride,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    logits_stride,
    BLOCK_K: tl.constexpr
):
    # One program computes the full logits vector for a single head
    # We iterate over K dimension in chunks of BLOCK_K and accumulate.
    # Initialize logits vector
    i_offsets = tl.arange(0, L)
    # We'll fill logits[i_offsets] by accumulating over K chunks
    acc = tl.zeros([L], dtype=tl.float32)

    # Reduction over Kc: qn @ Kc.T
    k0 = 0
    while k0 < Hc:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < Hc
        # Load qn slice
        qn_chunk = tl.load(qn_ptr + k_offsets * qn_stride, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Load Kc chunk [L, BLOCK_K]
        Kc_chunk = tl.load(
            Kc_ptr + i_offsets[:, None] * Kc_stride0 + k_offsets[None, :] * Kc_stride1,
            mask=(i_offsets[:, None] < L) & (mask_k[None, :]),
            other=0.0
        )  # [L, BLOCK_K]
        # Accumulate acc += qn_chunk * Kc_chunk.sum(axis=1)
        # Kc_chunk is [L, BLOCK_K], sum over axis=1 gives [L], then elementwise multiply with qn_chunk and reduce into acc
        # Note: Triton supports broadcasting and reductions
        acc += tl.sum(Kc_chunk * qn_chunk[None, :], axis=1)  # [L]
        k0 += BLOCK_K

    # Reduction over Kp: qp @ Kp.T
    k0 = 0
    while k0 < Hp:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < Hp
        qp_chunk = tl.load(qp_ptr + k_offsets * qp_stride, mask=mask_k, other=0.0)  # [BLOCK_K]
        Kp_chunk = tl.load(
            Kp_ptr + i_offsets[:, None] * Kp_stride0 + k_offsets[None, :] * Kp_stride1,
            mask=(i_offsets[:, None] < L) & (mask_k[None, :]),
            other=0.0
        )  # [L, BLOCK_K]
        acc += tl.sum(Kp_chunk * qp_chunk[None, :], axis=1)
        k0 += BLOCK_K

    acc *= sm_scale
    # Store logits
    tl.store(logits_ptr + i_offsets * logits_stride, acc, mask=i_offsets < L)


# Kernel 2: Compute output row for a single head h:
# out_row = softmax(logits_scaled) @ Kc
# This kernel computes lse (logsumexp) in first pass, then second pass writes output row.
# Inputs:
#   logits_ptr: [L] float32
#   Kc_ptr: [L, Hc] float32
#   out_ptr: [Hc] float32
#   sm_scale: float32 (unused here; we need scaled logits, so we read logits and apply scale)
#   Hc: int (head_dim_ckv)
#   L: int (num_tokens)
@triton.jit
def matvec_row_kernel(
    logits_ptr, Kc_ptr, out_ptr,
    sm_scale: tl.float32, Hc: tl.int32, L: tl.int32,
    Kc_stride0, Kc_stride1,
    out_stride,
    BLOCK_K: tl.constexpr
):
    # First pass: compute row-wise max and sum(exp(logits_scaled - max))
    # We iterate logits in chunks to find max and sum.
    row_max = -float('inf')
    row_sum = 0.0

    k0 = 0
    while k0 < L:
        i = k0 + tl.arange(0, BLOCK_K)
        mask_i = i < L
        logits_chunk = tl.load(logits_ptr + i * 1, mask=mask_i, other=-float('inf'))  # [BLOCK_K]
        # Compute scaled logits chunk
        scaled = logits_chunk * sm_scale
        # Max over this chunk
        chunk_max = tl.max(scaled, axis=0)
        row_max = tl.maximum(row_max, chunk_max)
        # Sum of exp(scaled - row_max) over this chunk
        exp_chunk = tl.exp(scaled - row_max)
        row_sum += tl.sum(exp_chunk, axis=0)
        k0 += BLOCK_K

    # lse for this row
    lse = tl.log(row_sum) / tl.log(2.0)

    # Second pass: compute output vector
    out_vec = tl.zeros([Hc], dtype=tl.float32)
    k0 = 0
    while k0 < L:
        i = k0 + tl.arange(0, BLOCK_K)
        mask_i = i < L
        logits_chunk = tl.load(logits_ptr + i * 1, mask=mask_i, other=0.0)  # [BLOCK_K]
        scaled = logits_chunk * sm_scale
        attn_chunk = tl.exp(scaled - lse)  # [BLOCK_K]
        # For each i in chunk, accumulate into out_vec[j] += attn_chunk[i] * Kc[i, j]
        # Loop over each i index in the chunk to update columns j
        # We'll do a simple outer accumulation: for m in [BLOCK_K], j loop
        # Note: Triton allows nested loops; BLOCK_K is constexpr.
        for m in range(BLOCK_K):
            valid = (k0 + m) < L
            # Load attn scalar
            attn_i = attn_chunk[m]
            if valid:
                # Load Kc row (k0+m) across j in chunks if needed
                # But we can load one row at a time by looping j in chunks
                # We need j dimension loop: Hc is runtime, so we loop over Hc in chunks.
                # However, Triton doesn't let us easily loop over runtime Hc; so we assume Hc fits in one BLOCK_K vector (overkill) or we implement a multi-pass over Hc.
                # To keep it general, we perform j accumulation via a while loop over j in chunks.
                j0 = 0
                while j0 < Hc:
                    j = j0 + tl.arange(0, BLOCK_K)
                    mask_j = j < Hc
                    # Load Kc row for this i
                    Kc_row = tl.load(
                        Kc_ptr + (k0 + m) * Kc_stride0 + j * Kc_stride1,
                        mask=mask_j,
                        other=0.0
                    )  # [BLOCK_K]
                    # Accumulate out_vec[j] += attn_i * Kc_row
                    out_vec += tl.where(mask_j, attn_i * Kc_row, 0.0)
                    j0 += BLOCK_K
        k0 += BLOCK_K

    # Store output row
    j = tl.arange(0, Hc)
    tl.store(out_ptr + j * out_stride, out_vec, mask=j < Hc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes: q_nope: [B, 16, 512], q_pe: [B, 16, 64], ckv_cache: [N, 1, 512], kpe_cache: [N, 1, 64]
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA for Triton."
        B = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Squeeze cache singleton dim
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [N, 64]

        # Prepare output and lse tensors
        output = torch.empty((B, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)  # we will cast to bfloat16 later
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q_nope.device)

        # Iterate over batch and heads; launch Triton kernels
        BLOCK_K = 64  # chunk size for token/K dimension in reductions

        for b in range(B):
            # Compute tok_idx for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                output[b].zero_()
                lse[b] = -float('inf')
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # token indices
            L = tok_idx.numel()

            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L, 512]
            Kp = Kp_all[tok_idx]  # [L, 64]
            # Ensure contiguous
            Kc = Kc.contiguous()
            Kp = Kp.contiguous()

            # Prepare qn and qp slices per head
            for h in range(num_qo_heads):
                qn = q_nope[b, h, :].contiguous().to(torch.float32)  # [512]
                qp = q_pe[b, h, :].contiguous().to(torch.float32)   # [64]

                # Allocate logits vector
                logits = torch.empty((L,), dtype=torch.float32, device=q_nope.device)

                # Launch matmul_add_row_kernel to compute logits for this head
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    sm_scale,
                    head_dim_ckv, head_dim_kpe, L,
                    1, 1,  # qn_stride, qp_stride (elements per index)
                    Kc.stride(0), Kc.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    1,  # logits_stride
                    BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Allocate output row
                out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=q_nope.device)

                # Launch matvec_row_kernel to compute out_row and lse for this head
                matvec_row_kernel[(1,)](
                    logits, Kc, out_row,
                    sm_scale,
                    head_dim_ckv, L,
                    Kc.stride(0), Kc.stride(1),
                    1,
                    BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Store output[b, h, :] = out_row
                output[b, h, :] = out_row

                # lse[b, h] already computed inside matvec_row_kernel; retrieve by recomputing from logits if needed
                # Here we compute lse separately via Triton as above; we could have it stored by the kernel argument,
                # but we didn't. Since the kernel computes lse, we need to get it; however, we don't have a direct
                # return value. To keep Triton-only, we compute lse in-kernel by a slight modification: we could pass
                # a pointer to store lse. For simplicity, we compute lse by torch on logits_scaled outside; but the
                # original requirement prohibits torch here. Therefore, we compute lse via a small Triton kernel that
                # reads logits and computes max/sum. To avoid extra kernel, we compute lse in forward using torch:
                # This is a temporary fix, but to adhere to "TRITON-ONLY", we will compute lse using Triton.
                # We'll use a tiny Triton kernel to compute lse.
                # Define a kernel to compute lse per head for given logits and sm_scale
                @triton.jit
                def compute_lse_kernel(logits_ptr, lse_ptr, L: tl.int32, sm_scale: tl.float32):
                    row_max = -float('inf')
                    row_sum = 0.0
                    k0 = 0
                    while k0 < L:
                        i = k0 + tl.arange(0, 64)
                        mask_i = i < L
                        logits_chunk = tl.load(logits_ptr + i * 1, mask=mask_i, other=-float('inf'))
                        scaled = logits_chunk * sm_scale
                        chunk_max = tl.max(scaled, axis=0)
                        row_max = tl.maximum(row_max, chunk_max)
                        exp_chunk = tl.exp(scaled - row_max)
                        row_sum += tl.sum(exp_chunk, axis=0)
                        k0 += 64
                    lse = tl.log(row_sum) / tl.log(2.0)
                    tl.store(lse_ptr, lse)

                # Launch lse computation for this head and batch
                lse[b, h] = compute_lse_kernel[(1,)](
                    logits, lse[b, h], L, sm_scale, num_warps=2, num_stages=2
                )

        # Cast output to bfloat16 to match original q_nope/q_pe dtype
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
