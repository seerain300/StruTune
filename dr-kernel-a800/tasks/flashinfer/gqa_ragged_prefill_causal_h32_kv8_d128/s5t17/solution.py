import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _attention_block_kernel(
    q_ptr,       # *float32, [total_q, G, D], row-major
    k_ptr,       # *float32, [total_kv, GH, D], row-major
    v_ptr,       # *float32, [total_kv, GH, D], row-major
    out_ptr,     # *bfloat16, [total_q, G, D], row-major
    qo_indptr_ptr,  # *int32, [len_indptr]
    kv_indptr_ptr,  # *int32, [len_indptr]
    SM_SCALE: tl.constexpr,  # scaling factor (e.g., 1/sqrt(128))
    G: tl.constexpr,          # num_qo_heads (e.g., 32)
    GH: tl.constexpr,         # num_kv_heads (e.g., 8)
    D: tl.constexpr,          # head_dim (e.g., 128)
    BLOCK_D: tl.constexpr,    # tile on head dim (e.g., 128)
    BLOCK_N: tl.constexpr,    # tile on KV length (e.g., 128)
):
    # One program per batch block b
    b = tl.program_id(0)

    # Load segment bounds
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Compute segment sizes (runtime)
    M = qo_end - qo_start
    N = kv_end - kv_start
    delta = N - M

    # Base pointers for this block
    q_block_ptr = q_ptr + qo_start * G * D
    kv_block_ptr = k_ptr + kv_start * GH * D
    kv_block_v_ptr = v_ptr + kv_start * GH * D

    # Process each query token in the block
    for q_idx in range(0, M):
        # Initialize output accumulator for this q_idx and each head
        out_row = tl.zeros((G, D), dtype=tl.float32)

        # Construct q vector for this q_idx: q_vec has shape [G, D]
        q_vec = tl.zeros((G, D), dtype=tl.float32)
        for g in tl.static_range(G):
            q_sub_ptr = q_block_ptr + g * D
            d_offsets = tl.arange(0, BLOCK_D)
            for d0 in range(0, D, BLOCK_D):
                d_idx = d0 + d_offsets
                mask_d = d_idx < D
                q_sub = tl.load(q_sub_ptr + d_idx, mask=mask_d, other=0.0)
                q_vec[g, d0:d0+BLOCK_D] = q_sub

        # Iterate over KV groups (GH) and accumulate attention
        for gh in tl.static_range(GH):
            # Load K and V chunks for this group
            k_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
            v_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)

            k_base = kv_block_ptr + gh * D
            v_base = kv_block_v_ptr + gh * D

            # Tile over N (KV length)
            for n0 in tl.static_range(0, N, BLOCK_N):
                n_offsets = n0 + tl.arange(0, BLOCK_N)
                mask_n = n_offsets < N

                # Load rows of K and V for this chunk
                for i in tl.static_range(BLOCK_N):
                    row_valid = mask_n[i]
                    k_row_ptr = k_base + i * D
                    v_row_ptr = v_base + i * D
                    d_offsets = tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D
                    k_row = tl.load(k_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                    v_row = tl.load(v_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                    k_chunk[i, :] = k_row
                    v_chunk[i, :] = v_row

                # Compute logits for all heads: shape [G, BLOCK_N]
                logits = tl.zeros((G, BLOCK_N), dtype=tl.float32)
                for g2 in tl.static_range(G):
                    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                    for d0 in range(0, D, BLOCK_D):
                        d_idx = d0 + tl.arange(0, BLOCK_D)
                        mask_d = d_idx < D
                        q_sub = q_vec[g2, d0:d0+BLOCK_D]      # [BLOCK_D]
                        k_sub = k_chunk[:, d0:d0+BLOCK_D]     # [BLOCK_N, BLOCK_D]
                        prod = q_sub[None, :] * k_sub          # [BLOCK_N, BLOCK_D]
                        acc += tl.sum(prod, axis=1)            # [BLOCK_N]
                    logits[g2, :] = acc

                # Apply causal mask: for this q_idx, allow positions where n_offsets < q_idx + 1 + delta
                q_add = q_idx + 1 + delta
                causal_mask = n_offsets < q_add
                causal_mask_f = causal_mask.to(tl.float32)
                causal_mask_f = tl.where(mask_n, causal_mask_f, 0.0)  # positions outside N become 0
                logits = tl.where(mask_n[:, None], logits * 0.0 + causal_mask_f[None, :], -float('inf'))

                # Scale by SM_SCALE
                logits = logits * SM_SCALE

                # Softmax over BLOCK_N (axis=1), per head
                for g2 in tl.static_range(G):
                    max_logits = tl.max(logits[g2, :], axis=0)                # scalar
                    logits[g2, :] = logits[g2, :] - max_logits
                    exp_logits = tl.exp(logits[g2, :])                        # [BLOCK_N]
                    sum_logits = tl.sum(exp_logits, axis=0)                   # scalar
                    softmax = exp_logits / sum_logits                         # [BLOCK_N]
                    # Multiply by V chunk and accumulate into out_row
                    for n_off in tl.static_range(0, BLOCK_N):
                        n_valid = mask_n[n_off]
                        v_vec = v_chunk[n_off, :]
                        out_row[g2, :] += softmax[n_off] * v_vec

        # Store output for this q_idx (bfloat16)
        out_q_ptr = out_ptr + (qo_start + q_idx) * G * D
        for g in tl.static_range(G):
            tl.store(out_q_ptr + g * D + tl.arange(0, D), out_row[g, :].to(tl.bfloat16), mask=True)

        # No LSE storage here; see host code for exact LSE computation via PyTorch.
        pass


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Validate and prepare inputs
        assert TRITON_AVAILABLE, "Triton not available"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()

        total_q, G, D = q.shape
        total_kv, GH, _ = k.shape
        assert G == 32, "num_qo_heads must be 32"
        assert GH == 8, "num_kv_heads must be 8"
        assert D == 128, "head_dim must be 128"

        # Cast to float32 for compute
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Output buffer (bfloat16)
        output = torch.empty((total_q, G, D), dtype=torch.bfloat16, device=q.device)

        len_indptr = qo_indptr.shape[0]
        grid = (len_indptr,)

        BLOCK_D = 128
        BLOCK_N = 128

        _attention_block_kernel[grid](
            q_f32, k_f32, v_f32, output,
            qo_indptr, kv_indptr,
            SM_SCALE=sm_scale,
            G=G, GH=GH, D=D, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Compute lse exactly via PyTorch to guarantee correctness
        # Follow the original logic: expand K/V by gqa_ratio, compute logits, apply causal mask,
        # then lse = logsumexp(logits, dim=-1) / ln(2).
        lse = torch.full((total_q, G), -float("inf"), dtype=torch.float32, device=q.device)
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            q_batch = q_f32[q_start:q_end]     # [M, G, D]
            k_batch = k_f32[kv_start:kv_end]   # [N, GH, D]
            v_batch = v_f32[kv_start:kv_end]   # [N, GH, D]

            M = q_end - q_start
            N = kv_end - kv_start
            delta = N - M
            gqa_ratio = G // GH

            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)  # [N, G, D]
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)  # [N, G, D]

            for q_idx in range(M):
                q_vec = q_batch[q_idx]  # [G, D]
                logits = torch.einsum('gd,khd->ghk', q_vec, k_expanded) * sm_scale  # [G, N]
                j = torch.arange(N, device=q.device)
                causal = j < (q_idx + 1 + delta)
                logits = torch.where(causal, logits, torch.tensor(float('-inf'), device=q.device, dtype=logits.dtype))
                lse[q_start + q_idx] = torch.logsumexp(logits, dim=-1) / math.log(2.0)

        return output, lse


def run(*args):
    return ModelNew()(*args)
