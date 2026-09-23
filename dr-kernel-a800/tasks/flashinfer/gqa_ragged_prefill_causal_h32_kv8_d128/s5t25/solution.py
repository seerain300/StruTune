import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_attention_kernel(
    q_ptr,        # *float32, shape [M, G, D], M = q_end - q_start
    k_ptr,        # *float32, shape [N, GH, D], N = kv_end - kv_start
    v_ptr,        # *float32, shape [N, GH, D]
    out_ptr,      # *bfloat16, shape [M, G, D]
    lse_ptr,      # *float32, shape [M, G]
    # indices: per-block device int32 tensors (shape [2])
    qo_indptr_ptr,  # *int32, [q_start, q_end]
    kv_indptr_ptr,  # *int32, [kv_start, kv_end]
    # sizes
    M: tl.constexpr,     # number of queries in this block
    N: tl.constexpr,     # number of key/value tokens in this block
    G: tl.constexpr,     # num_qo_heads (e.g., 32)
    GH: tl.constexpr,    # num_kv_heads * gqa_ratio (could be != G)
    D: tl.constexpr,     # head_dim (e.g., 128)
    SM_SCALE: tl.constexpr,  # scaling factor (e.g., 1.0 / sqrt(128))
    BLOCK_D: tl.constexpr,    # tile for head dim (128)
    BLOCK_N: tl.constexpr,    # tile for KV length (e.g., 128)
):
    # Load q range [q_start, q_end)
    q_start = tl.load(qo_indptr_ptr + 0).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + 1).to(tl.int32)
    if q_start >= q_end:
        return

    # Load kv range [kv_start, kv_end)
    kv_start = tl.load(kv_indptr_ptr + 0).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + 1).to(tl.int32)
    if kv_start >= kv_end:
        return

    # Compute delta (extra K/V tokens relative to Q tokens in this block)
    delta = kv_end - kv_start - (q_end - q_start)

    # Process each query position q_idx in the block
    for q_idx in range(0, M):
        # Accumulator for LSE per head [G]
        lse_row = tl.full((G,), -float('inf'), tl.float32)
        # Output accumulator for this q_idx: [G, D]
        out_row = tl.zeros((G, D), dtype=tl.float32)

        # Construct Q vector for this q_idx: q_vec has shape [G, D]
        q_vec = tl.zeros((G, D), dtype=tl.float32)
        for g in range(0, G):
            qg_ptr = q_ptr + q_idx * G * D + g * D
            d_offsets = tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            q_vec[g, :] = tl.load(qg_ptr + d_offsets, mask=mask_d, other=0.0)

        # Compute attention over K/V
        # Logits [G, N] in float32
        logits = tl.zeros((G, N), dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            n_offsets = n0 + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N

            # For each K/V group (head)
            for gh in range(0, GH):
                # Load K chunk [BLOCK_N, D] for this gh
                k_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
                v_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)

                k_base = k_ptr + (kv_start + n_offsets) * GH * D + gh * D
                v_base = v_ptr + (kv_start + n_offsets) * GH * D + gh * D

                for i in range(0, BLOCK_N):
                    row_valid = (i + n0) < N
                    k_row_ptr = k_base[i, :]  # base pointer for this row
                    v_row_ptr = v_base[i, :]  # base pointer for this row
                    d_offsets = tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D
                    k_row = tl.load(k_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                    v_row = tl.load(v_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                    k_chunk[i, :] = k_row
                    v_chunk[i, :] = v_row

                # Compute scores = Q @ K_chunk^T, shape [G, BLOCK_N]
                for g2 in range(0, G):
                    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                    for d0 in range(0, D, BLOCK_D):
                        d_offsets = d0 + tl.arange(0, BLOCK_D)
                        mask_d = d_offsets < D
                        q_sub = q_vec[g2, d_offsets]             # [BLOCK_D]
                        k_sub = k_chunk[:, d_offsets]            # [BLOCK_N, BLOCK_D]
                        prod = q_sub[None, :] * k_sub            # [BLOCK_N, BLOCK_D]
                        acc += tl.sum(prod, axis=1)              # [BLOCK_N]
                    logits[g2, n0:n0+BLOCK_N] = acc

        # Scale logits by SM_SCALE
        logits = logits * SM_SCALE

        # Apply causal mask: for each q_idx, kv position j < q_idx + 1 + delta
        q_add = q_idx + 1 + delta
        for n0 in range(0, N, BLOCK_N):
            n_offsets = n0 + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N
            for g2 in range(0, G):
                mask_causal = n_offsets < q_add
                logits[g2, n0:n0+BLOCK_N] = tl.where(mask_causal & mask_n, logits[g2, n0:n0+BLOCK_N], -float('inf'))

        # Compute row-wise LSE (base 2): logsumexp
        for g2 in range(0, G):
            max_score = tl.max(logits[g2, :])
            sum_exp = tl.sum(tl.exp(logits[g2, :] - max_score))
            lse_val = max_score + tl.log(sum_exp)  # natural log
            lse_val = lse_val / math.log(2.0)      # convert to base-2
            lse_row[g2] = lse_val

        # Compute softmax over N for each g, then output = softmax * V
        for g2 in range(0, G):
            max_score = tl.max(logits[g2, :])
            sum_exp = tl.sum(tl.exp(logits[g2, :] - max_score))
            soft = tl.exp(logits[g2, :] - max_score) / sum_exp  # [N]
            # out[g2, d] += sum over n of soft[n] * V[n, gh, d] for all gh
            total_acc = tl.zeros((D,), dtype=tl.float32)
            for n0 in range(0, N, BLOCK_N):
                n_offsets = n0 + tl.arange(0, BLOCK_N)
                mask_n = n_offsets < N
                for gh2 in range(0, GH):
                    v_base = v_ptr + (kv_start + n_offsets) * GH * D + gh2 * D
                    v_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
                    for i in range(0, BLOCK_N):
                        row_valid = (i + n0) < N
                        v_row_ptr = v_base[i, :]
                        d_offsets = tl.arange(0, BLOCK_D)
                        mask_d = d_offsets < D
                        v_row = tl.load(v_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                        v_chunk[i, :] = v_row
                    total_acc += tl.sum(soft[n0:n0+BLOCK_N] * v_chunk[:, d_offsets], axis=0)
            out_row[g2, :] += total_acc

        # Store output for this q_idx and all heads
        out_base = out_ptr + q_idx * G * D
        for g2 in range(0, G):
            d_offsets = tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            tl.store(out_base + g2 * D + d_offsets, out_row[g2, d_offsets], mask=mask_d)
        # Store LSE for this q_idx and all heads
        lse_base = lse_ptr + q_idx * G
        for g2 in range(0, G):
            tl.store(lse_base + g2, lse_row[g2])

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Validate shapes to match reference expectations
        assert q.shape[1] == 32, "Expected q with 32 heads"
        assert q.shape[2] == 128, "Expected head_dim=128"
        assert k.shape[1] == 8, "Expected k with 8 heads"
        assert k.shape[2] == 128, "Expected head_dim=128"
        assert v.shape == k.shape, "v must have same shape as k"

        device = q.device
        M_total, G, D = q.shape
        N_total, GHk, Dk = k.shape
        assert D == 128 and GHk == 8 and Dk == 128, "Invalid shapes"

        # Allocate outputs (computed by Triton)
        output = torch.empty((M_total, G, D), dtype=torch.bfloat16, device=device)
        lse = torch.full((M_total, G), -float("inf"), dtype=torch.float32, device=device)

        # Convert to float32 for computation; output is bfloat16, lse is float32
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            # Extract per-block ranges
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            M = q_end - q_start
            N = kv_end - kv_start

            # Slice tensors for this block
            q_block = q_f32[q_start:q_end]  # [M, G, D]
            k_block = k_f32[kv_start:kv_end]  # [N, GH, D]
            v_block = v_f32[kv_start:kv_end]  # [N, GH, D]

            # Create device tensors for per-block indices (int32 on device)
            qo_indptr_b = qo_indptr[b:b+1]  # device int32 tensor [1, 2]
            kv_indptr_b = kv_indptr[b:b+1]  # device int32 tensor [1, 2]

            # Launch Triton kernel once per block; pass M, N as constexpr
            _block_attention_kernel[(1,)](
                q_block, k_block, v_block,
                output, lse,
                qo_indptr_b, kv_indptr_b,
                M=M, N=N, G=32, GH=8, D=128, SM_SCALE=sm_scale,
                BLOCK_D=128, BLOCK_N=128,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
