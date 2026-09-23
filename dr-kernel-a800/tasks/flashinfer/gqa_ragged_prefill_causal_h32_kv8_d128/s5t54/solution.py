import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_attention_triton(
    q_ptr,          # *float32, [M, G, D] (device tensor for this block)
    k_ptr,          # *float32, [N, G, D] (device tensor for this block, expanded)
    v_ptr,          # *float32, [N, G, D] (device tensor for this block, expanded)
    out_ptr,        # *float32, [M, G, D] (device output buffer for this block)
    lse_ptr,        # *float32, [M, G]    (device lse buffer for this block)
    q_start,        # int32 scalar
    q_end,          # int32 scalar
    kv_start,       # int32 scalar
    kv_end,         # int32 scalar
    M,              # int32 scalar: number of queries in this block
    N,              # int32 scalar: number of KV tokens in this block
    sm_scale,       # float32 scalar
    G: tl.constexpr,      # 32
    D: tl.constexpr,      # 128
):
    # Load block indices
    q_start_val = tl.load(q_start)  # already scalar
    q_end_val = tl.load(q_end)      # already scalar
    kv_start_val = tl.load(kv_start)  # already scalar
    kv_end_val = tl.load(kv_end)      # already scalar

    # Compute actual M and N (passed; just sanity)
    # Note: We can use q_end - q_start as M, and kv_end - kv_start as N.

    inv_ln2 = 1.0 / math.log(2.0)

    # Loop over each query position i and head g
    for i in range(0, M):
        # Base offset for q[i] (we'll use flat indexing later)
        # We'll compute logits[i, g, :] for each g
        for g in range(0, G):
            # Compute logits[i, g, :] = sum_d q[i, g, d] * k[:, g, d], with causal mask j < i + 1 + (N - M)
            logits_vec = tl.zeros((N,), dtype=tl.float32)
            # Loop over d from 0 to D-1
            for d in range(0, D):
                # q[i, g, d]
                # q_ptr is laid out as [M, G, D] contiguous: index = ((i * G) + g) * D + d
                q_val = tl.load(q_ptr + ((i * G) + g) * D + d)
                # Accumulate over j
                # For j in [0, N):
                # k[j, g, d]
                # k_ptr is [N, G, D] contiguous: index = j * (G * D) + g * D + d
                for j in range(0, N):
                    k_val = tl.load(k_ptr + j * (G * D) + g * D + d)
                    # Apply causal mask: allow if j < i + 1 + (N - M)
                    allow = (j < (i + 1 + (N - M)))
                    # Multiply and accumulate only if allowed
                    logits_vec[j] += q_val * k_val * sm_scale if allow else 0.0

            # Store logits for this (i, g) vector
            # out_ptr layout [M, G, D] contiguous: index = (i * G + g) * D + offset over j
            out_row_ptr = out_ptr + (i * G + g) * D
            # Copy logits_vec into out_row_ptr
            # We can do element-wise store
            for jj in range(0, N):
                tl.store(out_row_ptr + jj, logits_vec[jj])

            # Compute LSE for this (i, g)
            # logsumexp over logits_vec (length N)
            # First, find max for numerical stability
            max_logit = -float('inf')
            for jj in range(0, N):
                max_logit = tl.maximum(max_logit, logits_vec[jj])
            exp_sum = 0.0
            for jj in range(0, N):
                exp_sum += tl.exp(logits_vec[jj] - max_logit)
            lse_val = max_logit + tl.log(exp_sum) * inv_ln2
            # Store to lse_ptr[i, g]
            lse_ptr[i * G + g] = lse_val

            # Compute softmax over logits_vec and output[i, g, :] = softmax @ v[:, g, :]
            sum_softmax = 0.0
            for jj in range(0, N):
                sum_softmax += tl.exp(logits_vec[jj] - max_logit)
            for jj in range(0, N):
                p = tl.exp(logits_vec[jj] - max_logit) / sum_softmax
                # v[j, g, d]
                v_val = tl.load(v_ptr + j * (G * D) + g * D + d)
                # Accumulate over d: output[i, g, d] += p * v[j, g, d]
                # We need to multiply and accumulate for each d; do it by summing p*v across j
                # But out_ptr already contains logits, we need to recompute? Instead, compute directly:
                # We'll compute output[i, g, :] vector by iterating over d again
                # However, Triton doesn't support storing to a specific d directly here; we'll recompute output.
                # To avoid confusion, we compute output by accumulating per d.
                pass
            # The above comment block shows a plan; since Triton lacks convenient row-wise output accumulation,
            # we implement output accumulation by iterating over j and d:
            # Prepare output[i, g, :] as float32
            out_row = tl.zeros((D,), dtype=tl.float32)
            for jj in range(0, N):
                p = tl.exp(logits_vec[jj] - max_logit) / sum_softmax
                # v_ptr is [N, G, D] contiguous: v[j, g, d] at index j*(G*D) + g*D + d
                # Multiply and accumulate for each d
                # We need a d-loop
                for dd in range(0, D):
                    v_val = tl.load(v_ptr + jj * (G * D) + g * D + dd)
                    out_row[dd] += p * v_val
            # Store out_row to out_ptr[(i, g, :)]
            # out_ptr layout: index = (i * G + g) * D + d
            out_row_base = out_ptr + (i * G + g) * D
            for dd in range(0, D):
                tl.store(out_row_base + dd, out_row[dd])

    # End of kernel. We stored outputs and lse as required.


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA and Triton availability
        assert q.is_cuda and k.is_cuda and v.is_cuda, "ModelNew expects CUDA tensors"
        assert TRITON_AVAILABLE, "Triton is not available"

        # Shapes and constants
        total_q = q.shape[0]
        total_kv = k.shape[0]
        G = 32
        GH = 8
        D = 128
        assert G == 32 and GH == 8 and D == 128
        len_indptr = qo_indptr.shape[0]

        # Output and lse across all blocks
        output_blocks = []
        lse_blocks = []

        # Process each block b
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            M = q_end - q_start
            N = kv_end - kv_start

            # Slice per-block tensors
            q_batch = q[q_start:q_end]                    # [M, G, D] bf16, but we cast to float32 for kernel
            k_batch = k[kv_start:kv_end]                 # [N, GH


def run(*args):
    return ModelNew()(*args)
