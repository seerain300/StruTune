import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Kernel 1: Compute logits = Q @ K^T for the entire segment and a specific head h
# Operates on one head h; processes full Q (q_vec) and full K (k_vec) via 1D vectorized loads/stores.
@triton.jit
def _compute_logits_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    q_stride_q, q_stride_h, q_stride_d,
    k_stride_k, k_stride_h, k_stride_d,
    logits_stride_q, logits_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr  # set to cover full segment sizes
):
    h = tl.program_id(0)  # specific head index

    # Create 1D indices for Q and K
    q_idx = tl.arange(0, BLOCK_Q)  # (Q,)
    k_idx = tl.arange(0, BLOCK_K)  # (K,)

    q_mask = q_idx < num_q_tokens
    k_mask = k_idx < num_kv_tokens

    # Accumulator for logits [Q, K]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Loop over d dimension (head_dim), Triton allows this range since bounds are integers
    for d in range(0, head_dim):
        # Load Q[:, h, d] -> shape (Q,)
        q_vec = tl.load(
            q_ptr + q_idx * q_stride_q + h * q_stride_h + d * q_stride_d,
            mask=q_mask,
            other=0.0
        )  # (Q,)
        # Load K[:, h, d] -> shape (K,)
        k_vec = tl.load(
            k_ptr + k_idx * k_stride_k + h * k_stride_h + d * k_stride_d,
            mask=k_mask,
            other=0.0
        )  # (K,)
        # Outer product accumulation: acc[q, k] += q_vec[q] * k_vec[k]
        acc += q_vec[:, None] * k_vec[None, :]

    # Store acc to logits_ptr[q, h, k]
    tl.store(
        logits_ptr + q_idx[:, None] * logits_stride_q + h * 0 + k_idx[None, :] * logits_stride_k,
        acc,
        mask=(q_idx[:, None] < num_q_tokens) & (k_idx[None, :] < num_kv_tokens)
    )

# Kernel 2: Compute lse per query for a specific head h from logits_seg (2D vector)
# Operates on one head h; uses 1D vector for Q, computes max and sum_exp across K with causal mask.
@triton.jit
def _lse_masked_kernel(
    logits_seg_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    ln2, delta,  # delta = num_kv_tokens - num_q_tokens
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    h = tl.program_id(0)  # specific head index

    # 1D indices for Q and K
    q_idx = tl.arange(0, BLOCK_Q)  # (Q,)
    k_idx = tl.arange(0, BLOCK_K)  # (K,)

    q_mask = q_idx < num_q_tokens
    k_mask = k_idx < num_kv_tokens

    # Compute lse for each q in vectorized form
    cur_max = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    cur_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Load logits_seg[q, k] for all q and k
    vals = tl.load(
        logits_seg_ptr + q_idx[:, None] * 0 + k_idx[None, :] * 0,
        mask=(q_idx[:, None] < num_q_tokens) & (k_idx[None, :] < num_kv_tokens),
        other=-float('inf')
    )  # (Q, K)

    # Build causal mask: k < (q_idx + 1 + delta) for each q
    allowed_max = q_idx + 1 + delta  # (Q,)
    mask_causal = k_idx[None, :] < allowed_max[:, None]  # (Q, K)
    vals = tl.where(mask_causal, vals, -float('inf'))

    # Reduce across K to get max and sum_exp per q
    cur_max = tl.maximum(cur_max, tl.max(vals, axis=1))
    cur_sum = tl.sum(tl.exp(vals - cur_max[:, None]), axis=1)

    # Store lse per q
    tl.store(
        lse_ptr + q_idx * 0,
        cur_max + tl.log(cur_sum) - ln2,
        mask=q_mask
    )

# Kernel 3: Softmax along K with causal mask and output reduction against v_expanded for a specific head h
# Operates on one head h; 1D vector for Q, reduce over K via V chunks of BLOCK_D
@triton.jit
def _softmax_output_kernel(
    logits_seg_ptr, v_expanded_ptr, output_seg_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    logits_stride_q, logits_stride_k,
    v_stride_k, v_stride_h, v_stride_d,
    output_stride_q, output_stride_h, output_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    h = tl.program_id(0)  # specific head index

    # 1D indices for Q and K
    q_idx = tl.arange(0, BLOCK_Q)  # (Q,)
    k_idx = tl.arange(0, BLOCK_K)  # (K,)

    q_mask = q_idx < num_q_tokens
    k_mask = k_idx < num_kv_tokens

    # Load lse for this head (vector over Q)
    lse_vals = tl.load(lse_ptr + q_idx * 0)

    # For each d chunk, compute softmax and accumulate output
    for d0 in range(0, head_dim, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)  # (D,)
        d_mask = d_idx < head_dim

        # Compute softmax per q across K and accumulate output
        # Load logits_seg[q, k] for all q and k
        vals = tl.load(
            logits_seg_ptr + q_idx[:, None] * 0 + k_idx[None, :] * 0,
            mask=(q_idx[:, None] < num_q_tokens) & (k_idx[None, :] < num_kv_tokens),
            other=-float('inf')
        )  # (Q, K)
        # Causal mask: k < (q_idx + 1 + delta)
        allowed_max = q_idx + 1 + (num_kv_tokens - num_q_tokens)  # (Q,)
        mask_causal = k_idx[None, :] < allowed_max[:, None]      # (Q, K)
        vals = tl.where(mask_causal, vals, -float('inf'))

        # Compute max and sum_exp per q
        max_val = tl.max(vals, axis=1)
        sum_exp = tl.sum(tl.exp(vals - max_val[:, None]), axis=1)  # (Q,)

        # Compute softmax per k: exp(vals - lse_vals[:, None] - max_val[:, None])
        soft = tl.exp(vals - max_val[:, None][..., None] - lse_vals[:, None][..., None])  # (Q, K)

        # Now reduce soft over K against v_expanded[:, h, d] to produce output[q, h, d]
        for d_off in range(0, BLOCK_D):
            d_curr = d0 + d_off
            if d_curr >= head_dim:
                break
            out_vec = tl.zeros((BLOCK_Q,), dtype=tl.float32)
            v_vec = tl.load(
                v_expanded_ptr + k_idx * v_stride_k + h * v_stride_h + d_curr * v_stride_d,
                mask=k_mask,
                other=0.0
            )  # (K,)
            out_vec = tl.sum(soft * v_vec[None, :], axis=1)  # (Q,)
            # Store output as bfloat16
            tl.store(
                output_seg_ptr + q_idx * output_stride_q + h * output_stride_h + d_curr * output_stride_d,
                tl.cast(out_vec, tl.bfloat16),
                mask=q_mask
            )

# ModelNew forward (Triton-only, no torch elementwise ops in host)
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Triton-only forward: ensure tensors are on CUDA
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

        # Cast to float32 for compute
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Expand K/V by GQA ratio
        k_expanded = k_f32.repeat_interleave(self.gqa_ratio, dim=1).contiguous()
        v_expanded = v_f32.repeat_interleave(self.gqa_ratio, dim=1).contiguous()

        # Output and lse buffers
        total_q, num_qo_heads, head_dim = q_f32.shape
        total_kv, num_kv_heads, _ = k_f32.shape
        assert num_qo_heads == self.num_qo_heads
        assert num_kv_heads == self.num_kv_heads
        assert head_dim == self.head_dim

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q_f32.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q_f32.device)

        # Process segments
        n_batches = qo_indptr.shape[0] - 1
        for b in range(n_batches):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Allocate per-segment buffers
            logits_seg = torch.empty((num_q_tokens, num_kv_tokens), dtype=torch.float32, device=q_f32.device)
            lse_seg = torch.empty((num_q_tokens,), dtype=torch.float32, device=q_f32.device)
            output_seg = torch.empty((num_q_tokens, self.num_qo_heads, head_dim), dtype=torch.bfloat16, device=q_f32.device)

            # Strides for segment tensors
            q_batch = q_f32[q_start:q_end]
            k_expanded_batch = k_expanded[kv_start:kv_end]
            v_expanded_batch = v_expanded[kv_start:kv_end]

            q_stride_q = q_batch.stride(0)
            q_stride_h = q_batch.stride(1)
            q_stride_d = q_batch.stride(2)

            k_stride_k = k_expanded_batch.stride(0)
            k_stride_h = k_expanded_batch.stride(1)
            k_stride_d = k_expanded_batch.stride(2)

            v_stride_k = v_expanded_batch.stride(0)
            v_stride_h = v_expanded_batch.stride(1)
            v_stride_d = v_expanded_batch.stride(2)

            # Grid sizes: cover entire segment
            BLOCK_Q = num_q_tokens  # set to cover full Q
            BLOCK_K = num_kv_tokens  # set to cover full K
            BLOCK_D = 16  # output dim chunk

            # 1) Compute logits for each head h
            for h in range(self.num_qo_heads):
                _compute_logits_kernel[(1,)](
                    q_batch, k_expanded_batch, logits_seg,
                    num_q_tokens, num_kv_tokens, head_dim,
                    q_stride_q, q_stride_h, q_stride_d,
                    k_stride_k, k_stride_h, k_stride_d,
                    logits_seg.stride(0), logits_seg.stride(1),
                    BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
                )

                # 2) LSE with causal mask for head h
                _lse_masked_kernel[(1,)](
                    logits_seg, lse_seg,
                    num_q_tokens, num_kv_tokens, head_dim,
                    self.ln2, (num_kv_tokens - num_q_tokens),
                    BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
                )

                # 3) Softmax + output for head h
                _softmax_output_kernel[(1,)](
                    logits_seg, v_expanded_batch, output_seg, lse_seg,
                    num_q_tokens, num_kv_tokens, head_dim,
                    logits_seg.stride(0), logits_seg.stride(1),
                    v_expanded_batch.stride(0), v_expanded_batch.stride(1), v_expanded_batch.stride(2),
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
                )

                # Copy segment results to global output/lse for head h
                output[q_start:q_end, h, :] = output_seg.to(torch.bfloat16)
                lse[q_start:q_end, h] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
