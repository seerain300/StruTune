import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: compute logits = Q @ K^T for a specific head h
# Operates on one head h per launch; tiles over Q (BLOCK_Q) and K (BLOCK_K).
# It computes acc[q, k] = sum_d Q[q, h, d] * K[k, h, d] for d in 0..127 (compile-time).
@triton.jit
def _compute_logits_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    q_stride_q, q_stride_h, q_stride_d,
    k_stride_k, k_stride_h, k_stride_d,
    logits_stride_q, logits_stride_h, logits_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_q = tl.program_id(0)  # tiles along Q
    pid_k = tl.program_id(1)  # tiles along K
    h = tl.program_id(2)      # specific head index

    q_start = pid_q * BLOCK_Q
    k_start = pid_k * BLOCK_K

    q_offsets = q_start + tl.arange(0, BLOCK_Q)  # (Q,)
    k_offsets = k_start + tl.arange(0, BLOCK_K)  # (K,)

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulator for logits chunk [Q, K]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Loop over d dimension with compile-time constant bound (head_dim=128)
    for d in range(0, head_dim):
        # Load Q[:, h, d] -> shape (Q,)
        q_vec = tl.load(
            q_ptr + q_offsets * q_stride_q + h * q_stride_h + d * q_stride_d,
            mask=q_mask,
            other=0.0
        )  # (Q,)
        # Load K[:, h, d] -> shape (K,)
        k_vec = tl.load(
            k_ptr + k_offsets * k_stride_k + h * k_stride_h + d * k_stride_d,
            mask=k_mask,
            other=0.0
        )  # (K,)
        # Outer product and accumulation: acc[q, k] += q_vec[q] * k_vec[k]
        acc += q_vec[:, None] * k_vec[None, :]

    # Store acc to logits_ptr[q, h, k]
    tl.store(
        logits_ptr + q_offsets[:, None] * logits_stride_q + h * logits_stride_h + k_offsets[None, :] * logits_stride_k,
        acc,
        mask=q_mask[:, None] & k_mask[None, :]
    )

# Triton kernel: compute per-(query, head) lse = logsumexp(masked logits) / ln(2)
# Operates on one head h per launch; tiles over Q (BLOCK_Q).
# Iterates over K in chunks of BLOCK_K with compile-time constant bounds up to 128.
@triton.jit
def _lse_masked_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    ln2, delta,  # delta = num_kv_tokens - num_q_tokens
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_q = tl.program_id(0)  # tiles along Q
    h = tl.program_id(1)      # specific head index

    q_start = pid_q * BLOCK_Q

    q_offsets = q_start + tl.arange(0, BLOCK_Q)  # (Q,)
    q_mask = q_offsets < num_q_tokens

    max_val = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Iterate K chunks up to head_dim with compile-time constant bound
    for k0 in range(0, 128):
        k_offsets = k0 + tl.arange(0, 128)  # (128,)
        k_mask = k_offsets < num_kv_tokens

        # Compute per q: max and sum_exp across K with causal mask
        for q_off in range(0, BLOCK_Q):
            q_idx = q_start + q_off
            if q_idx >= num_q_tokens:
                break
            cur_max = -float('inf')
            cur_sum = 0.0
            for kk in range(0, 128):
                k_idx = k0 + kk
                if k_idx >= num_kv_tokens:
                    break
                mask_causal = k_idx < (q_idx + 1 + delta)
                val = tl.load(
                    logits_ptr + q_idx * 0 + h * 0 + k_idx * 0,
                    mask=True,
                    other=-float('inf')
                )
                val = tl.where(mask_causal, val, -float('inf'))
                cur_max = tl.maximum(cur_max, val)
                cur_sum += tl.exp(val - cur_max)
            max_val[q_off] = cur_max
            sum_exp[q_off] = cur_sum

    # Compute lse = max + log(sum_exp) - ln2
    lse_vals = max_val + tl.log(sum_exp) - ln2
    tl.store(
        lse_ptr + q_offsets * 0,  # placeholder for pointer arithmetic
        lse_vals,
        mask=q_mask
    )

# Triton kernel: softmax along K with causal mask and output reduction against v_expanded for a specific head h
# Operates on one head h; tiles Q in BLOCK_Q, reduces over K via V chunks of BLOCK_D (compile-time constants).
@triton.jit
def _softmax_output_kernel(
    logits_ptr, v_ptr, output_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    logits_stride_q, logits_stride_h, logits_stride_k,
    v_stride_k, v_stride_h, v_stride_d,
    output_stride_q, output_stride_h, output_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_q = tl.program_id(0)  # tiles along Q
    h = tl.program_id(1)      # specific head index

    q_start = pid_q * BLOCK_Q

    q_offsets = q_start + tl.arange(0, BLOCK_Q)  # (Q,)
    q_mask = q_offsets < num_q_tokens

    # Load lse for this head (vector over Q)
    lse_vals = tl.load(lse_ptr + q_offsets * 0)

    # For each d chunk, compute softmax and accumulate output
    for d0 in range(0, head_dim, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)  # (D,)
        d_mask = d_offsets < head_dim

        # Compute and store output for each q in this tile
        for q_off in range(0, BLOCK_Q):
            q_idx = q_start + q_off
            if q_idx >= num_q_tokens:
                break
            # Recompute max and sum_exp across K with causal mask for softmax normalization
            max_val = -float('inf')
            for k0 in range(0, 128):
                mask_causal = k0 < (q_idx + 1 + (num_kv_tokens - (q_idx - q_start)))  # delta per q segment
                val = tl.load(
                    logits_ptr + q_idx * 0 + h * 0 + k0 * 0,
                    mask=True,
                    other=-float('inf')
                )
                val = tl.where(mask_causal, val, -float('inf'))
                max_val = tl.maximum(max_val, val)

            sum_exp = 0.0
            for k0 in range(0, 128):
                mask_causal = k0 < (q_idx + 1 + (num_kv_tokens - (q_idx - q_start)))
                val = tl.load(
                    logits_ptr + q_idx * 0 + h * 0 + k0 * 0,
                    mask=True,
                    other=-float('inf')
                )
                val = tl.where(mask_causal, val, -float('inf'))
                sum_exp += tl.exp(val - max_val)

            # Now compute output over D chunk: output[q, h, d] = sum_k softmax_k * v[k, h, d]
            for d_off in range(0, BLOCK_D):
                d_idx = d0 + d_off
                if d_idx >= head_dim:
                    break
                out_val = 0.0
                for k0 in range(0, 128):
                    mask_causal = k0 < (q_idx + 1 + (num_kv_tokens - (q_idx - q_start)))
                    val = tl.load(
                        logits_ptr + q_idx * 0 + h * 0 + k0 * 0,
                        mask=True,
                        other=-float('inf')
                    )
                    val = tl.where(mask_causal, val, -float('inf'))
                    soft = tl.exp(val - (lse_vals[q_off] + max_val + tl.log(sum_exp)))
                    # Load v[k0, h, d]
                    v_val = tl.load(
                        v_ptr + k0 * v_stride_k + h * v_stride_h + d_idx * v_stride_d
                    )
                    out_val += soft * v_val
                # Store output as bfloat16
                tl.store(
                    output_ptr + q_idx * output_stride_q + h * output_stride_h + d_idx * output_stride_d,
                    tl.cast(out_val, tl.bfloat16),
                    mask=True
                )
        # (We keep loops within compile-time bounds to satisfy Triton’s restrictions.)

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

        # Expand K/V by GQA ratio (repeat along head dimension)
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
            logits_seg = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=q_f32.device)
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=q_f32.device)
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

            logits_stride_q = logits_seg.stride(0)
            logits_stride_h = logits_seg.stride(1)
            logits_stride_k = logits_seg.stride(2)

            output_stride_q = output_seg.stride(0)
            output_stride_h = output_seg.stride(1)
            output_stride_d = output_seg.stride(2)

            # Launch Triton kernels per segment and head
            BLOCK_Q = 32
            BLOCK_K = 64
            BLOCK_D = 16

            # 1) Compute logits for each head h
            for h in range(self.num_qo_heads):
                _compute_logits_kernel[(triton.cdiv(num_q_tokens, BLOCK_Q), triton.cdiv(num_kv_tokens, BLOCK_K), 1)](
                    q_batch, k_expanded_batch, logits_seg,
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    q_stride_q, q_stride_h, q_stride_d,
                    k_stride_k, k_stride_h, k_stride_d,
                    logits_stride_q, logits_stride_h, logits_stride_k,
                    BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
                )

            # 2) LSE with causal mask
            for h in range(self.num_qo_heads):
                _lse_masked_kernel[(triton.cdiv(num_q_tokens, BLOCK_Q), 1)](
                    logits_seg, lse_seg,
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    self.ln2, (num_kv_tokens - num_q_tokens),
                    BLOCK_Q=BLOCK_Q, BLOCK_K=64
                )

            # 3) Softmax + output
            for h in range(self.num_qo_heads):
                _softmax_output_kernel[(triton.cdiv(num_q_tokens, BLOCK_Q), 1)](
                    logits_seg, v_expanded_batch, output_seg, lse_seg,
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    logits_stride_q, logits_stride_h, logits_stride_k,
                    v_stride_k, v_stride_h, v_stride_d,
                    output_stride_q, output_stride_h, output_stride_d,
                    BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, BLOCK_K=64
                )

            # Copy segment results to global output/lse
            output[q_start:q_end] = output_seg
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
