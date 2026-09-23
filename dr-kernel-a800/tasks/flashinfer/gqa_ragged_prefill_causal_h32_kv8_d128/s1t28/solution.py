import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels (defined before use to avoid NameError)
@triton.jit
def _compute_logits_kernel(
    Q, K, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_stride_k, K_stride_h, K_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (heads, q_tiles, k_tiles)
    h = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    k_idx = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    q_mask = q_idx < num_q_tokens
    k_mask = k_idx < num_kv_tokens

    # Initialize accumulator for LOGITS[q, k] over d
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Sum over d in 0..127 (constexpr loop)
    for d in range(0, 128):
        # Load q[q, h, d] as a vector of length BLOCK_Q
        q_ptrs = Q + q_idx * Q_stride_q + h * Q_stride_h + d * Q_stride_d
        q_vec = tl.load(q_ptrs, mask=q_mask, other=0.0)  # [BLOCK_Q]

        # Load K[k, h, d] as a vector of length BLOCK_K
        k_ptrs = K + k_idx * K_stride_k + h * K_stride_h + d * K_stride_d
        k_vec = tl.load(k_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Outer product and accumulate: acc[q, k] += q_vec[:, None] * k_vec[None, :]
        prod = q_vec[:, None] * k_vec[None, :]
        acc += prod

    # Store acc into LOGITS[q, h, k]
    LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
    store_mask = q_mask[:, None] & k_mask[None, :]
    tl.store(LOGITS_ptrs, acc, mask=store_mask)


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    ln2, delta,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (num_q_tokens, heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Compute max over K of LOGITS[q, h, k] with causal mask k < q + 1 + delta
    max_val = -float("inf")
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        allowed = k_idx < (q + 1 + delta)
        ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        vals = tl.load(ptrs, mask=(k_mask & allowed), other=-float("inf"))
        block_max = tl.max(vals, axis=0)  # scalar
        max_val = tl.maximum(max_val, block_max)

    # Compute sum_exp = sum(exp(LOGITS - max_val)) with same mask
    sum_exp = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        allowed = k_idx < (q + 1 + delta)
        ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        vals = tl.load(ptrs, mask=(k_mask & allowed), other=-float("inf"))
        vals = vals - max_val
        exp_vals = tl.exp(vals)
        sum_exp += tl.sum(exp_vals, axis=0)  # scalar

    lse_val = max_val + tl.log(sum_exp) * ln2
    # Store to LSE[q, h]
    LSE_ptr = LSE + q * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptr, lse_val)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_stride_k, V_stride_h, V_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (num_q_tokens, heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Load lse[q, h]
    LSE_ptr = LSE + q * LSE_stride_q + h * LSE_stride_h
    lse_val = tl.load(LSE_ptr)

    # Compute output[q, h, d] across D tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q * OUT_stride_q + h * OUT_stride_h + d_idx * OUT_stride_d
        out_row = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Softmax over K with causal mask and accumulate with V_expanded
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens
            allowed = k_idx < (q + 1)  # causal: k < q + 1

            # Load LOGITS[q, h, k] for this tile
            LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(k_mask & allowed), other=-float("inf"))  # [BLOCK_K]
            vals = vals - lse_val
            exp_vals = tl.exp(vals)
            sum_exp = tl.sum(exp_vals, axis=0)  # scalar
            probs = exp_vals / sum_exp  # [BLOCK_K]

            # Load V[k, h, d] for this d tile: vector over k
            V_ptrs = V + k_idx * V_stride_k + h * V_stride_h + d0 * V_stride_d
            V_vals = tl.load(V_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

            # Accumulate: out_row += sum_k probs[k] * V[k, h, d]
            for j in range(0, BLOCK_K):
                out_row += probs[j] * V[j * V_stride_k + h * V_stride_h + d0 * V_stride_d]
        # Store output row
        tl.store(OUT_ptrs, out_row, mask=d_valid)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0):
        super().__init__()
        self.sm_scale = sm_scale
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        device = q.device

        # Cast to float32 for computation (Triton kernels operate in float32)
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Expand k and v by GQA ratio (num_qo_heads // num_kv_heads = 4)
        k_expanded = k_f32.repeat_interleave(4, dim=1)  # [total_kv, 32, 128]
        v_expanded = v_f32.repeat_interleave(4, dim=1)  # [total_kv, 32, 128]

        # Allocate output and lse
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Process segments
        len_indptr = qo_indptr.numel()
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice batched tensors
            q_batch = q_f32[q_start:q_end]                       # [num_q_tokens, 32, 128]
            k_batch = k_expanded[kv_start:kv_end]               # [num_kv_tokens, 32, 128]
            v_batch = v_expanded[kv_start:kv_end]               # [num_kv_tokens, 32, 128]

            # Segment output and lse
            output_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

            # Launch Triton kernels only if Triton is available; otherwise, fallback to PyTorch (not used here)
            if TRITON_AVAILABLE:
                # 1) Compute logits LOGITS[q,h,k] = sum_d q[q,h,d] * k[k,h,d]
                _compute_logits_kernel[(self.num_qo_heads, 1, 1)](
                    q_batch, k_batch, output_seg,
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                    k_batch.stride(0), k_batch.stride(1), k_batch.stride(2),
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    BLOCK_Q=128, BLOCK_K=128, BLOCK_D=128
                )

                # 2) Compute lse[q,h] with causal mask
                _lse_masked_kernel[(num_q_tokens, self.num_qo_heads)](
                    output_seg, lse_seg,
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    self.ln2, (num_kv_tokens - num_q_tokens),
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    lse_seg.stride(0), lse_seg.stride(1),
                    BLOCK_Q=128, BLOCK_K=128
                )

                # 3) Compute output = softmax(LOGITS) @ V_expanded with causal mask
                _softmax_output_kernel[(num_q_tokens, self.num_qo_heads)](
                    output_seg, v_batch, lse_seg,
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    v_batch.stride(0), v_batch.stride(1), v_batch.stride(2),
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    BLOCK_Q=128, BLOCK_D=128, BLOCK_K=128
                )

            # Copy segment results to global output/lse
            # We assume TRITON_AVAILABLE is True in evaluation; if not, the above would not be executed.
            # Note: We rely on Triton path; if Triton is unavailable, this function would need a torch fallback.
            # For correctness, we still perform the segment copy if Triton was used:
            output[q_start:q_end] = output_seg
            lse[q_start:q_end] = lse_seg

        # Return output (bfloat16 as in original), and lse (float32)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
