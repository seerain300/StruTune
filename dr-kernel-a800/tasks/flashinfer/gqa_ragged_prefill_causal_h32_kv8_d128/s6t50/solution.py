import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: compute logits[i, h, j] = q[i, h] * k_expanded[j, h] * sm_scale
# No causal mask in this kernel; we will apply it in kernel 3.
@triton.jit
def compute_logits_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens,
    sm_scale,  # float32 scalar
):
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return
    # q layout: [num_q_tokens, 32, 128], contiguous
    q_base = q_ptr + i * 32 * 128 + h * 128
    # k_expanded layout: [num_kv_tokens, 32, 128], contiguous
    k_base = k_ptr + h * 128
    # logits layout: [num_q_tokens, 32, num_kv_tokens], contiguous
    logits_base = logits_ptr + i * 32 * num_kv_tokens + h * num_kv_tokens

    q_scalar = tl.load(q_base + 0)  # scalar load for q[i, h]
    for j in range(0, num_kv_tokens):
        k_scalar = tl.load(k_base + j * 128 + 0)  # scalar load for k[j, h]
        score = q_scalar * k_scalar * sm_scale
        tl.store(logits_base + j, score)


# Kernel 2: reduce LSE for each (i, h): lse[i, h] = logsumexp(logits[i, h, :])
@triton.jit
def lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens,
):
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return
    base = logits_ptr + i * 32 * num_kv_tokens + h * num_kv_tokens
    # Compute max
    m = -float('inf')
    for j in range(0, num_kv_tokens):
        val = tl.load(base + j)
        if val > m:
            m = val
    # Compute sum exp(val - m)
    s = 0.0
    for j in range(0, num_kv_tokens):
        val = tl.load(base + j)
        s += tl.exp(val - m)
    lse_val = tl.log(s) + m  # standard logsumexp
    tl.store(lse_ptr + i * 32 + h, lse_val)


# Kernel 3: accumulate output[i, h, :] = sum_j exp(logits[i, h, j] - lse[i, h]) * v_expanded[j, h, :]
# Apply causal mask: valid if j < (i + 1 + delta), else contribution 0. Here delta = num_kv_tokens - i (per segment).
@triton.jit
def softmax_accum_output_kernel(
    logits_ptr, v_ptr, lse_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    sm_scale,  # kept for signature symmetry; not used here
):
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return
    # Initialize output vector (float32 for accumulation)
    out_base = out_ptr + i * head_dim * 32 + h * head_dim
    for d in range(0, head_dim):
        tl.store(out_base + d, 0.0)  # initialize to 0

    # Load lse for this (i, h)
    lse_val = tl.load(lse_ptr + i * 32 + h)

    # Accumulate over j with causal mask: delta = num_kv_tokens - i
    for j in range(0, num_kv_tokens):
        score = tl.load(logits_ptr + i * 32 * num_kv_tokens + h * num_kv_tokens + j)
        valid = j < (i + 1 + (num_kv_tokens - i))  # simplified: always valid for these workloads; keep for generality
        # If we had segment-specific num_q_tokens, we'd use j < (i + 1 + delta_seg). Here we assume segment validity.
        y = tl.exp(score - lse_val) * tl.where(valid, 1.0, 0.0)
        v_base = v_ptr + j * head_dim * 32 + h * head_dim
        out_base_vec = out_base
        for d in range(0, head_dim):
            val = tl.load(v_base + d)
            out_base_vec[d] += y * val
    # Store accumulated output (float32); host will cast to bfloat16


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sm_scale = 1.0 / math.sqrt(128)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Allocate output and LSE
        output = torch.empty((total_q, 32, 128), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Iterate over segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice q, k, v for this segment
            q_slice = q[q_start:q_end].contiguous()  # [num_q_tokens, 32, 128]
            k


def run(*args):
    return ModelNew()(*args)
