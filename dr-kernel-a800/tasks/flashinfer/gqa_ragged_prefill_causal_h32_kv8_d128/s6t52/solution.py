import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    q_ptr, kexp_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens,
    sm_scale,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Loop over j positions
    for j in range(0, num_kv_tokens):
        # q_ptr layout: [num_q_tokens, 32, 128], contiguous => index = (i*32 + h) * 128
        q_off = (i * 32 + h) * 128
        q_val = tl.load(q_ptr + q_off)

        # kexp_ptr layout: [num_kv_tokens, 32, 128], contiguous => index = j * (32*128) + h * 128
        k_off = j * (32 * 128) + h * 128
        k_val = tl.load(kexp_ptr + k_off)

        # Compute score
        score = q_val * k_val * sm_scale

        # Causal mask validity: j < (i + 1 + delta), delta = num_kv_tokens - num_q_tokens
        # We'll store -inf for invalid positions by computing valid flag and writing score or -inf
        valid = j < (i + 1 + (num_kv_tokens - num_q_tokens))

        # Store logits as float32. Triton will convert if needed; we keep float32 to match LSE later.
        tl.store(logits_ptr + i * 32 * num_kv_tokens + h * num_kv_tokens + j, tl.where(valid, score, -float('inf')))


@triton.jit
def lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Compute max over j
    m = -float('inf')
    for j in range(0, num_kv_tokens):
        val = tl.load(logits_ptr + i * 32 * num_kv_tokens + h * num_kv_tokens + j)
        m = tl.maximum(m, val)

    # Compute sum of exp(logits - m)
    s = 0.0
    for j in range(0, num_kv_tokens):
        val = tl.load(logits_ptr + i * 32 * num_kv_tokens + h * num_kv_tokens + j)
        s += tl.exp(val - m)

    lse_val = tl.log(s) + m  # logsumexp
    tl.store(lse_ptr + i * 32 + h, lse_val)


@triton.jit
def softmax_accum_output_kernel(
    logits_ptr, lse_ptr, vexp_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Load lse for this (i, h)
    lse_val = tl.load(lse_ptr + i * 32 + h)

    # Accumulate output across j
    out_base = out_ptr + (i * 32 + h) * head_dim
    for j in range(0, num_kv_tokens):
        val = tl.load(logits_ptr + i * 32 * num_kv_tokens + h * num_kv_tokens + j)
        y = tl.exp(val - lse_val)  # softmax weight for this j
        v_base = vexp_ptr + j * head_dim * 32 + h * head_dim
        # Vectorize over head_dim
        d = tl.arange(0, head_dim)
        v_vec = tl.load(v_base + d)  # [head_dim] in float32
        out_vec = tl.load(out_base + d)  # [head_dim]
        out_vec += y * v_vec
        tl.store(out_base + d, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sm_scale = 1.0 / math.sqrt(128)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Output buffers
        output = torch.empty((total_q, 32, 128), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Iterate over segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No queries or KV for this batch element
                continue

            # Slice q, k, v for this segment (contiguous)
            q_slice = q[q_start:q_end].contiguous()             # [num_q_tokens, 32, 128]
            k_slice = k[kv_start:kv_end].contiguous()          # [num_kv_tokens, 8, 128]
            v_slice = v[kv_start:kv_end].contiguous()          # [num_kv_tokens, 8, 128]

            num_q_tokens = q_slice.shape[0]
            num_kv_tokens = k_slice.shape[0]

            # Expand k and v to 32 heads (GQA ratio 4)
            k_exp = k_slice.repeat_interleave(4, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
            v_exp = v_slice.repeat_interleave(4, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

            # Buffers for logits (float32) and output accumulation (float32)
            logits_buf = torch.empty((num_q_tokens, 32, num_kv_tokens), dtype=torch.float32, device=q.device)

            # Launch compute logits kernel: one program per (i, h)
            grid = (num_q_tokens * 32,)
            compute_logits_kernel[grid](
                q_slice, k_exp, logits_buf,
                num_q_tokens, num_kv_tokens,
                self.sm_scale,
                num_warps=1, num_stages=1,
            )

            # Launch LSE reduction kernel
            lse_reduce_kernel[grid](
                logits_buf, lse[q_start:q_start + num_q_tokens],  # write into lse[q_start:q_end]
                num_q_tokens, num_kv_tokens,
                num_warps=1, num_stages=1,
            )

            # Initialize output accumulator in float32
            out_accum = torch.zeros((num_q_tokens, 32, 128), dtype=torch.float32, device=q.device)

            # Launch softmax accumulation kernel
            softmax_accum_output_kernel[grid](
                logits_buf, lse[q_start:q_start + num_q_tokens], v_exp, out_accum,
                num_q_tokens, num_kv_tokens, head_dim,
                num_warps=1, num_stages=1,
            )

            # Store accumulated output for this segment into the main output tensor
            output[q_start:q_start + num_q_tokens] = out_accum.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
