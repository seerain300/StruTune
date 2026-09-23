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
        # Load q[i, h] (q_ptr layout: [num_q_tokens, 32, 128] contiguous => index = (i*32 + h) * 128)
        q_off = (i * 32 + h) * 128
        q_val = tl.load(q_ptr + q_off)

        # Load k_exp[j, h] (kexp_ptr layout: [num_kv_tokens, 32, 128] contiguous => index = j * (32*128) + h * 128)
        k_off = j * (32 * 128) + h * 128
        k_val = tl.load(kexp_ptr + k_off)

        # Compute score
        score = q_val * k_val * sm_scale

        # Causal mask: valid if j < (i + 1 + (num_kv_tokens - num_q_tokens))
        valid = j < (i + 1 + (num_kv_tokens - num_q_tokens))
        score = tl.where(valid, score, -float("inf"))

        # Store logits[i, h, j] (logits_ptr layout: [num_q_tokens, 32, num_kv_tokens] contiguous)
        logits_off = i * (32 * num_kv_tokens) + h * num_kv_tokens + j
        tl.store(logits_ptr + logits_off, score)


@triton.jit
def lse_reduce_kernel(logits_ptr, lse_ptr, num_q_tokens, num_kv_tokens):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Compute max over j
    m = -float("inf")
    for j in range(0, num_kv_tokens):
        off = i * (32 * num_kv_tokens) + h * num_kv_tokens + j
        val = tl.load(logits_ptr + off)
        m = tl.maximum(m, val)

    # Compute sum exp(logits - m)
    s = 0.0
    for j in range(0, num_kv_tokens):
        off = i * (32 * num_kv_tokens) + h * num_kv_tokens + j
        val = tl.load(logits_ptr + off)
        s += tl.exp(val - m)

    lse_val = tl.log(s) + m  # logsumexp
    lse_off = i * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)


@triton.jit
def softmax_accum_output_kernel(logits_ptr, lse_ptr, vexp_ptr, out_ptr,
                                num_q_tokens, num_kv_tokens, head_dim):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Load lse[i, h]
    lse_off = i * 32 + h
    m = tl.load(lse_ptr + lse_off)

    # Base output offset for (i, h)
    out_base = (i * 32 + h) * head_dim

    # Accumulate output over j: out[i, h, d] += exp(logits[i, h, j] - m) * v_exp[j, h, d]
    for j in range(0, num_kv_tokens):
        # Load score (softmax weight) for this j
        off = i * (32 * num_kv_tokens) + h * num_kv_tokens + j
        score = tl.load(logits_ptr + off)
        y = tl.exp(score - m)  # softmax weight

        # Load v_exp[j, h, :] vector across head_dim
        v_off = j * (32 * head_dim) + h * head_dim
        d = tl.arange(0, head_dim)
        v_vec = tl.load(vexp_ptr + v_off + d)

        # Store accumulation into out_ptr
        out_off = out_base + d
        tl.store(out_ptr + out_off, tl.load(out_ptr + out_off, mask=None, other=0.0) + y * v_vec)


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

        # Allocate outputs
        output = torch.empty((total_q, 32, 128), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Segment loop
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice and cast to float32 for computation
            q_slice = q[q_start:q_end].contiguous().to(torch.float32)  # [num_q_tokens, 32, 128]
            k_slice = k[kv_start:kv_end].contiguous().to(torch.float32)  # [num_kv_tokens, 8, 128]
            v_slice = v[kv_start:kv_end].contiguous().to(torch.float32)  # [num_kv_tokens, 8, 128]

            num_q_tokens = q_slice.shape[0]
            num_kv_tokens = k_slice.shape[0]

            # Expand k and v to 32 heads
            k_exp = k_slice.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]
            v_exp = v_slice.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate logits buffer [num_q_tokens, 32, num_kv_tokens], float32
            logits = torch.empty((num_q_tokens, 32, num_kv_tokens), dtype=torch.float32, device=q.device)

            # Launch compute_logits_kernel: one program per (i, h)
            grid = (num_q_tokens * 32,)
            compute_logits_kernel[grid](
                q_slice, k_exp, logits,
                num_q_tokens, num_kv_tokens,
                self.sm_scale,
            )

            # Compute LSE per (i, h)
            grid_lse = (num_q_tokens * 32,)
            lse_reduce_kernel[grid_lse](
                logits, lse, num_q_tokens, num_kv_tokens
            )

            # Allocate output for this segment in float32 and accumulate
            out_fp32 = torch.empty((num_q_tokens, 32, 128), dtype=torch.float32, device=q.device)

            grid_out = (num_q_tokens * 32,)
            softmax_accum_output_kernel[grid_out](
                logits, lse, v_exp, out_fp32,
                num_q_tokens, num_kv_tokens, 128
            )

            # Write to global output: slice output[q_start:q_start+num_q_tokens]
            output[q_start:q_start + num_q_tokens] = out_fp32.to(torch.bfloat16)

        # Return output (bfloat16) and lse (float32)
        return output, lse


def run(*args):
    return ModelNew()(*args)
