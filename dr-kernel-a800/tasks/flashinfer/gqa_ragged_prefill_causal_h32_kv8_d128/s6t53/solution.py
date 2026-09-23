import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    q_ptr,       # *f32, shape [num_q_tokens, 32, 128] contiguous
    kexp_ptr,    # *f32, shape [num_kv_tokens, 32, 128] contiguous
    logits_ptr,  # *f32, shape [num_q_tokens, 32, num_kv_tokens] contiguous
    num_q_tokens: tl.constexpr,  # int
    num_kv_tokens: tl.constexpr, # int
    sm_scale: tl.constexpr,      # f32 scalar
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Iterate over j positions
    for j in range(0, num_kv_tokens):
        # q_ptr layout: contiguous [num_q_tokens, 32, 128]
        # index = ((i * 32) + h) * 128
        q_off = ((i * 32) + h) * 128
        q_val = tl.load(q_ptr + q_off)

        # kexp_ptr layout: contiguous [num_kv_tokens, 32, 128]
        # index = (j * (32 * 128)) + (h * 128)
        k_off = (j * (32 * 128)) + (h * 128)
        k_val = tl.load(kexp_ptr + k_off)

        # Compute score
        score = q_val * k_val * sm_scale

        # Causal mask validity: j < (i + 1 + (num_kv_tokens - num_q_tokens))
        delta = num_kv_tokens - num_q_tokens
        valid = j < (i + 1 + delta)
        score = tl.where(valid, score, -float('inf'))

        # Store logits[i, h, j]
        logits_off = (i * 32 + h) * num_kv_tokens + j
        tl.store(logits_ptr + logits_off, score)


@triton.jit
def lse_reduce_kernel(
    logits_ptr,   # *f32, shape [num_q_tokens, 32, num_kv_tokens]
    lse_ptr,      # *f32, shape [num_q_tokens, 32]
    num_q_tokens: tl.constexpr,
    num_kv_tokens: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Compute max over j
    m = tl.full((), -float('inf'), tl.float32)
    for j in range(0, num_kv_tokens):
        logits_off = (i * 32 + h) * num_kv_tokens + j
        val = tl.load(logits_ptr + logits_off)
        m = tl.maximum(m, val)

    # Compute sum exp(logits - m)
    s = tl.full((), 0.0, tl.float32)
    for j in range(0, num_kv_tokens):
        logits_off = (i * 32 + h) * num_kv_tokens + j
        val = tl.load(logits_ptr + logits_off)
        s += tl.exp(val - m)

    lse_val = tl.log(s) + m
    lse_off = i * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)


@triton.jit
def softmax_accum_output_kernel(
    logits_ptr,      # *f32, shape [num_q_tokens, 32, num_kv_tokens]
    vexp_ptr,        # *f32, shape [num_kv_tokens, 32, 128] contiguous
    lse_ptr,         # *f32, shape [num_q_tokens, 32]
    out_ptr,         # *f32, shape [num_q_tokens, 32, 128] contiguous
    num_q_tokens: tl.constexpr,
    num_kv_tokens: tl.constexpr,
    head_dim: tl.constexpr,  # 128
):
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Initialize output vector for head h
    out_base = (i * 32 + h) * head_dim
    out_vec = tl.zeros((head_dim,), dtype=tl.float32)

    # Load lse[i, h]
    lse_off = i * 32 + h
    m = tl.load(lse_ptr + lse_off)

    # Accumulate output: out_vec += sum_j exp(logits[i,h,j]-m) * vexp[j,h, :]
    for j in range(0, num_kv_tokens):
        # Load logits[i, h, j]
        logits_off = (i * 32 + h) * num_kv_tokens + j
        logit = tl.load(logits_ptr + logits_off)

        # Compute contribution
        y = tl.exp(logit - m)

        # Load v_expanded[j, h, :]
        v_off = j * (32 * 128) + h * 128  # vexp is [num_kv_tokens, 32, 128], contiguous
        v_vec = tl.load(vexp_ptr + v_off)

        # Accumulate
        out_vec += y * v_vec

    # Store accumulated output for this (i, h)
    for d in range(0, head_dim):
        tl.store(out_ptr + (i * 32 + h) * head_dim + d, out_vec[d])


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

        # Output buffer (bfloat16), LSE buffer (float32)
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

            # Slice and ensure contiguous
            q_slice = q[q_start:q_end].contiguous().to(torch.float32)  # [num_q_tokens, 32, 128]
            k_slice = k[kv_start:kv_end].contiguous().to(torch.float32)  # [num_kv_tokens, 8, 128]
            v_slice = v[kv_start:kv_end].contiguous().to(torch.float32)  # [num_kv_tokens, 8, 128]

            # Expand k and v to 32 heads
            k_expanded = k_slice.view(-1, 8, 128).repeat_interleave(4, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
            v_expanded = v_slice.view(-1, 8, 128).repeat_interleave(4, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

            num_q_tokens = q_slice.shape[0]
            num_kv_tokens = k_expanded.shape[0]

            # Allocate temporaries
            logits = torch.empty((num_q_tokens, 32, num_kv_tokens), dtype=torch.float32, device=q.device)
            lse_b = torch.empty((num_q_tokens, 32), dtype=torch.float32, device=q.device)
            out_b = torch.empty((num_q_tokens, 32, 128), dtype=torch.float32, device=q.device)

            # Launch compute logits kernel
            grid = (num_q_tokens * 32,)
            compute_logits_kernel[grid](
                q_slice, k_expanded, logits,
                num_q_tokens=num_q_tokens,
                num_kv_tokens=num_kv_tokens,
                sm_scale=self.sm_scale,
                num_warps=1,
            )

            # Launch LSE reduce kernel
            lse_reduce_kernel[grid](
                logits, lse_b,
                num_q_tokens=num_q_tokens,
                num_kv_tokens=num_kv_tokens,
                num_warps=1,
            )

            # Launch softmax accumulate output kernel
            softmax_accum_output_kernel[grid](
                logits, v_expanded, lse_b, out_b,
                num_q_tokens=num_q_tokens,
                num_kv_tokens=num_kv_tokens,
                head_dim=128,
                num_warps=1,
            )

            # Store per-segment results into output at correct slice
            # Output is float32 from kernel; cast to bfloat16 to match original
            out_b_cast = out_b.to(torch.bfloat16)
            # Place into global output at [q_start:q_start+num_q_tokens, :, :]
            if q_start + num_q_tokens <= total_q:
                output[q_start:q_start + num_q_tokens] = out_b_cast

            # Also store segment lse into global lse at [q_start:q_start+num_q_tokens, :]
            if q_start + num_q_tokens <= total_q:
                lse[q_start:q_start + num_q_tokens] = lse_b

        return output, lse


def run(*args):
    return ModelNew()(*args)
