import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def fused_gqa_row_scalar_kernel(
    q_ptr, k_ptr, v_ptr, indptr_ptr,  # input pointers
    out_ptr, lse_ptr,                 # output pointers
    sm_scale,                         # float32 scalar
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    # program ids for (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # GQA mapping: kv_head = h // 4
    kv_head = pid_h // 4

    # Load indptr[start, end] for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start  # runtime int32 scalar

    # Load q vector for this head (b, h) as fp32: q[b, h, :]
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D)).to(tl.float32)  # [D], vectorized only over constexpr D

    # Accumulate sum_exp in base-2 for logsumexp
    sum_exp = 0.0
    LN2 = 0.6931471805599453  # ln(2)
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t

        # Base offsets for k[idx, kv_head, :] and v[idx, kv_head, :]
        # k_ptr layout: [Np, 1, K, D], so k_base = idx * (K * D) + kv_head * D
        k_base = idx * (K * D) + kv_head * D
        v_base = idx * (K * D) + kv_head * D

        # Compute dot = sum(q_vec * k_row) over D using scalar steps
        dot = 0.0
        for d in range(0, D):
            kd = tl.load(k_ptr + k_base + d).to(tl.float32)  # scalar load
            qd = q_vec[d]
            dot += qd * kd

        # Accumulate sum_exp in base-2
        sum_exp += tl.exp(dot * sm_scale * LN2)

    # Compute lse (base-2) as log(sum_exp) / LN2
    lse = tl.log(sum_exp) / LN2

    # Prepare output vector as fp32 and compute attn-based accumulation
    out_vec = tl.zeros((D,), dtype=tl.float32)

    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t
        k_base = idx * (K * D) + kv_head * D
        v_base = idx * (K * D) + kv_head * D

        dot = 0.0
        for d in range(0, D):
            kd = tl.load(k_ptr + k_base + d).to(tl.float32)
            qd = q_vec[d]
            dot += qd * kd

        attn = tl.exp((dot - lse) * sm_scale)  # softmax scaling

        for d in range(0, D):
            vd = tl.load(v_ptr + v_base + d).to(tl.float32)
            out_vec[d] += attn * vd

    # Store output vector (fp32) to out_ptr at [b, h, :]
    out_offset = pid_b * H * D + pid_h * D
    tl.store(out_ptr + out_offset, out_vec)

    # Store lse (base-2) to lse_ptr at [b, h]
    lse_offset = pid_b * H + pid_h
    tl.store(lse_ptr + lse_offset, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity for simple indexing
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Extract shapes
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device

        # Allocate fp32 output for Triton and lse tensor
        output_fp32 = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        MAX_TOKENS = 1024  # upper bound; guarded by num_tokens

        fused_gqa_row_scalar_kernel[grid](
            q, k_cache, v_cache, kv_indptr,
            output_fp32, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, MAX_TOKENS=MAX_TOKENS,
            num_warps=1, num_stages=1,
        )

        # Convert output to bfloat16 as in original run
        output = output_fp32.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
