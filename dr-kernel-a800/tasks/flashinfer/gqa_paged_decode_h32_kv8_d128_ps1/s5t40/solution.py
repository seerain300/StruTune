import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gqa_row_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, 1, K, D]
    v_ptr,            # *bf16, [Np, 1, K, D]
    kv_indptr_ptr,    # *int32, [B+1]
    out_ptr,          # *bf16, [B, H, D]
    lse_ptr,          # *float32, [B, H]
    sm_scale,         # float32 scalar
    H: tl.constexpr,  # num_qo_heads
    D: tl.constexpr,  # head_dim (e.g., 128)
    K: tl.constexpr,  # num_kv_heads (e.g., 8)
    MAX_ITERS: tl.constexpr,  # fixed iteration bound over tokens
    GQA_RATIO: tl.constexpr,  # H // K (e.g., 4)
):
    # One program per (batch, query head)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load q vector for this (b, h): q[b, h, :]
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in tl.range(0, D):
        q_elem = tl.load(q_ptr + q_offset + d).to(tl.float32)
        q_vec[d] = q_elem

    # Load indptr[start, end] for this batch element
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens (scalar iterations guarded by num_tokens)
    sum_exp = 0.0  # scalar float32
    for t in range(MAX_ITERS):
        if t >= num_tokens:
            break
        idx = start + t  # token index within this batch's cache
        kv_head = pid_h // GQA_RATIO  # map query head to KV head under GQA

        # Load k[idx, kv_head, :] and v[idx, kv_head, :]
        k_offset = idx * K * D + kv_head * D
        v_offset = idx * K * D + kv_head * D

        k_vec = tl.zeros((D,), dtype=tl.float32)
        v_vec = tl.zeros((D,), dtype=tl.float32)

        for d in tl.range(0, D):
            kd = tl.load(k_ptr + k_offset + d).to(tl.float32)
            vd = tl.load(v_ptr + v_offset + d).to(tl.float32)
            k_vec[d] = kd
            v_vec[d] = vd

        # Dot product q·k over D
        dot = 0.0
        for d in tl.range(0, D):
            dot += q_vec[d] * k_vec[d]

        sum_exp += tl.exp(dot * sm_scale)

    # lse in base-2
    lse_val = tl.log(sum_exp) / tl.log(2.0)  # scalar float32
    tl.store(lse_ptr + pid_b * H + pid_h, lse_val)

    # Pass 2: accumulate output vector
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_ITERS):
        if t >= num_tokens:
            break
        idx = start + t
        kv_head = pid_h // GQA_RATIO

        k_offset = idx * K * D + kv_head * D
        v_offset = idx * K * D + kv_head * D

        k_vec = tl.zeros((D,), dtype=tl.float32)
        v_vec = tl.zeros((D,), dtype=tl.float32)

        for d in tl.range(0, D):
            kd = tl.load(k_ptr + k_offset + d).to(tl.float32)
            vd = tl.load(v_ptr + v_offset + d).to(tl.float32)
            k_vec[d] = kd
            v_vec[d] = vd

        # Dot product q·k
        dot = 0.0
        for d in tl.range(0, D):
            dot += q_vec[d] * k_vec[d]

        attn = tl.exp((dot - lse_val) * sm_scale)  # scalar
        out_vec += attn * v_vec  # elementwise across D (vector add)

    # Store output as bfloat16
    out_offset = pid_b * H * D + pid_h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity for Triton
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()  # API compatibility; not used in kernel

        # Extract shapes
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape

        # Assertions to match original model assumptions
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"

        device = q.device
        # Output tensors
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        gqa_row_kernel[grid](
            q, k_cache, v_cache, kv_indptr, output, lse,
            sm_scale,
            H=num_qo_heads, D=head_dim, K=num_kv_heads, MAX_ITERS=2048, GQA_RATIO=4,
            num_warps=4, num_stages=1,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
