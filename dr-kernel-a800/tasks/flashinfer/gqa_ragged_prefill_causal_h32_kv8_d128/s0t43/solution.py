import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_segments_kernel(
    q_ptr,  # [total_q, 32, 128], float32
    k_ptr,  # [total_kv, 8, 128], float32
    v_ptr,  # [total_kv, 8, 128], float32
    out_ptr,  # [total_q, 32, 128], float32, initialized to zeros
    lse_ptr,  # [total_q, 32], float32, initialized to -inf
    qo_indptr_ptr,  # [len_indptr], int32
    kv_indptr_ptr,  # [len_indptr], int32
    total_q, total_kv, sm_scale,
    NUM_SEGMENTS: tl.constexpr,  # len_indptr - 1
):
    """
    Triton kernel:
    - Grid axis 0: segment b in 0..NUM_SEGMENTS-1
    - Grid axis 1: query position i in 0..Nq-1 (computed implicitly via total_q)
    - For each (b, i), compute attention for all q heads h in 0..31:
      - Load q[i, h, :], compute logits across 8 kv heads (j=0..7) as dot products, apply mask using delta = Nk - Nq for this segment.
      - Compute base-2 logsumexp over the 8 logits.
      - Compute softmax across the 8 positions.
      - Accumulate output[i, h, :] += soft[j] * v[kv_start + j, (h % 8), :] for j in 0..7
      - Accumulate lse[i, h] += lse_val (atomic add since we may run multiple segments)
    """
    b = tl.program_id(axis=0)
    # We need to iterate i over all query positions in this segment. Since Triton doesn't support dynamic loops,
    # we instead compute i via total_q grid and rely on the host to launch enough programs; but to keep it simple,
    # we let the host pass a combined grid of (NUM_SEGMENTS, total_q) and assume axis1 = i. Here, axis1 is i.
    # So we derive i from program_id(axis=1). But Triton doesn't let us read axis1? Fix: we launch with grid=(NUM_SEGMENTS, total_q),
    # and Triton will pass axis1=i. We can then use it. However, to be explicit, we'll use a static loop over i; Triton doesn't allow that.
    # Instead, we rely on the host launching exactly (NUM_SEGMENTS, total_q) programs, where each program handles one (b, i).
    # Triton will map program_id(axis=0)=b, program_id(axis=1)=i.

    # Since Triton doesn't expose axis mapping easily, we structure the kernel to handle one (b, i) per program by launching
    # grid=(NUM_SEGMENTS, total_q) and then use program_id(axis=0) and program_id(axis=1) as b and i respectively.

    # Get b and i from program_id. Triton uses program_id(axis=0) and program_id(axis=1). We can't query axis names,
    # but in the launch we set grid=(NUM_SEGMENTS, total_q), so b=pid0, i=pid1.
    b = tl.program_id(axis=0)
    i = tl.program_id(axis=1)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b)          # int32
    q_end = tl.load(qo_indptr_ptr + b + 1)       # int32
    kv_start = tl.load(kv_indptr_ptr + b)        # int32
    kv_end = tl.load(kv_indptr_ptr + b + 1)      # int32

    Nq = q_end - q_start
    Nk = kv_end - kv_start

    # If segment is empty, return
    if Nq <= 0 or Nk <= 0:
        return

    delta = Nk - Nq

    # Process each query head h
    for h in tl.static_range(32):
        # Skip if i >= Nq (shouldn't happen if we launch exactly total_q programs)
        if i >= Nq:
            return

        # Load q[i, h, :]
        q_row_base = (q_start + i) * 32 * 128 + h * 128
        q_vec = tl.load(q_ptr + q_row_base)  # [128], float32

        # Compute logits for j in 0..7 using mapping orig_h = h % 8
        logits = tl.zeros((8,), dtype=tl.float32)
        for j in tl.static_range(8):
            orig_h = h % 8
            k_base = (kv_start + j) * 8 * 128 + orig_h * 128  # k has 8 heads
            k_vec = tl.load(k_ptr + k_base)  # [128]
            dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
            logits[j] = dot
            # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
            if j >= (i + 1 + delta):
                logits[j] = -float("inf")

        # Base-2 logsumexp over 8 positions
        m = tl.max(logits, axis=0)
        sum_exp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)
        # Softmax across 8 positions
        soft = tl.exp(logits - lse_val)  # [8], float32

        # Accumulate output[i, h, :] += soft[j] * v[kv_start + j, orig_h, :] for j in 0..7
        out_row_base = (q_start + i) * 32 * 128 + h * 128
        for j in tl.static_range(8):
            orig_h = h % 8
            v_base = (kv_start + j) * 8 * 128 + orig_h * 128
            v_vec = tl.load(v_ptr + v_base)  # [128]
            # Atomic add to accumulate across segments
            tl.atomic_add(out_ptr + out_row_base, v_vec * soft[j])

        # Accumulate lse[i, h]
        lse_index = (q_start + i) * 32 + h
        tl.atomic_add(lse_ptr + lse_index, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized forward:
        - q: [total_q, 32, 128], bfloat16
        - k: [total_kv, 8, 128], bfloat16
        - v: [total_kv, 8, 128], bfloat16
        - qo_indptr: int32 [len_indptr]
        - kv_indptr: int32 [len_indptr]
        - sm_scale: float32 (e.g., 1/sqrt(128))
        Returns:
        - output: [total_q, 32, 128], float32
        - lse: [total_q, 32], float32 (base-2 logsumexp)
        """
        # Assumptions and checks
        assert q.shape[1] == 32 and k.shape[1] == 8 and v.shape[1] == 8
        assert q.shape[2] == 128 and k.shape[2] == 128 and v.shape[2] == 128
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.numel()
        # The original asserts in the provided PyTorch code:
        # assert num_qo_heads == 32; assert num_kv_heads == 8; assert head_dim == 128

        # Convert inputs to float32 for kernel
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()
        qo_indptr_f32 = qo_indptr.to(torch.int32)
        kv_indptr_f32 = kv_indptr.to(torch.int32)

        # Allocate outputs (float32, as we compute in Triton)
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: grid over segments and query positions
        NUM_SEGMENTS = len_indptr - 1
        grid = (NUM_SEGMENTS, total_q)
        attention_gqa_segments_kernel[grid](
            q_f32, k_f32, v_f32, output, lse, qo_indptr_f32, kv_indptr_f32,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=NUM_SEGMENTS,
            num_warps=4,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
