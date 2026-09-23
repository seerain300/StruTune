import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def fused_gqa_row_simple_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [num_pages, 1, K, D]
    v_ptr,            # *bf16, [num_pages, 1, K, D]
    indptr_ptr,       # *int32, [B+1]
    indices_ptr,      # *int32, [num_tokens_total]
    out_ptr,          # *bf16, [B, H, D]
    lse_ptr,          # *fp32,  [B, H]
    sm_scale: tl.float32,
    B: tl.constexpr,     # batch size (kept for completeness)
    H: tl.constexpr,     # 32
    D: tl.constexpr,     # 128
    K: tl.constexpr,     # 8
    gqa_ratio: tl.constexpr,  # H // K == 4
    MAX_TOKENS: tl.constexpr, # e.g., 1024
):
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # query head index

    # Load q vector for this (b, h) in fp32: q[b, h, :]
    q_offset = pid_b * (H * D) + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # shape [D]

    # Load indptr[start, end] for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start  # number of tokens in cache for this batch element

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens
    sum_exp = 0.0
    for t in range(0, MAX_TOKENS):
        if t < num_tokens:
            idx = tl.load(indices_ptr + start + t).to(tl.int32)  # scalar token index
            kv_head = pid_h // gqa_ratio  # 0..7
            # Load k row and v row for this token and kv_head (1D vectors of length D)
            k_row_ptr = k_ptr + idx * (K * D) + kv_head * D
            v_row_ptr = v_ptr + idx * (K * D) + kv_head * D
            # Use a constexpr loop over D to avoid tl.arange on dynamic sizes
            d = 0
            k_row = 0.0
            while d < D:
                kd = tl.load(k_row_ptr + d).to(tl.float32)
                vd = tl.load(v_row_ptr + d).to(tl.float32)
                # accumulate k_row and v_row if needed (we only need k_row for dot)
                k_row += kd
                d += 1
            # Dot product over D using the pre-summed k_row is incorrect; instead, compute dot per element
            # Fix: compute dot via elementwise loads and sum
            dot = 0.0
            d = 0
            while d < D:
                kd = tl.load(k_row_ptr + d).to(tl.float32)
                # q_vec[d] access: load scalar
                qd = tl.load(q_ptr + q_offset + d).to(tl.float32)
                dot += qd * kd
                d += 1
            sum_exp += tl.exp(dot * sm_scale)
        else:
            break

    # Compute lse in base-2
    ln2 = 0.6931471805599453  # natural log of 2
    lse = tl.log(sum_exp) / ln2  # fp32

    # Pass 2: compute output = sum_t attn_t * v_row_t, where attn_t = exp((dot - lse) * sm_scale)
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(0, MAX_TOKENS):
        if t < num_tokens:
            idx = tl.load(indices_ptr + start + t).to(tl.int32)
            kv_head = pid_h // gqa_ratio
            k_row_ptr = k_ptr + idx * (K * D) + kv_head * D
            v_row_ptr = v_ptr + idx * (K * D) + kv_head * D
            dot = 0.0
            d = 0
            while d < D:
                kd = tl.load(k_row_ptr + d).to(tl.float32)
                qd = tl.load(q_ptr + q_offset + d).to(tl.float32)
                dot += qd * kd
                d += 1
            attn = tl.exp((dot - lse) * sm_scale)  # fp32
            d = 0
            v_row = 0.0
            while d < D:
                vd = tl.load(v_row_ptr + d).to(tl.float32)
                out_vec[d] += attn * vd
                d += 1
        else:
            break

    # Store output as bfloat16 and lse as fp32
    out_offset = pid_b * (H * D) + pid_h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))
    tl.store(lse_ptr + pid_b * H + pid_h, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Basic assertions matching original
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"
        assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be batch_size + 1"
        assert kv_indices.shape[0] == kv_indptr[-1].item(), "kv_indices length must equal last indptr"

        device = q.device

        # Allocate outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        fused_gqa_row_simple_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=(num_qo_heads // num_kv_heads), MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
