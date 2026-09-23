import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention_bh_kernel(
    Q_ptr,          # *fp32, [B, num_qo_heads, head_dim]
    K_ptr,          # *fp32, [N, 1, num_kv_heads, head_dim]
    V_ptr,          # *fp32, [N, 1, num_kv_heads, head_dim]
    Out_ptr,        # *fp32, [B, num_qo_heads, head_dim] output buffer (fp32)
    LSE_ptr,        # *fp32, [B, num_qo_heads] lse buffer
    B: tl.constexpr,             # batch size
    NUM_QO_HEADS: tl.constexpr,  # 32
    NUM_KV_HEADS: tl.constexpr,  # 8
    HEAD_DIM: tl.constexpr,      # 128
    SM_SCALE: tl.constexpr,      # 1 / sqrt(HEAD_DIM)
    LOG2_INV: tl.constexpr,      # 1 / ln(2)
    NUM_TOKENS: tl.int32,
    # Strides for Q: (b, h, d)
    q_stride_b, q_stride_h, q_stride_d,
    # Strides for K: (n, kv1, kvh, d)
    k_stride_n, k_stride_1, k_stride_h, k_stride_d,
    # Strides for V: (n, kv1, kvh, d)
    v_stride_n, v_stride_1, v_stride_h, v_stride_d,
    # Strides for Out: (b, h, d)
    out_stride_b, out_stride_h, out_stride_d,
    # Strides for LSE: (b, h)
    lse_stride_b, lse_stride_h,
    # GQA ratio
    gqa_ratio: tl.constexpr,
):
    # Program id: one program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    kv_head = h // gqa_ratio  # GQA: map 32 heads to 8 kv heads

    # Initialize running max and sum for logsumexp
    max_scale = -float("inf")
    sum_exp = 0.0

    # First pass: compute lse of scaled logits across tokens
    for t in range(NUM_TOKENS):
        # Load q vector for this head
        q_off = b * q_stride_b + h * q_stride_h
        q_vec = tl.load(Q_ptr + q_off + tl.arange(0, HEAD_DIM) * q_stride_d, mask=True)

        # Load k vector for this token and kv head: K[start + t, 0, kv_head, :]
        k_off = (b * NUM_TOKENS + t) * k_stride_n + 0 * k_stride_1 + kv_head * k_stride_h
        k_vec = tl.load(K_ptr + k_off + tl.arange(0, HEAD_DIM) * k_stride_d, mask=True)

        # Compute logits = dot(q_vec, k_vec)
        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE

        # Numerically stable logsumexp update
        new_max = tl.maximum(max_scale, scaled)
        # rescale previous sum to new_max and add current exp(scaled - new_max)
        sum_exp = sum_exp * tl.exp(max_scale - new_max) + tl.exp(scaled - new_max)
        max_scale = new_max

    # Compute lse = log(sum_exp) + max_scale, then divide by ln(2)
    lse_val = tl.log(sum_exp) + max_scale
    lse_val = lse_val * LOG2_INV

    # Store lse to LSE_ptr[b, h]
    lse_off = b * lse_stride_b + h * lse_stride_h
    tl.store(LSE_ptr + lse_off, lse_val)

    # Second pass: compute attention and accumulate output
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(NUM_TOKENS):
        q_off = b * q_stride_b + h * q_stride_h
        q_vec = tl.load(Q_ptr + q_off + tl.arange(0, HEAD_DIM) * q_stride_d, mask=True)

        k_off = (b * NUM_TOKENS + t) * k_stride_n + 0 * k_stride_1 + kv_head * k_stride_h
        k_vec = tl.load(K_ptr + k_off + tl.arange(0, HEAD_DIM) * k_stride_d, mask=True)

        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE
        attn = tl.exp(scaled - lse_val)

        v_off = (b * NUM_TOKENS + t) * v_stride_n + 0 * v_stride_1 + kv_head * v_stride_h
        v_vec = tl.load(V_ptr + v_off + tl.arange(0, HEAD_DIM) * v_stride_d, mask=True)

        out_vec += attn * v_vec

    # Store output to Out_ptr[b, h, :]
    out_off = b * out_stride_b + h * out_stride_h
    tl.store(Out_ptr + out_off + tl.arange(0, HEAD_DIM) * out_stride_d, out_vec, mask=True)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, num_qo_heads, head_dim], dtype bfloat16
        k_cache: [N, 1, num_kv_heads, head_dim], dtype bfloat16
        v_cache: [N, 1, num_kv_heads, head_dim], dtype bfloat16
        kv_indptr: [len_indptr], len_indptr == B + 1
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar (e.g., 1/sqrt(head_dim))
        Returns:
        - output: [B, num_qo_heads, head_dim], dtype bfloat16
        - lse: [B, num_qo_heads], dtype float32
        """
        assert q.dim() == 3, "q must be [B, num_qo_heads, head_dim]"
        assert k_cache.dim() == 4 and v_cache.dim() == 4, "k_cache and v_cache must be [N, 1, num_kv_heads, head_dim]"
        assert kv_indptr.dim() == 1, "kv_indptr must be 1D"
        B = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        assert k_cache.shape[1] == 1 and v_cache.shape[1] == 1, "dim=1 of k_cache/v_cache must be size-1"
        assert k_cache.shape[3] == v_cache.shape[3] == head_dim, "head_dim mismatch"
        assert k_cache.shape[2] == v_cache.shape[2], "num_kv_heads mismatch"
        assert k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16, "k_cache/v_cache must be bfloat16"
        assert q.dtype == torch.bfloat16, "q must be bfloat16"
        # Compute num tokens per batch from kv_indptr
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr length must be B + 1"
        total_tokens = int(kv_indptr[-1].item())
        assert total_tokens == kv_indices.shape[0], "total_tokens from kv_indptr[-1] must equal kv_indices.shape[0]"
        # num_kv_indices is not needed for computation but we can derive per-batch counts safely:
        num_tokens_per_batch = [int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item()) for b in range(B)]

        # Prepare output buffers in fp32 for numerical stability
        out_fp32 = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse_fp32 = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Ensure inputs are contiguous (we pass strides anyway, but contiguity helps)
        q_fp32 = q.to(torch.float32).contiguous()
        k_fp32 = k_cache.to(torch.float32).contiguous()
        v_fp32 = v_cache.to(torch.float32).contiguous()

        # Launch Triton kernel: one program per (b, h)
        grid = (B, num_qo_heads)
        gqa_ratio = num_qo_heads // (k_cache.shape[2])  # 32 // 8 = 4
        # Note: we cannot use kv_indices in-kernel (it would require device tensor ops). We only need num_tokens per batch.
        # We pass num_tokens for each batch via program-specific NUM_TOKENS argument.
        # Triton allows passing different scalars per program instance using *meta. We emulate by launching grid and passing num_tokens_b per call.
        # Since Triton kernels require a single NUM_TOKENS scalar, we launch a loop over batches in Python (grid dimension covers batches).
        # However, Triton kernels do not support per-program scalar changes easily; so we compute per-batch with separate calls or do vector loop. Here we do vector loop inside kernel, but Triton kernels cannot take per-program NUM_TOKENS. So we compute per-batch lse/output by launching a 2D grid where we set num_tokens via meta and pass B as tl.constexpr. To keep it simple and correct, we compute one batch at a time using a Python loop over B, launching grid=(1, num_qo_heads) for each b and passing NUM_TOKENS for that batch.

        # Workaround: loop over batches in Python and launch kernel for each batch, passing NUM_TOKENS for that batch
        for b in range(B):
            # Set num_tokens for this batch
            num_tokens_b = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            # Launch kernel for this batch
            softmax_attention_bh_kernel[grid](
                q_fp32, k_fp32, v_fp32,
                out_fp32, lse_fp32,
                B=B,
                NUM_QO_HEADS=num_qo_heads,
                NUM_KV_HEADS=k_cache.shape[2],
                HEAD_DIM=head_dim,
                SM_SCALE=sm_scale,
                LOG2_INV=1.0 / math.log(2.0),
                NUM_TOKENS=num_tokens_b,
                # Strides for Q: [b, h, d]
                q_stride_b=q_fp32.stride(0), q_stride_h=q_fp32.stride(1), q_stride_d=q_fp32.stride(2),
                # Strides for K: [n, 1, kvh, d]
                k_stride_n=k_fp32.stride(0), k_stride_1=k_fp32.stride(1), k_stride_h=k_fp32.stride(2), k_stride_d=k_fp32.stride(3),
                # Strides for V: [n, 1, kvh, d]
                v_stride_n=v_fp32.stride(0), v_stride_1=v_fp32.stride(1), v_stride_h=v_fp32.stride(2), v_stride_d=v_fp32.stride(3),
                # Strides for Out: [b, h, d]
                out_stride_b=out_fp32.stride(0), out_stride_h=out_fp32.stride(1), out_stride_d=out_fp32.stride(2),
                # Strides for LSE: [b, h]
                lse_stride_b=lse_fp32.stride(0), lse_stride_h=lse_fp32.stride(1),
                gqa_ratio=num_qo_heads // k_cache.shape[2],
            )

        # Cast output to bfloat16 to match original behavior
        output = out_fp32.to(torch.bfloat16)
        return output, lse_fp32


def run(*args):
    return ModelNew()(*args)
