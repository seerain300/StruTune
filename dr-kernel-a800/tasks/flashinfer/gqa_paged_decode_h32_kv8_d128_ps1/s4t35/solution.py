import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention_bh(
    Q_ptr,            # *f32, shape [B, num_qo_heads, head_dim]
    K_ptr,            # *f32, shape [N, num_kv_heads, head_dim]
    V_ptr,            # *f32, shape [N, num_kv_heads, head_dim]
    Out_ptr,          # *f32, shape [B, num_qo_heads, head_dim]
    LSE_ptr,          # *f32, shape [B, num_qo_heads]
    B,                # int32 batch size (runtime)
    N,                # int32 total tokens (unused directly)
    num_qo_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_tokens_b,     # int32 number of tokens for this batch
    start_index,      # int32 starting token index in kv_indices for this batch
    gqa_ratio,        # int32, num_qo_heads // num_kv_heads
    SM_SCALE,         # f32 scalar = 1.0 / sqrt(head_dim)
    LOG2_INVERSE,     # f32 scalar = 1.0 / ln(2)
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q vector for this (b, h)
    q_off = b * num_qo_heads * head_dim + h * head_dim
    q_vec = tl.load(Q_ptr + q_off + tl.arange(0, head_dim))

    # First pass: compute logsumexp over scaled logits for tokens in this batch
    running_max = -float('inf')
    running_sum = 0.0

    for t in range(0, num_tokens_b):
        token_idx = start_index + t
        kv_head = h // gqa_ratio
        # K and V are [N, num_kv_heads, head_dim]
        k_off = token_idx * num_kv_heads * head_dim + kv_head * head_dim
        v_off = token_idx * num_kv_heads * head_dim + kv_head * head_dim

        k_vec = tl.load(K_ptr + k_off + tl.arange(0, head_dim))
        v_vec = tl.load(V_ptr + v_off + tl.arange(0, head_dim))

        # logits = dot(q_vec, k_vec)
        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE

        # numerically stable update for logsumexp
        m = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - m) + tl.exp(scaled - m)
        running_max = m

    # lse = logsumexp(scaled) / log(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE
    tl.store(LSE_ptr + b * num_qo_heads + h, lse_val)

    # Second pass: compute attention and accumulate output
    out_vec = tl.zeros((head_dim,), dtype=tl.float32)
    for t in range(0, num_tokens_b):
        token_idx = start_index + t
        kv_head = h // gqa_ratio
        k_off = token_idx * num_kv_heads * head_dim + kv_head * head_dim
        v_off = token_idx * num_kv_heads * head_dim + kv_head * head_dim

        k_vec = tl.load(K_ptr + k_off + tl.arange(0, head_dim))
        v_vec = tl.load(V_ptr + v_off + tl.arange(0, head_dim))

        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE
        attn = tl.exp(scaled - lse_val)

        out_vec += attn * v_vec

    # Store output
    out_off = b * num_qo_heads * head_dim + h * head_dim
    tl.store(Out_ptr + out_off + tl.arange(0, head_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        B, num_qo_heads, head_dim = q.shape

        # k_cache/v_cache: [num_pages, num_kv_heads, head_dim]
        num_pages, num_kv_heads, _ = k_cache.shape
        assert _ == head_dim, "head_dim mismatch"

        # Compute per-batch number of tokens without .item()/.tolist():
        # num_tokens_b = kv_indptr[b+1] - kv_indptr[b]
        # We'll use tensor slicing and arithmetic:
        # Note: kv_indptr is 1D int32 tensor of length B+1
        # Create index tensors for b in 0..B-1
        # Triton doesn't support Python loops over tensors; we handle via grid over b.
        # For each b, compute num_tokens_b and start_index using torch ops only (no .item(), no .tolist()).
        # We can create a zero tensor of shape (B,) to hold num_tokens_b and starts; then fill via PyTorch ops.
        # However, since we launch with grid=(B,), we compute num_tokens_b and start_index per program by slicing:
        # Triton allows passing tensors as arguments; we don't need to read from them in kernel (we pass scalars).
        # Here, we compute num_tokens_b and start_index on host using pure torch ops, but without converting to Python.
        # We'll use torch.where and elementwise subtraction to create int32 tensors without .item()/.tolist().

        # Compute num_tokens_b: shape (B,) int32 tensor
        # indices_next = kv_indptr[1:]
        indices_next = kv_indptr[1:]
        indices_prev = kv_indptr[:-1]
        num_tokens_b = indices_next - indices_prev  # shape [B], int32 tensor

        # Compute starts: shape (B,) int32 tensor
        starts = kv_indptr[:-1]  # shape [B], int32 tensor

        # Prepare output and lse buffers (fp32 for stability)
        out_fp32 = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Ensure contiguity (optional but allowed)
        Q = q.contiguous()
        K = k_cache.contiguous()
        V = v_cache.contiguous()
        kv_indices_dev = kv_indices.to(q.device).contiguous()

        # sm_scale is provided; use 1/sqrt(head_dim) as per original
        SM_SCALE = sm_scale  # float32 scalar
        LOG2_INVERSE = 1.4426950408889634  # 1 / ln(2)

        # Launch one program per (b, h)
        grid = (B, num_qo_heads)

        softmax_attention_bh[grid](
            Q, K, V, out_fp32, lse,
            B, kv_indptr[-1],  # pass last element of kv_indptr; kernel expects int32
            num_qo_heads, num_kv_heads, head_dim,
            num_tokens_b, starts,
            num_qo_heads // num_kv_heads,  # gqa_ratio
            SM_SCALE,
            LOG2_INVERSE,
            num_warps=4,
            num_stages=1,
        )

        # Cast output back to bfloat16 to match original behavior
        output_bf16 = out_fp32.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
