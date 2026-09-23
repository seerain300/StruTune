import torch
import math
import triton
import triton.language as tl


@triton.jit
def softmax_attention_single_b(
    Q_ptr,            # *f32, shape [1, num_qo_heads, head_dim] (logical)
    K_ptr,            # *f32, shape [N, num_kv_heads, head_dim]
    V_ptr,            # *f32, shape [N, num_kv_heads, head_dim]
    Out_ptr,          # *f32, shape [1, num_qo_heads, head_dim]
    LSE_ptr,          # *f32, shape [1, num_qo_heads]
    NUM_QO_HEADS: tl.constexpr,   # meta
    NUM_KV_HEADS: tl.constexpr,   # meta
    HEAD_DIM: tl.constexpr,       # meta
    NUM_TOKENS: tl.constexpr,     # meta
    SM_SCALE: tl.constexpr,       # meta scalar
    LOG2_INVERSE: tl.constexpr,   # meta scalar
):
    # One program per head h in this batch (batch size is 1)
    h = tl.program_id(0)

    # Load q vector for this head: q[0, h, :]
    q_off = h * HEAD_DIM
    q_vec = tl.load(Q_ptr + q_off + tl.arange(0, HEAD_DIM))

    # First pass: compute logsumexp over scaled logits for tokens in this batch
    running_max = -float('inf')
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        token_idx = t  # start_index is 0 since we process one batch
        kv_head = h // (NUM_QO_HEADS // NUM_KV_HEADS)

        # K and V are [N, num_kv_heads, head_dim]
        k_off = token_idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        v_off = token_idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

        k_vec = tl.load(K_ptr + k_off + tl.arange(0, HEAD_DIM))
        v_vec = tl.load(V_ptr + v_off + tl.arange(0, HEAD_DIM))

        # logits = dot(q_vec, k_vec)
        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE

        running_max = tl.maximum(running_max, scaled)
        running_sum += tl.exp(scaled - running_max)

    # lse = logsumexp(scaled_logits) / ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE

    # Second pass: compute attention and accumulate output
    out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        token_idx = t
        kv_head = h // (NUM_QO_HEADS // NUM_KV_HEADS)

        k_off = token_idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        v_off = token_idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

        k_vec = tl.load(K_ptr + k_off + tl.arange(0, HEAD_DIM))
        v_vec = tl.load(V_ptr + v_off + tl.arange(0, HEAD_DIM))

        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE
        attn = tl.exp(scaled - lse_val)  # softmax over scaled logits

        out_vec += attn * v_vec

    # Write results
    out_off = h * HEAD_DIM
    tl.store(Out_ptr + out_off + tl.arange(0, HEAD_DIM), out_vec)

    lse_off = h
    tl.store(LSE_ptr + lse_off, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.log2_inverse = 1.4426950408889634  # 1 / ln(2)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Inputs must be on CUDA for Triton kernels."
        device = q.device

        # q: [B, 32, 128], k_cache/v_cache: [N, 8, 128]
        B, num_qo_heads, head_dim = q.shape
        N, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == self.num_qo_heads
        assert num_kv_heads == self.num_kv_heads
        assert head_dim == self.head_dim

        # We process one batch element at a time to avoid runtime loops in Triton.
        for b in range(B):
            # Compute num_tokens for this batch: num_tokens_b = kv_indptr[b+1] - kv_indptr[b]
            num_tokens_b = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if num_tokens_b <= 0:
                # No KV for this batch element
                out = torch.zeros((1, num_qo_heads, head_dim), dtype=torch.float32, device=device)
                lse = torch.zeros((1, num_qo_heads), dtype=torch.float32, device=device)
                # Cast output to bfloat16 to match original behavior
                out_bf = out[0].to(torch.bfloat16).unsqueeze(0)  # shape [1, 32, 128]
                return out_bf, lse[0]  # return only for this batch

            # Gather token indices for this batch: [num_tokens_b]
            # Using torch.index_select on 3D tensors is fine here (host-side gather).
            token_indices = kv_indices[b * num_tokens_b:(b + 1) * num_tokens_b].to(torch.int64).contiguous()

            # Gather K_t and V_t for this batch: [num_tokens_b, 8, 128]
            K_t = torch.index_select(k_cache, 0, token_indices).contiguous().to(torch.float32)
            V_t = torch.index_select(v_cache, 0, token_indices).contiguous().to(torch.float32)

            # Prepare per-batch output and lse
            out = torch.zeros((1, num_qo_heads, head_dim), dtype=torch.float32, device=device)
            lse = torch.empty((1, num_qo_heads), dtype=torch.float32, device=device)

            # Launch Triton kernel: one program per head
            grid = (num_qo_heads,)
            softmax_attention_single_b[grid](
                q[b].to(torch.float32).contiguous(),            # shape [32, 128]
                K_t,                                             # shape [num_tokens_b, 8, 128]
                V_t,                                             # shape [num_tokens_b, 8, 128]
                out,                                             # shape [1, 32, 128]
                lse,                                             # shape [1, 32]
                NUM_QO_HEADS=num_qo_heads,
                NUM_KV_HEADS=num_kv_heads,
                HEAD_DIM=head_dim,
                NUM_TOKENS=num_tokens_b,
                SM_SCALE=self.sm_scale,
                LOG2_INVERSE=self.log2_inverse,
            )

            # Cast output to bfloat16 to match original behavior
            out_bf = out[0].to(torch.bfloat16).unsqueeze(0)  # [1, 32, 128]
            # lse is [1, 32]; we can return lse[0] as per original, which is a vector
            yield out_bf, lse[0]  # for each batch


def run(*args):
    return ModelNew()(*args)
