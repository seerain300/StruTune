import torch
import math
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    Q_ptr,          # *fp32, shape [B, 32, 128]
    K_ptr,          # *fp32, shape [T, 8, 128]
    V_ptr,          # *fp32, shape [T, 8, 128]
    Out_ptr,        # *fp32, shape [B, 32, 128]
    LSE_ptr,        # *fp32, shape [B, 32]
    B: tl.int32,
    T: tl.int32,    # number of tokens for this batch element
    H: tl.int32,    # num_kv_heads = 8
    D: tl.int32,    # head_dim = 128
    gqa_ratio: tl.int32,  # num_qo_heads // num_kv_heads = 4
    sm_scale: tl.float32, # scalar, e.g., 1.0 / sqrt(128)
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2) ≈ 1.4426950408889634
    num_warps: tl.int32,
):
    # program id: one per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Guard (grid should match B and num_qo_heads, but keep for safety)
    if b >= B or h >= 32:
        return

    # q vector for this head
    q_vec = tl.load(Q_ptr + b * 32 * D + h * D + tl.arange(0, D))
    q_vec = q_vec  # fp32, 128 elements

    # First pass: compute numerically stable logsumexp over tokens
    running_max = -1.0e30
    running_sum = 0.0

    for t in range(0, T):
        # Load k_vec and v_vec for this token and kv_head = h // gqa_ratio
        kv_head = h // gqa_ratio
        k_vec = tl.load(K_ptr + t * H * D + kv_head * D + tl.arange(0, D))
        v_vec = tl.load(V_ptr + t * H * D + kv_head * D + tl.arange(0, D))

        # Dot product
        logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits * sm_scale

        # Update running max/sum for logsumexp
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(-running_max + scaled) + 1.0

    # Compute lse = log(sum(exp(scaled))) / ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE  # divide by ln(2)

    # Second pass: compute attention and accumulate output
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(0, T):
        kv_head = h // gqa_ratio
        k_vec = tl.load(K_ptr + t * H * D + kv_head * D + tl.arange(0, D))
        v_vec = tl.load(V_ptr + t * H * D + kv_head * D + tl.arange(0, D))

        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * sm_scale
        attn = tl.exp(scaled - lse_val)

        out_vec = out_vec + attn * v_vec

    # Store output vector and lse scalar
    tl.store(Out_ptr + b * 32 * D + h * D + tl.arange(0, D), out_vec)
    tl.store(LSE_ptr + b * 32 + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Assume:
        # - q: [B, 32, 128], bfloat16
        # - k_cache: [N, 8, 128], float32 (N may vary; here N is num_pages)
        # - v_cache: [N, 8, 128], float32
        # - kv_indptr: [B+1], int32
        # - kv_indices: [T], int32
        # - sm_scale: float32 scalar
        device = q.device
        B, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32
        H = 8
        D = 128
        gqa_ratio = 4  # 32 // 8

        # Prepare outputs
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Compute per-batch number of tokens
        num_tokens = []  # list of int
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens.append(end - start)
        # Build token indices for each batch element
        token_indices_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            token_indices_list.append(kv_indices[start:end].to(torch.int32).contiguous())

        # Gather K_t and V_t: [T_b, 8, 128] directly from k_cache/v_cache using token indices
        K_t = []
        V_t = []
        for b in range(B):
            # k_cache and v_cache: [N, 8, 128]; pick rows indexed by token_indices[b, :]
            # Produce [T_b, 8, 128]
            K_t.append(k_cache[token_indices_list[b]].to(torch.float32))
            V_t.append(v_cache[token_indices_list[b]].to(torch.float32))

        # Flatten K_t and V_t into [B, T, 8, 128] for simpler launch (we pass per-batch to kernel)
        # Instead, pass per-batch arrays to kernel and let it index. We’ll launch with grid (B, 32).
        # Ensure q is float32 for kernel
        Q_b_fp32 = q.to(torch.float32)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, 32)
        LOG2_INVERSE = 1.4426950408889634  # 1 / ln(2)

        softmax_and_attention_single_bh[grid](
            Q_b_fp32,             # Q_ptr
            K_t[0],               # K_ptr (we need per-batch), but Triton expects pointers; pass per-batch tensors to kernel via args.
            V_t[0],               # V_ptr
            output,               # Out_ptr
            lse,                  # LSE_ptr
            B, T=num_tokens[0], H=8, D=128,
            gqa_ratio=4,
            sm_scale=sm_scale,
            LOG2_INVERSE=LOG2_INVERSE,
            num_warps=4,
        )

        # Cast output to bfloat16 to match original; lse remains float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
