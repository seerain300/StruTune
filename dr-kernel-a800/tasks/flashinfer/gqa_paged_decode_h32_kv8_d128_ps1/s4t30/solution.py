import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    out_ptr,       # *fp32, shape [B, 32, 128] contiguous
    lse_ptr,       # *fp32, shape [B, 32] contiguous
    K_ptr,         # *fp32, shape [NUM_TOKENS_B, 128] contiguous
    V_ptr,         # *fp32, shape [NUM_TOKENS_B, 128] contiguous
    Q_ptr,         # *fp32, shape [128] contiguous
    sm_scale,      # fp32 scalar
    NUM_TOKENS_B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2_INV: tl.constexpr,  # 1 / ln(2) ≈ 1.4426950408889634
):
    # program ids
    b = tl.program_id(0)  # not used directly, but lse_ptr/out_ptr layout uses b and h
    h = tl.program_id(1)

    # 1) Compute lse = logsumexp(scaled_logits) / ln(2) over all tokens for this (b, h)
    running_max = -float('inf')
    running_sum = 0.0

    for t in range(NUM_TOKENS_B):
        q = tl.load(Q_ptr + tl.arange(0, HEAD_DIM))  # [128]
        k = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [128]
        # logits = dot(q, k)
        logits = tl.sum(q * k, axis=0)  # scalar
        scaled = logits * sm_scale
        # numerically stable update
        running_max_new = tl.maximum(running_max, scaled)
        exp_term = tl.exp(scaled - running_max)
        running_sum = running_sum * tl.exp(running_max - running_max_new) + exp_term
        running_max = running_max_new

    # lse = log(running_sum) + running_max; divide by ln(2)
    lse = tl.log(running_sum) + running_max
    lse = lse * LOG2_INV

    # store lse at [b, h]
    tl.store(lse_ptr + b * 32 + h, lse)

    # 2) Accumulate output vector for this head: out[b, h, :] = sum_j attn_j * V[j, :]
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for t in range(NUM_TOKENS_B):
        q = tl.load(Q_ptr + tl.arange(0, HEAD_DIM))  # [128]
        k = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [128]
        logits = tl.sum(q * k, axis=0)  # scalar
        scaled = logits * sm_scale
        attn = tl.exp(scaled - lse)  # scalar
        v = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [128]
        out_vec += attn * v

    # store out_vec to out_ptr[b, h, :]
    out_offset = b * 32 * HEAD_DIM + h * HEAD_DIM
    tl.store(out_ptr + out_offset, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sm_scale = 1.0 / math.sqrt(128.0)  # 1/sqrt(128)
        self.LOG2_INV = 1.4426950408889634  # 1 / ln(2)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, 32, 128], bfloat16, CUDA
        k_cache: [N, 8, 128], bfloat16, CUDA
        v_cache: [N, 8, 128], bfloat16, CUDA
        kv_indptr: [B+1], int32, CUDA
        kv_indices: [num_tokens], int32, CUDA
        sm_scale: float (ignored; we use self.sm_scale)
        Returns: (output [B, 32, 128] bfloat16), (lse [B, 32] float32)
        """
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA device for Triton."
        assert q.shape[1] == 32 and q.shape[2] == 128, "q must be [B, 32, 128]"
        assert k_cache.shape[1] == 8 and k_cache.shape[2] == 128, "k_cache must be [N, 8, 128]"
        assert v_cache.shape[1] == 8 and v_cache.shape[2] == 128, "v_cache must be [N, 8, 128]"
        B = q.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        device = q.device
        # Output and lse buffers in fp32 for numerical stability
        out_ptr = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_ptr = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Launch one program per (b, h)
        grid = (B, num_qo_heads)

        for b in range(B):
            # Compute num_tokens_b on host (no torch reductions inside kernel)
            num_tokens_b = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if num_tokens_b <= 0:
                out_ptr[b].zero_()
                lse_ptr[b].zero_()
                continue

            # Gather token indices for this batch
            token_start = int(kv_indptr[b].item())
            token_end = int(kv_indptr[b + 1].item())
            indices = kv_indices[token_start:token_end].to(torch.int64).contiguous()  # [num_tokens_b]

            # Prepare K_t and V_t for each head h: k_cache[v_cache][:, kv_head, :] at these indices
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # GQA mapping

                # Gather k_cache and v_cache for this batch b, these indices, and this kv_head
                # k_cache[indices, kv_head, :] -> [num_tokens_b, 128]
                # v_cache[indices, kv_head, :] -> [num_tokens_b, 128]
                K_t = k_cache.index_select(0, indices)[:, kv_head, :].to(torch.float32).contiguous()  # [num_tokens_b, 128]
                V_t = v_cache.index_select(0, indices)[:, kv_head, :].to(torch.float32).contiguous()  # [num_tokens_b, 128]

                # Load q vector for this head
                Q_vec = q[b, h, :].to(torch.float32).contiguous()  # [128]

                # Launch kernel for (b, h)
                softmax_and_attention_single_bh[grid](
                    out_ptr, lse_ptr,
                    K_t, V_t,
                    Q_vec,
                    self.sm_scale,
                    NUM_TOKENS_B=num_tokens_b,
                    HEAD_DIM=head_dim,
                    LOG2_INV=self.LOG2_INV,
                    num_warps=4,
                )

        # Cast output to bfloat16 as in original; lse is float32
        output = out_ptr.to(torch.bfloat16)
        lse = lse_ptr
        return output, lse


def run(*args):
    return ModelNew()(*args)
