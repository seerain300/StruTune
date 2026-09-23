import torch
import math
import triton
import triton.language as tl


# Triton kernel: computes attention for a single (b, h) pair.
# Args:
# - Out_ptr: pointer to output[b, h, :]
# - LSE_ptr: pointer to lse[b, h]
# - Q_h_ptr: pointer to q[b, h], 1D vector of length head_dim
# - K_ptr: pointer to k_t[num_tokens, :], shape [num_tokens, head_dim], contiguous
# - V_ptr: pointer to v_t[num_tokens, :], same shape as K_ptr
# - num_tokens: int32
# - sm_scale: float32
# - head_dim: int32
# - LOG2_INVERSE: float32 (1/ln(2))
# - KV_HEAD: constexpr int (used to pick KV head mapping, not used directly since we already have Q_h_ptr for h)
@triton.jit
def softmax_and_attention_single_bh(
    Out_ptr, LSE_ptr,
    Q_h_ptr, K_ptr, V_ptr,
    num_tokens, sm_scale, head_dim,
    LOG2_INVERSE: tl.constexpr,
    KV_HEAD: tl.constexpr
):
    # First pass: compute lse = logsumexp(scaled_logits) / ln(2)
    running_max = -float('inf')
    running_sum = 0.0

    t = 0
    while t < num_tokens:
        # Load q vector for head h
        i = tl.arange(0, head_dim)
        q_vec = tl.load(Q_h_ptr + i)

        # Load k vector for token t
        k_row_ptr = K_ptr + t * head_dim
        k_vec = tl.load(k_row_ptr + i)

        # logits_t = dot(q_vec, k_vec)
        logits = tl.dot(q_vec, k_vec)  # scalar

        # scaled logits
        scaled = logits * sm_scale

        # numerically stable update
        if scaled > running_max:
            running_sum = running_sum * tl.exp(running_max - scaled) + 1.0
            running_max = scaled
        else:
            running_sum += tl.exp(scaled - running_max)

        t += 1

    # lse = log(running_sum) + running_max; divide by ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE
    tl.store(LSE_ptr, lse_val)

    # Second pass: compute output vector
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    t = 0
    while t < num_tokens:
        i = tl.arange(0, head_dim)
        q_vec = tl.load(Q_h_ptr + i)
        k_row_ptr = K_ptr + t * head_dim
        k_vec = tl.load(k_row_ptr + i)

        logits = tl.dot(q_vec, k_vec)
        scaled = logits * sm_scale
        attn = tl.exp(scaled - lse_val)  # softmax over scaled logits

        v_row_ptr = V_ptr + t * head_dim
        v_vec = tl.load(v_row_ptr + i)

        out_vec += attn * v_vec
        t += 1

    # Store output vector
    out_row_ptr = Out_ptr
    tl.store(out_row_ptr + i, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.sm_scale = 1.0 / math.sqrt(128.0)
        self.log2_inverse = 1.0 / math.log(2.0)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # q: [B, num_qo_heads, head_dim], bfloat16
        # k_cache: [num_pages, num_kv_heads, head_dim], bfloat16 (typical)
        # v_cache: [num_pages, num_kv_heads, head_dim], bfloat16
        # kv_indptr: [B+1], int32 (we'll read .item())
        # kv_indices: [num_kv_indices], int32
        # sm_scale: float32 scalar (matches original)

        B = q.shape[0]
        device = q.device

        # Output and lse tensors
        output = torch.empty((B, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)  # compute in fp32
        lse = torch.empty((B, self.num_qo_heads), dtype=torch.float32, device=device)

        # GQA ratio
        gqa_ratio = self.num_qo_heads // self.num_kv_heads

        # Precompute sm_scale (use provided sm_scale or default)
        # We'll use the provided sm_scale argument to run-time kernel. Kernel takes sm_scale as argument.
        # Build output LSE pointer layout by launching per (b, h)
        # Compute per-batch num_tokens
        # Note: kv_indptr is [B+1], int32
        # num_tokens_b = kv_indptr[b+1] - kv_indptr[b]
        # Also, ensure kv_indices is consistent with indptr (num_kv_indices == kv_indptr[-1].item())
        # We'll construct token_indices for each batch b.

        # Launch Triton kernel per (b, h)
        grid = (B, self.num_qo_heads)

        for b in range(B):
            # Compute number of tokens for this batch
            num_tokens = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if num_tokens <= 0:
                # No KV entries for this batch; output zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices for this batch: idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
            # Build idx list on host
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            token_indices = kv_indices[start:end].to(torch.int32).contiguous()

            # Gather K_t and V_t directly: k_cache[idx, kv_head, :]
            # kv_head mapping: h // gqa_ratio
            # We will compute KV_HEAD mapping per (b, h) in the kernel via constexpr (not needed now).
            # For Triton kernel, we pass K_ptr and V_ptr as [num_tokens, head_dim] contiguous.

            # For each head h, we need Q[b, h]. We'll pass Q_h_ptr per h. To do that, we can compute per h in loop below,
            # but Triton launch requires fixed args. Instead, we launch loop with Python. Triton does not support
            # nested per-head loops in launch; we'll do it in Python.

            # Compute Q_h_ptr for each head h: q[b, h] -> [head_dim] float32
            for h in range(self.num_qo_heads):
                # Q vector for this head
                q_h = q[b, h].to(torch.float32).contiguous()  # [head_dim]

                # Compute KV head mapping for GQA: kv_head = h // gqa_ratio
                kv_head = h // gqa_ratio

                # Gather K_t and V_t: [num_tokens, head_dim]
                # k_cache shape is [num_pages, num_kv_heads, head_dim]
                # idx are the token positions, kv_head selects the KV head.
                # We need k_cache[token, kv_head, :]
                K_t = k_cache[token_indices, kv_head, :].to(torch.float32).contiguous()  # [num_tokens, head_dim]
                V_t = v_cache[token_indices, kv_head, :].to(torch.float32).contiguous()  # [num_tokens, head_dim]

                # Flatten K_t and V_t to [num_tokens * head_dim] for row-major pointer arithmetic
                # Alternatively, keep as [num_tokens, head_dim] and compute row base = t * head_dim.
                # We'll keep [num_tokens, head_dim] to avoid flattening; Triton expects pointers.

                # Launch Triton kernel for this (b, h)
                # Prepare pointers:
                # Out_ptr: output[b, h, :] flattened as 1D vector length head_dim
                out_vec = torch.empty(self.head_dim, dtype=torch.float32, device=device)
                out_ptr = out_vec
                lse_ptr = lse[b, h]

                # K_ptr and V_ptr: contiguous [num_tokens, head_dim]
                K_ptr = K_t
                V_ptr = V_t

                # Q_h_ptr: q_h is 1D
                Q_h_ptr = q_h

                # Launch kernel
                softmax_and_attention_single_bh[(1,)](
                    out_ptr, lse_ptr,
                    Q_h_ptr, K_ptr, V_ptr,
                    num_tokens, sm_scale, self.head_dim,
                    LOG2_INVERSE=self.log2_inverse,
                    KV_HEAD=kv_head
                )

                # Store the computed head into output tensor
                output[b, h] = out_vec

        # Return output in bfloat16 and lse in float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
