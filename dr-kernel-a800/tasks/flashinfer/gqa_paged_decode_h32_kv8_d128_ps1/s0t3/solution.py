import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute per-token logits for a given (b, h): logits[t] = (q[b,h,:] · k[token,:]) * sm_scale
# Grid: (num_tokens,)
# Inputs:
#   q_ptr:       *f32, (D,)
#   k_ptr:       *f32, (num_tokens, D)
#   logits_ptr:  *f32, (num_tokens,)
#   num_tokens:  i32
#   D:           i32
#   sm_scale:    f32
@triton.jit
def _compute_logits_kernel_1tok(
    q_ptr,               # *f32, (D,)
    k_ptr,               # *f32, (num_tokens, D)
    logits_ptr,          # *f32, (num_tokens,)
    num_tokens: tl.constexpr,  # i32, can be constexpr for clarity, but we pass runtime below
    D: tl.constexpr,            # i32
    sm_scale: tl.constexpr,     # f32
):
    t = tl.program_id(0)
    if t >= num_tokens:
        return
    acc = 0.0
    for i in range(0, D):
        q_i = tl.load(q_ptr + i)
        k_i = tl.load(k_ptr + t * D + i)
        acc += q_i * k_i
    tl.store(logits_ptr + t, acc * sm_scale)


# Triton kernel: compute output vector for (b, h) using softmax over scaled logits
# Grid: (num_tokens,)
# Inputs:
#   logits_ptr:    *f32, (num_tokens,)
#   v_ptr:         *f32, (num_tokens, D)
#   out_ptr:       *f32, (D,)
# Outputs:
#   out_ptr[i] = sum_t softmax(logits_scaled)[t] * v_ptr[t, i]
@triton.jit
def _compute_out_atomic_kernel(
    logits_ptr,  # *f32, (num_tokens,)
    v_ptr,       # *f32, (num_tokens, D)
    out_ptr,     # *f32, (D,)
    num_tokens: tl.constexpr,  # i32
    D: tl.constexpr,           # i32
):
    t = tl.program_id(0)
    if t >= num_tokens:
        return
    # Compute exp and sum for normalization
    sum_exp = 0.0
    for tt in range(0, num_tokens):
        sum_exp += tl.exp(logits_ptr[tt])
    x_t = tl.exp(logits_ptr[t])
    attn_t = x_t / sum_exp
    v_vec = tl.load(v_ptr + t * D + tl.arange(0, D))
    for i in range(0, D):
        tl.atomic_add(out_ptr + i, attn_t * v_vec[i])


# Triton kernel: compute lse[b, h] = logsumexp(logits_scaled) / ln(2)
# Grid: (B, Hq)
# Inputs:
#   logits_ptr:    *f32, (num_tokens,)
#   b_idx:         i32
#   h_idx:         i32
#   num_toks:      i32
#   lse_ptr:       *f32, (B * Hq,)
#   MAX_TOKS:      tl.constexpr
@triton.jit
def _lse_atomic_kernel_2d(
    logits_ptr,     # *f32, (num_tokens,)
    b_idx: tl.constexpr,  # i32
    h_idx: tl.constexpr,  # i32
    num_toks: tl.constexpr,  # i32
    lse_ptr,        # *f32, (B * Hq,)
    MAX_TOKS: tl.constexpr,
):
    max_val = -float('inf')
    sum_exp = 0.0
    t = 0
    while t < MAX_TOKS:
        if t < num_toks:
            x = tl.load(logits_ptr + t)
            if x > max_val:
                sum_exp = sum_exp * tl.exp(max_val - x) + 1.0
                max_val = x
            else:
                sum_exp += tl.exp(x - max_val)
        t += 1
    lse = max_val + tl.log(sum_exp) / tl.log(2.0)
    out_index = b_idx * Hq + h_idx
    tl.store(lse_ptr + out_index, lse)


class ModelNew(torch.nn.Module):
    def __init__(self, head_dim=128, num_qo_heads=32, num_kv_heads=8, sm_scale=1.0 / math.sqrt(128)):
        super().__init__()
        self.head_dim = head_dim
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.gqa_ratio = num_qo_heads // num_kv_heads
        self.sm_scale = float(sm_scale)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are contiguous and float32 for Triton
        device = q.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"
        q = q.contiguous().to(torch.float32)  # [B, Hq, D]
        B, Hq, D = q.shape
        assert k_cache.shape[1] == 1 and v_cache.shape[1] == 1, "Expected k_cache/v_cache second dim = 1"
        P = k_cache.shape[0]
        Hk = k_cache.shape[2]
        assert Hk == self.num_kv_heads, "num_kv_heads mismatch"

        # Build flattened K/V arrays indexed by token position for all batches
        # First determine total number of tokens across all batches
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr length must be batch_size + 1"
        num_tokens_total = int(kv_indptr[-1].item())

        # Allocate per-token K/V tensors [num_tokens_total, Hk, D]
        k_token = torch.empty((num_tokens_total, Hk, D), dtype=torch.float32, device=device)
        v_token = torch.empty((num_tokens_total, Hk, D), dtype=torch.float32, device=device)

        # Populate k_token and v_token: for each b in [0..B-1], tokens t in [kv_indptr[b], kv_indptr[b+1))
        token_idx = 0
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            for t in range(start, end):
                idx = int(kv_indices[t].item())
                assert 0 <= idx < P, f"kv_indices out of range for batch {b}, token {t}, idx={idx}, P={P}"
                k_bh = k_cache[idx, 0].contiguous().to(torch.float32)  # [Hk, D]
                v_bh = v_cache[idx, 0].contiguous().to(torch.float32)  # [Hk, D]
                k_token[token_idx] = k_bh
                v_token[token_idx] = v_bh
                token_idx += 1
        assert token_idx == num_tokens_total, "Failed to populate all tokens"

        # Prepare output and lse
        output = torch.zeros((B, Hq, D), dtype=torch.float32, device=device)  # float32 for accumulation
        lse = torch.full((B, Hq), -float("inf"), dtype=torch.float32, device=device)

        # For each (b, h), compute logits_vec for tokens, output accumulation, and lse
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start
            if num_tokens_b <= 0:
                continue

            for h in range(Hq):
                kv_head = h // self.gqa_ratio

                # q_vec for this (b,h)
                q_vec = q[b, h, :].to(torch.float32)  # [D]

                # Initialize logits vector for this (b,h)
                logits_vec = torch.empty(num_tokens_b, dtype=torch.float32, device=device)

                # Gather k and v for this batch b and head kv_head: we only need rows token_idx in [start, end)
                # k_token[start:end, kv_head, :] and v_token[start:end, kv_head, :]
                # Launch Triton kernel to compute logits for all tokens of this (b,h)
                grid = (num_tokens_b,)
                _compute_logits_kernel_1tok[grid](
                    q_vec,          # *f32, (D,)
                    k_token[start:end, kv_head, :],  # *f32, (num_tokens_b, D)
                    logits_vec,     # *f32, (num_tokens_b,)
                    num_tokens_b,   # i32
                    D,              # i32
                    self.sm_scale,  # f32
                )

                # Now compute out vector using Triton kernel
                out_vec = torch.zeros(D, dtype=torch.float32, device=device)
                _compute_out_atomic_kernel[grid](
                    logits_vec,     # *f32, (num_tokens_b,)
                    v_token[start:end, kv_head, :],  # *f32, (num_tokens_b, D)
                    out_vec,        # *f32, (D,)
                    num_tokens_b,   # i32
                    D,              # i32
                )

                # Accumulate into output[b, h, :]
                output[b, h, :] = output[b, h, :] + out_vec

                # Compute lse for (b, h) using Triton kernel
                _lse_atomic_kernel_2d[(B, Hq)](
                    logits_vec,     # *f32, (num_tokens_b,)
                    b,              # i32
                    h,              # i32
                    num_tokens_b,   # i32
                    lse,            # *f32, (B * Hq,)
                    MAX_TOKS=1024,  # compile-time bound; safe upper bound
                )

        # Cast output to bfloat16 for final return; lse remains float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# Example usage: ModelNew follows the same interface as the original Model's forward
# get_inputs() helper can be reused for testing if needed.


def run(*args):
    return ModelNew()(*args)
