import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute per-token logits for a given (b, h)
# Grid: (num_tokens,)
# Inputs:
#   q_ptr:       *f32, [D] pointer to q[b, h, :]
#   k_ptr:       *f32, [num_tokens, D] pointer to K gathered per token
#   logits_ptr:  *f32, [num_tokens] output buffer
#   num_tokens:  i32 (runtime)
#   D:           i32 (runtime)
#   sm_scale:    f32
@triton.jit
def _compute_logits_kernel_2d(
    q_ptr,               # *f32, [D]
    k_ptr,               # *f32, [num_tokens, D]
    logits_ptr,          # *f32, [num_tokens]
    num_tokens,          # i32
    D,                   # i32
    sm_scale,            # f32
):
    t = tl.program_id(0)
    if t >= num_tokens:
        return
    acc = 0.0
    i = 0
    while i < D:
        idx = i + tl.arange(0, 1)  # scalar offset
        q_vec = tl.load(q_ptr + idx)  # idx is scalar; Triton supports scalar indexing
        k_vec = tl.load(k_ptr + t * D + idx)
        acc += q_vec[0] * k_vec[0]
        i += 1
    tl.store(logits_ptr + t, acc * sm_scale)


# Triton kernel: compute output vector for (b, h) using softmax over scaled logits
# Grid: (num_tokens,)
# Inputs:
#   logits_ptr:    *f32, [num_tokens]
#   v_ptr:         *f32, [num_tokens, D]
#   out_ptr:       *f32, [D]
# Outputs:
#   out_ptr[i] = sum_t softmax(logits_scaled)[t] * v_ptr[t, i]
@triton.jit
def _compute_out_atomic_kernel_2d(
    logits_ptr,  # *f32, [num_tokens]
    v_ptr,       # *f32, [num_tokens, D]
    out_ptr,     # *f32, [D]
    num_tokens,  # i32
    D: tl.constexpr,  # compile-time constant head dim
):
    t = tl.program_id(0)
    if t >= num_tokens:
        return
    x_t = tl.exp(tl.load(logits_ptr + t))
    sum_exp = 0.0
    tt = 0
    while tt < num_tokens:
        sum_exp += tl.exp(tl.load(logits_ptr + tt))
        tt += 1
    attn_t = x_t / sum_exp
    v_vec = tl.load(v_ptr + t * D + tl.arange(0, D))
    for i in range(0, D):
        tl.atomic_add(out_ptr + i, attn_t * v_vec[i])


# Triton kernel: compute lse[b, h] = logsumexp(logits_scaled) / ln(2)
# Grid: (B, Hq)
# Inputs:
#   logits_ptr:    *f32, [num_tokens] per (b,h)
#   b_idx:         i32
#   h_idx:         i32
#   num_tokens:    i32
#   lse_ptr:       *f32, [B * Hq] to write lse
#   MAX_TOKS:      tl.constexpr (compile-time bound)
@triton.jit
def _lse_atomic_kernel_2d(
    logits_ptr,  # *f32, [num_tokens]
    b_idx,       # i32
    h_idx,       # i32
    num_tokens,  # i32
    lse_ptr,     # *f32, [B * Hq]
    MAX_TOKS: tl.constexpr,  # e.g., 1024
):
    max_val = -float('inf')
    sum_exp = 0.0
    t = 0
    while t < MAX_TOKS:
        if t < num_tokens:
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
        # We keep head_dim as constexpr for Triton kernels
        self.D = head_dim

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # q: [B, Hq, D], k_cache, v_cache: [P, 1, Hk, D]
        B, Hq, D = q.shape
        Hk = v_cache.shape[2]
        assert Hq % Hk == 0, "num_qo_heads must be divisible by num_kv_heads"
        gqa_ratio = Hq // Hk

        # Flatten kv_indptr and kv_indices for arbitrary inputs (no fixed assumptions)
        # We compute per-batch token ranges using kv_indptr and kv_indices (as in original code).
        # Device must be CUDA; Triton kernels require CUDA tensors.
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Tensors must be on CUDA for Triton"

        device = q.device
        output = torch.zeros((B, Hq, D), dtype=torch.float32, device=device)  # compute in f32, cast at end
        lse = torch.full((B, Hq), -float("inf"), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            # Determine token range for this batch
            page_start = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            num_tokens = int((page_end - page_start) * 1)  # for num_pages=1, each element has its own tokens
            if num_tokens == 0:
                # No KV tokens for this batch element; output zero and lse stays -inf
                lse[b] = torch.logsumexp(torch.tensor([], device=device, dtype=torch.float32))  # dummy to keep shape, but we won't use it as num_tokens=0
                continue

            token_indices = kv_indices[page_start:page_end].to(torch.int32)
            # Gather K and V per token (GQA mapping)
            # k_cache shape [P, 1, Hk, D] => [P, Hk, D]; we select kv_head = h // gqa_ratio
            kv_head = 0  # always valid for GQA when num_tokens>0; but better compute per token robustly
            # We need to map query head h to kv head; original code uses h // gqa_ratio per token per head.
            # Since we are doing one token at a time, compute kv_head = (h // gqa_ratio) but here we have single head mapping per token.
            # However, for each t, kv_head depends on token index. Since token_indices are identical for all h, kv_head is same for all t.
            # We will compute kv_head as h // gqa_ratio for a representative head; but we need it per token. To keep it general, we'll use kv_indices mapping via k_cache/v_cache strides:
            # k_cache is [P, 1, Hk, D]; for each t, we need k[token, kv_head, :].
            # We don't have per-token kv_head; but original code uses h // gqa_ratio and k_cache is indexed by token, and kv_indices is not used for selecting kv_head in the loop — the loop uses kv_indptr ranges and kv_indices to gather tokens, and then selects kv head based on h // gqa_ratio from q's head. Since q has fixed Hq and Hk, kv_head is constant across tokens for each head h.
            # Therefore, we can compute kv_head = h // gqa_ratio once and use it for all tokens for a given (b,h).
            for h in range(Hq):
                kv_head = h // gqa_ratio

                # Prepare q vector for this head
                q_vec = q[b, h, :].to(torch.float32).contiguous()
                q_ptr = q_vec

                # Gather k and v for tokens: k_ptr and v_ptr are [num_tokens, D]
                k_ptr = torch.empty((num_tokens, D), dtype=torch.float32, device=device)
                v_ptr = torch.empty((num_tokens, D), dtype=torch.float32, device=device)

                # Iterate tokens to populate k_ptr and v_ptr; since we need per-token kv head, we compute from q's head mapping.
                # Note: kv_head is same for all tokens for this (b,h) because of GQA mapping.
                for tt in range(0, num_tokens):
                    token_idx = int(token_indices[tt].item())  # Python int
                    k_chunk = k_cache[token_idx, 0, kv_head, :].to(torch.float32).contiguous()
                    v_chunk = v_cache[token_idx, 0, kv_head, :].to(torch.float32).contiguous()
                    k_ptr[tt, :] = k_chunk
                    v_ptr[tt, :] = v_chunk

                # Allocate per-token logits
                logits = torch.empty((num_tokens,), dtype=torch.float32, device=device)

                # Launch Triton kernel to compute logits per token
                # We pass D as tl.constexpr (compile-time) to Triton kernel
                grid = (num_tokens,)
                _compute_logits_kernel_2d[grid](
                    q_ptr, k_ptr, logits, num_tokens, self.D, self.sm_scale
                )

                # Launch Triton kernel to compute output accumulation per token using atomics
                out = torch.zeros((D,), dtype=torch.float32, device=device)
                _compute_out_atomic_kernel_2d[grid](
                    logits, v_ptr, out, num_tokens, self.D
                )
                # Add to output[b, h, :]
                output[b, h, :] += out

                # Compute lse[b, h] using Triton kernel
                # Pass logits vector pointer; Triton will read elements up to num_tokens
                _lse_atomic_kernel_2d[(1,)](  # (B, Hq) grid; we dispatch per (b,h)
                    logits, b, h, num_tokens, lse, MAX_TOKS=1024  # MAX_TOKS must be tl.constexpr
                )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
