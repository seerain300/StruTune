import math
import torch
import triton
import triton.language as tl


# Kernel 1: compute logits[t] = (q[b, h, :] · k[token, kv_head, :]) * sm_scale
# Grid: (num_tokens,)
# Inputs:
#   q_ptr:       *f32, [D] pointer to q[b, h, :]
#   k_ptr:       *f32, [num_tokens, D] pointer to K gathered per token
#   logits_ptr:  *f32, [num_tokens] output buffer
#   b, h:        i32
#   num_tokens:  i32
#   D:           i32
#   kv_head:     i32 (since num_qo_heads % num_kv_heads == 4, GQA)
#   sm_scale:    f32
@triton.jit
def _compute_logits_per_token(
    q_ptr,              # *f32, [D]
    k_ptr,              # *f32, [num_tokens, D]
    logits_ptr,         # *f32, [num_tokens]
    b: tl.constexpr,    # (unused but kept for clarity)
    h: tl.constexpr,    # (unused but kept for clarity)
    num_tokens: tl.constexpr,  # compile-time bound for grid; not used in loops
    D: tl.constexpr,               # head_dim, compile-time
    kv_head: tl.constexpr,         # i32
    sm_scale: tl.constexpr,        # f32
):
    t = tl.program_id(0)
    if t >= num_tokens:
        return
    # Compute dot(q, k[t, kv_head, :])
    dot = 0.0
    for d in range(0, D):
        qd = tl.load(q_ptr + d)
        kd = tl.load(k_ptr + t * D + d)
        dot += qd * kd
    tl.store(logits_ptr + t, dot * sm_scale)


# Kernel 2: accumulate output[b, h, :] += softmax(logits_scaled)[t] * v[token, kv_head, :]
# Grid: (num_tokens,)
# Inputs:
#   logits_ptr:    *f32, [num_tokens]
#   v_ptr:         *f32, [num_tokens, D]
#   out_ptr:       *f32, [D]
#   num_tokens:    i32
#   D:             i32
# Outputs:
#   out_ptr[i] += attn[t] * v_ptr[t, i] for all i in 0..D-1 (atomic_add)
@triton.jit
def _accumulate_out_atomic_per_token(
    logits_ptr,  # *f32, [num_tokens]
    v_ptr,       # *f32, [num_tokens, D]
    out_ptr,     # *f32, [D]
    num_tokens,  # i32 (runtime)
    D: tl.constexpr,  # head_dim, compile-time
):
    t = tl.program_id(0)
    if t >= num_tokens:
        return
    # Compute sum_exp across all tokens
    sum_exp = 0.0
    tt = 0
    while tt < num_tokens:
        xt = tl.exp(tl.load(logits_ptr + tt))
        sum_exp += xt
        tt += 1
    # Compute attn for this token
    xt = tl.exp(tl.load(logits_ptr + t))
    attn = xt / sum_exp
    # Atomically accumulate into out_ptr
    v_vec = tl.load(v_ptr + t * D + tl.arange(0, D))
    for i in range(0, D):
        tl.atomic_add(out_ptr + i, attn * v_vec[i])


# Kernel 3: compute lse[b, h] = logsumexp(logits_scaled) / ln(2)
# Grid: (B, Hq)
# Inputs:
#   logits_ptr:  *f32, [num_tokens] (per-batch) — host must pass per batch slice
#   b_idx:       i32
#   h_idx:       i32
#   num_tokens:  i32
# Outputs:
#   lse_ptr[b_idx * Hq + h_idx] = (max + log(sum_exp)) / ln(2)
@triton.jit
def _lse_per_bh(
    logits_ptr,   # *f32, [num_tokens]
    b_idx,        # i32
    h_idx,        # i32
    num_tokens,   # i32
    lse_ptr,      # *f32, [B * Hq]
):
    max_val = -float('inf')
    sum_exp = 0.0
    t = 0
    while t < num_tokens:
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

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices):
        # Expect q: [B, Hq, D], k_cache: [P, 1, Hk, D], v_cache: same
        B, Hq, D = q.shape
        P, _, Hk, _ = k_cache.shape
        assert Hk == self.num_kv_heads, "num_kv_heads mismatch"
        assert D == self.head_dim, "head_dim mismatch"
        assert Hq == self.num_qo_heads, "num_qo_heads mismatch"

        # Prepare device and dtypes
        device = q.device
        # We will work in float32 in Triton and return bfloat16 for output
        q_f32 = q.to(torch.float32)
        # k_cache and v_cache are [P, 1, Hk, D]; kv_indices selects tokens from flattened P
        k_cache_f32 = k_cache.to(torch.float32)
        v_cache_f32 = v_cache.to(torch.float32)

        # Compute per-batch token ranges
        # kv_indptr is 0-based inclusive cumsum: [0, len_b0, len_b1, ...]
        # tokens for batch b are indices [kv_indptr[b]: kv_indptr[b+1])
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr length must be batch_size + 1"
        num_tokens_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_list.append(end - start)
        # The original logic expects each batch b to have some tokens, but some tests may have empty (start == end).
        # We handle empty by skipping in host; however, typical workloads have num_tokens > 0.

        # Prepare outputs
        output = torch.zeros((B, Hq, D), dtype=torch.float32, device=device)
        lse = torch.full((B, Hq), -float('inf'), dtype=torch.float32, device=device)

        # For each batch b
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens = end - start
            if num_tokens <= 0:
                continue  # nothing to do for this batch

            # Gather token indices for this batch
            token_indices = kv_indices[start:end].to(torch.int32)  # [num_tokens]
            # Map to k/v with GQA: kv_head = h // gqa_ratio
            # Prepare K and V for each head across tokens
            # We will compute per-token logits and per-token accumulation across all tokens for each head
            # Note: we do not rely on torch operations in host; all heavy math in Triton.

            # Precompute kv_head mapping for output heads (GQA). For output, we need kv_head for each h.
            # We will run kernels per head h.
            for h in range(Hq):
                kv_head = h // self.gqa_ratio  # GQA mapping

                # 1) Compute logits[t] for all tokens t in this batch
                # k_ptr is [num_tokens, D], v_ptr is [num_tokens, D]
                k_flat = k_cache_f32.squeeze(1)   # [P, Hk, D] -> [P, D] not needed; we gather per token
                v_flat = v_cache_f32.squeeze(1)   # similarly
                # Build per-token k/v pointers: for each t, read k[token_indices[t], kv_head, :]
                # Create tensors for k_ptr and v_ptr on device by gathering, then pass to Triton.
                # However, Triton kernels must get pointers; we can gather directly into small tensors and pass pointers.
                # But to minimize memory, we will construct pointers via indexing into original tensors using token_indices and kv_head.

                # Create k_per_batch and v_per_batch of shape [num_tokens, D] by gathering:
                # k_per_batch[t, :] = k_cache_f32[token_indices[t], kv_head, :]
                k_per_batch = torch.empty((num_tokens, D), dtype=torch.float32, device=device)
                v_per_batch = torch.empty((num_tokens, D), dtype=torch.float32, device=device)

                for t_idx in range(num_tokens):
                    p = int(token_indices[t_idx].item())
                    k_per_batch[t_idx] = k_cache_f32[p, 0, kv_head]  # [D]
                    v_per_batch[t_idx] = v_cache_f32[p, 0, kv_head]  # [D]

                # q_vec for this (b, h)
                q_vec = q_f32[b, h]  # [D], float32

                # Allocate logits buffer
                logits = torch.empty((num_tokens,), dtype=torch.float32, device=device)

                # Launch Triton kernel to compute logits per token
                # We pass num_tokens as compile-time constexpr for grid size; Triton allows scalar loop over D.
                _compute_logits_per_token[(num_tokens,)](
                    q_vec,                # *f32, [D]
                    k_per_batch,          # *f32, [num_tokens, D]
                    logits,               # *f32, [num_tokens]
                    b, h,                 # (unused but kept)
                    D=self.head_dim,      # constexpr
                    kv_head=kv_head,      # i32
                    sm_scale=self.sm_scale,
                )

                # 2) Accumulate output[b, h, :] using atomics per token
                out_vec = torch.zeros((D,), dtype=torch.float32, device=device)
                _accumulate_out_atomic_per_token[(num_tokens,)](
                    logits,              # *f32, [num_tokens]
                    v_per_batch,         # *f32, [num_tokens, D]
                    out_vec,             # *f32, [D]
                    num_tokens=num_tokens,
                    D=self.head_dim,     # constexpr
                )

                # 3) Compute lse[b, h]
                lse_bh = torch.empty((), dtype=torch.float32, device=device)  # dummy, not used here directly
                _lse_per_bh[(1,)](
                    logits,              # *f32, [num_tokens]
                    b, h,                # indices
                    num_tokens,          # runtime i32
                    lse,                 # *f32, [B*Hq]
                )

                # Store output and lse
                output[b, h] = out_vec  # float32 vector

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
