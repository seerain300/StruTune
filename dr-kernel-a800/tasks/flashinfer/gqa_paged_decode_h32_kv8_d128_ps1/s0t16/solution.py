import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits_scaled[b, h, t] = (q[b, h, :] · k[token, kv_head]) * sm_scale
# Grid: (B, Hq)
# Outputs:
#   logits_ptr: *f32, shape [B, Hq, MAX_TOKS]
@triton.jit
def _compute_logits_bh_kernel(
    q_ptr,               # *f32, [B*Hq, D], contiguous as q.view(B*Hq, D)
    k_ptr,               # *f32, [B, 1, Hk, D] -> we index by (idx, kv_head)
    kv_indptr_ptr,       # *i32, [B+1]
    kv_indices_ptr,      # *i32, [num_tokens]
    logits_ptr,          # *f32, [B, Hq, MAX_TOKS]
    B: tl.constexpr,     # int
    Hq: tl.constexpr,    # int
    D: tl.constexpr,     # int
    Hk: tl.constexpr,    # int
    gqa_ratio: tl.constexpr,  # int
    MAX_TOKS: tl.constexpr,   # int, number of tokens in this invocation
    sm_scale: tl.constexpr,   # float
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Compute range of tokens for this batch element
    start = tl.load(kv_indptr_ptr + b)       # i32
    end = tl.load(kv_indptr_ptr + b + 1)     # i32
    num_tokens = end - start                  # i32 scalar

    # GQA mapping: kv_head for this output head
    kv_head = h // gqa_ratio

    # Base offset into q for (b, h)
    base_q = (b * Hq + h) * D

    t = 0
    while t < MAX_TOKS:
        if t >= num_tokens:
            t += 1
            continue
        # token index within num_pages
        idx = tl.load(kv_indices_ptr + start + t)  # i32
        # Base offset into k for (idx, kv_head, :)
        base_k = idx * Hk * D + kv_head * D
        # Load q[b, h, :] vector
        q_vec = tl.load(q_ptr + base_q + tl.arange(0, D))  # [D]
        # Load k[token, kv_head, :]
        k_vec = tl.load(k_ptr + base_k + tl.arange(0, D))  # [D]
        # Dot product
        dot = tl.sum(q_vec * k_vec, axis=0)                # scalar
        scaled = dot * sm_scale
        # Store to logits buffer at [b, h, t]
        tl.store(logits_ptr + b * (Hq * MAX_TOKS) + h * MAX_TOKS + t, scaled)
        t += 1


# Triton kernel: compute lse[b, h] = logsumexp(logits_scaled[b, h, :]) / ln(2)
# Grid: (B, Hq)
@triton.jit
def _lse_per_bh_kernel(
    logits_ptr,  # *f32, [B, Hq, MAX_TOKS]
    lse_ptr,     # *f32, [B, Hq]
    B: tl.constexpr,
    Hq: tl.constexpr,
    MAX_TOKS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    base = b * (Hq * MAX_TOKS) + h * MAX_TOKS

    # Online logsumexp update
    m = -float("inf")
    s = 0.0
    t = 0
    while t < MAX_TOKS:
        val = tl.load(logits_ptr + base + t)
        if t == 0:
            m = val
        else:
            m_new = tl.maximum(m, val)
            s = s * tl.exp(m - m) + tl.exp(val - m_new)
            m = m_new
        t += 1
    lse_val = m + tl.log(s)  # logsumexp
    ln2 = 0.6931471805599453
    tl.store(lse_ptr + b * Hq + h, lse_val / ln2)


# Triton kernel: accumulate output[b, h, :] += softmax(logits_scaled[b,h,:]) * v[token, kv_head, :]
# Grid: (B, Hq)
@triton.jit
def _accumulate_output_bh_kernel(
    q_ptr,               # *f32, [B*Hq, D]
    k_ptr,               # *f32, [B, 1, Hk, D]
    v_ptr,               # *f32, [B, 1, Hk, D]
    kv_indptr_ptr,       # *i32, [B+1]
    kv_indices_ptr,      # *i32, [num_tokens]
    out_ptr,             # *f32, [B, Hq, D]
    B: tl.constexpr,
    Hq: tl.constexpr,
    D: tl.constexpr,
    Hk: tl.constexpr,
    gqa_ratio: tl.constexpr,
    MAX_TOKS: tl.constexpr,
    sm_scale: tl.constexpr,  # not used here; original code didn't scale in forward
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    start = tl.load(kv_indptr_ptr + b)   # i32
    end = tl.load(kv_indptr_ptr + b + 1) # i32
    num_tokens = end - start              # i32 scalar
    kv_head = h // gqa_ratio

    base_q = (b * Hq + h) * D

    # Compute sum_exp across all tokens for this (b,h): sum_t exp(scaled_logits)
    sum_exp = 0.0
    t = 0
    while t < MAX_TOKS:
        if t >= num_tokens:
            t += 1
            continue
        idx = tl.load(kv_indices_ptr + start + t)   # i32
        base_k = idx * Hk * D + kv_head * D
        q_vec = tl.load(q_ptr + base_q + tl.arange(0, D))  # [D]
        k_vec = tl.load(k_ptr + base_k + tl.arange(0, D))  # [D]
        dot = tl.sum(q_vec * k_vec, axis=0)                # scalar
        scaled = dot * sm_scale
        sum_exp += tl.exp(scaled)
        t += 1

    # Accumulate output[b, h, :] = sum_t softmax(...) * v[token, kv_head, :]
    i = 0
    while i < D:
        out_i = 0.0
        t = 0
        while t < MAX_TOKS:
            if t >= num_tokens:
                t += 1
                continue
            idx = tl.load(kv_indices_ptr + start + t)   # i32
            base_k = idx * Hk * D + kv_head * D
            q_vec = tl.load(q_ptr + base_q + tl.arange(0, D))  # [D]
            k_vec = tl.load(k_ptr + base_k + tl.arange(0, D))  # [D]
            dot = tl.sum(q_vec * k_vec, axis=0)
            scaled = dot * sm_scale
            attn = tl.exp(scaled) / sum_exp  # softmax probability
            base_v = idx * Hk * D + kv_head * D
            v_vec = tl.load(v_ptr + base_v + tl.arange(0, D))  # [D]
            out_i += attn * v_vec[i]
            t += 1
        tl.store(out_ptr + b * (Hq * D) + h * D + i, out_i)
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self, B, Hq, D, Hk, gqa_ratio):
        super().__init__()
        self.B = B
        self.Hq = Hq
        self.D = D
        self.Hk = Hk
        self.gqa_ratio = gqa_ratio

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA."
        # Shapes (assertions)
        B = self.B
        Hq = self.Hq
        D = self.D
        Hk = self.Hk
        gqa_ratio = self.gqa_ratio

        assert q.shape == (B, Hq, D)
        assert k_cache.shape == (B, 1, Hk, D) and v_cache.shape == (B, 1, Hk, D)

        # Make contiguous
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # num_tokens is derived from kv_indptr (len_indptr == B+1, num_tokens = kv_indices.shape[0])
        num_tokens = kv_indices.numel()

        # Allocate intermediate buffers
        logits = torch.empty((B, Hq, num_tokens), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, Hq), dtype=torch.float32, device=q.device)
        output = torch.empty((B, Hq, D), dtype=torch.float32, device=q.device)

        # Launch Triton kernels
        # 1) compute logits_scaled
        _compute_logits_bh_kernel[(B, Hq)](
            q.view(B * Hq, D), k_cache, kv_indptr, kv_indices, logits,
            B=self.B, Hq=self.Hq, D=self.D, Hk=self.Hk, gqa_ratio=self.gqa_ratio,
            MAX_TOKS=num_tokens, sm_scale=float(sm_scale),
            num_warps=1,
        )

        # 2) compute lse
        _lse_per_bh_kernel[(B, Hq)](
            logits, lse,
            B=self.B, Hq=self.Hq, MAX_TOKS=num_tokens,
            num_warps=1,
        )

        # 3) accumulate output
        _accumulate_output_bh_kernel[(B, Hq)](
            q.view(B * Hq, D), k_cache, v_cache, kv_indptr, kv_indices, output,
            B=self.B, Hq=self.Hq, D=self.D, Hk=self.Hk, gqa_ratio=self.gqa_ratio,
            MAX_TOKS=num_tokens, sm_scale=float(sm_scale),
            num_warps=1,
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
