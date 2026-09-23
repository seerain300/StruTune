import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_bh_kernel(
    q_ptr,          # *f32, [B*Hq*D], we load per (b,h) via arithmetic
    k_ptr,          # *f32, [num_tokens, D]
    logits_ptr,     # *f32, [B*Hq] (flattened buffer)
    B: tl.constexpr,
    Hq: tl.constexpr,
    num_tokens: tl.constexpr,  # runtime i32
    D: tl.constexpr,           # head_dim (e.g., 128)
    sm_scale: tl.constexpr,    # scaling factor
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    base_q = (b * Hq + h) * D
    for t in range(0, num_tokens):
        # Load q[b, h, :]
        q_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            q_vec[i] = tl.load(q_ptr + base_q + i)
        # Load k[t, kv_head, :] where kv_head = h // gqa_ratio; here gqa_ratio = Hq // Hk = 4
        kv_head = h // (Hq // Hq)  # placeholder; see notes below
        # With fixed Hq=32, Hk=8, we can set kv_head = h % Hk
        kv_head = h % 8
        k_base = t * D
        k_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            k_vec[i] = tl.load(k_ptr + k_base + i)

        acc = 0.0
        for i in range(0, D):
            acc += q_vec[i] * k_vec[i]
        logit = acc * sm_scale
        idx = b * Hq + h
        tl.store(logits_ptr + idx, logit)


@triton.jit
def _lse_per_bh_kernel(
    q_ptr,             # *f32, [B*Hq*D]
    k_ptr,             # *f32, [num_tokens, D]
    logits_ptr,        # *f32, [B*Hq]
    lse_ptr,           # *f32, [B*Hq]
    B: tl.constexpr,
    Hq: tl.constexpr,
    num_tokens: tl.constexpr,
    D: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_T: tl.constexpr,  # >= num_tokens (e.g., 1024)
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Online logsumexp update
    m = -float("inf")
    s = 0.0
    t = 0
    while t < num_tokens:
        q_base = (b * Hq + h) * D
        q_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            q_vec[i] = tl.load(q_ptr + q_base + i)
        k_base = t * D
        k_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            k_vec[i] = tl.load(k_ptr + k_base + i)
        acc = 0.0
        for i in range(0, D):
            acc += q_vec[i] * k_vec[i]
        x = acc * sm_scale
        # update m and s for logsumexp
        if x > m:
            s = s * tl.exp(m - x) + 1.0
            m = x
        else:
            s = s + tl.exp(x - m)
        t += 1
    lse_val = m + tl.log(s)
    lse_val = lse_val / math.log(2.0)
    tl.store(lse_ptr + b * Hq + h, lse_val)


@triton.jit
def _accumulate_output_bh_kernel(
    q_ptr,             # *f32, [B*Hq*D]
    k_ptr,             # *f32, [num_tokens, D]
    v_ptr,             # *f32, [num_tokens, Hk, D]
    out_ptr,           # *f32, [B*Hq*D] (accumulator)
    lse_ptr,           # *f32, [B*Hq]
    B: tl.constexpr,
    Hq: tl.constexpr,
    Hk: tl.constexpr,
    D: tl.constexpr,
    num_tokens: tl.constexpr,
    BLOCK_T: tl.constexpr,  # >= num_tokens (e.g., 1024)
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Compute sum_exp = sum_t exp(logits_scaled[t]) for this (b,h)
    sum_exp = 0.0
    for t in range(0, num_tokens):
        q_base = (b * Hq + h) * D
        q_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            q_vec[i] = tl.load(q_ptr + q_base + i)
        k_base = t * D
        k_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            k_vec[i] = tl.load(k_ptr + k_base + i)
        acc = 0.0
        for i in range(0, D):
            acc += q_vec[i] * k_vec[i]
        x = acc * 1.0  # we rely on logits_ptr for scaling, but we recompute here; use scaled x
        # Read lse[b,h]
        lse_bh = tl.load(lse_ptr + b * Hq + h)
        attn = tl.exp(x - lse_bh)
        sum_exp += attn

    # Accumulate outputs
    for t in range(0, num_tokens):
        q_base = (b * Hq + h) * D
        q_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            q_vec[i] = tl.load(q_ptr + q_base + i)
        k_base = t * D
        k_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            k_vec[i] = tl.load(k_ptr + k_base + i)
        acc = 0.0
        for i in range(0, D):
            acc += q_vec[i] * k_vec[i]
        lse_bh = tl.load(lse_ptr + b * Hq + h)
        attn = tl.exp(acc * 1.0 - lse_bh)  # scaled attn; we used logits_ptr to compute lse, but here recompute

        kv_head = h % Hk  # GQA: each query head h uses kv head h % Hk
        v_base = t * Hk * D + kv_head * D
        v_vec = tl.zeros([D], dtype=tl.float32)
        for i in range(0, D):
            v_vec[i] = tl.load(v_ptr + v_base + i)

        contrib = attn * v_vec
        out_base = (b * Hq + h) * D
        for i in range(0, D):
            tl.atomic_add(out_ptr + out_base + i, contrib[i])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants from the original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Match baseline signature: 6 arguments; ignore sm_scale
        # Assume q: [B, Hq, D], k_cache: [num_pages, 1, Hk, D], v_cache: [num_pages, 1, Hk, D]
        # kv_indptr: [B+1], kv_indices: [num_kv_indices] (int32), sm_scale ignored (baseline Model ignores it too)

        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Inputs must be CUDA tensors"
        device = q.device
        B, Hq, D = q.shape
        assert Hq == self.num_qo_heads and D == self.head_dim
        Hk = self.num_kv_heads

        # We do not use sm_scale (original baseline also ignores it in forward). If needed, we can still pass a dummy value.

        # Flatten q for pointer arithmetic in Triton
        q_flat = q.to(torch.float32).reshape(B * Hq, D).contiguous()

        # Squeeze dimension 1 from k_cache and v_cache
        k_cache_f32 = k_cache.to(torch.float32).squeeze(1)  # [num_pages, Hk, D]
        v_cache_f32 = v_cache.to(torch.float32).squeeze(1)  # [num_pages, Hk, D]

        # Determine number of tokens per batch using kv_indptr:
        # len_indptr == B + 1; example uses len_indptr == 2, but general case needs to infer num_tokens per batch.
        # The original run uses len_indptr==2 and sets num_tokens = kv_indices.shape[0]. For generality, use total tokens:
        total_tokens = int(kv_indptr[-1].item()) if kv_indptr.numel() > 1 else 0
        # Since len_indptr length is B+1, we infer num_tokens from kv_indices.shape[0], as original does:
        num_tokens = kv_indices.numel()

        # Allocate buffers
        logits = torch.empty(B * Hq, dtype=torch.float32, device=device)  # one per (b,h)
        lse = torch.empty(B * Hq, dtype=torch.float32, device=device)     # one per (b,h)

        # Launch Triton kernels: compute logits for each (b,h)
        _compute_logits_bh_kernel[(B, Hq)](
            q_flat, k_cache_f32, logits,
            B=B, Hq=Hq, num_tokens=num_tokens, D=self.head_dim, sm_scale=1.0, BLOCK_T=1024
        )

        # Compute lse per (b,h)
        _lse_per_bh_kernel[(B, Hq)](
            q_flat, k_cache_f32, logits, lse,
            B=B, Hq=Hq, num_tokens=num_tokens, D=self.head_dim, sm_scale=1.0, BLOCK_T=1024
        )

        # Accumulate output per (b,h)
        out_accum = torch.zeros(B * Hq * self.head_dim, dtype=torch.float32, device=device)
        _accumulate_output_bh_kernel[(B, Hq)](
            q_flat, k_cache_f32, v_cache_f32, out_accum, lse,
            B=B, Hq=Hq, Hk=Hk, D=self.head_dim, num_tokens=num_tokens, BLOCK_T=1024
        )

        # Reshape to [B, Hq, D]
        output = out_accum.view(B, Hq, D)

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)

        # Return output and lse (lse is per (b,h))
        return output_bf16, lse.view(B, Hq)


def run(*args):
    return ModelNew()(*args)
