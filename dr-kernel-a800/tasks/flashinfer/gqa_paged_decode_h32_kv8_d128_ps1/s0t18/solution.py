import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_bh_kernel(
    q_ptr,            # *f32, [B*Hq*D] flattened
    k_ptr,            # *f32, [num_pages*Hk*D] flattened
    kv_indptr_ptr,    # *i32, [B+1]
    kv_indices_ptr,   # *i32, [num_tokens]
    logits_ptr,       # *f32, [B*Hq*num_tokens] flattened
    B: tl.constexpr,  # int
    Hq: tl.constexpr, # int
    D: tl.constexpr,  # int
    Hk: tl.constexpr, # int
    gqa_ratio: tl.constexpr,  # int (Hq // Hk)
    sm_scale,         # f32
    num_tokens,       # i32
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Compute base offsets
    # q[b, h, :] contiguous offset
    q_base = (b * Hq + h) * D
    # h -> kv_head
    kv_head = h // gqa_ratio
    # Gather all tokens for this batch
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + t)
        # k[tok_idx, kv_head, :] offset
        # k layout: [num_pages, Hk, D] with H = 1 in our inputs
        k_offset = tok_idx * (Hk * D) + kv_head * D
        # Load q and k vectors
        q_vec = tl.load(q_ptr + q_base + tl.arange(0, D))
        k_vec = tl.load(k_ptr + k_offset + tl.arange(0, D))
        # Dot product
        acc = 0.0
        i = 0
        while i < D:
            acc += q_vec[i] * k_vec[i]
            i += 1
        logits_val = acc * sm_scale
        # Store to logits[B*Hq*num_tokens]
        logits_index = b * (Hq * num_tokens) + h * num_tokens + t
        tl.store(logits_ptr + logits_index, logits_val)
        t += 1


@triton.jit
def _lse_per_bh_kernel(
    logits_ptr,       # *f32, [B*Hq*num_tokens]
    lse_ptr,          # *f32, [B*Hq]
    B: tl.constexpr,  # int
    Hq: tl.constexpr, # int
    num_tokens: tl.constexpr,  # int
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    index = b * (Hq * num_tokens) + h * num_tokens
    # online logsumexp
    m = -float("inf")
    s = 0.0
    t = 0
    while t < num_tokens:
        x = tl.load(logits_ptr + index + t)
        if x > m:
            s += tl.exp(x - m)
            m = x
        else:
            s += tl.exp(m + x - 2.0 * m)
        t += 1
    lse_val = m + tl.log(s)  # logsumexp
    lse_val = lse_val / math.log(2.0)  # divide by ln(2)
    tl.store(lse_ptr + b * Hq + h, lse_val)


@triton.jit
def _accumulate_output_bh_kernel(
    q_ptr,            # *f32, [B*Hq*D] flattened
    k_ptr,            # *f32, [num_pages*Hk*D] flattened
    kv_indptr_ptr,    # *i32, [B+1]
    kv_indices_ptr,   # *i32, [num_tokens]
    v_ptr,            # *f32, [num_pages*Hk*D] flattened
    output_ptr,       # *f32, [B*Hq*D] flattened
    B: tl.constexpr,  # int
    Hq: tl.constexpr, # int
    D: tl.constexpr,  # int
    Hk: tl.constexpr, # int
    gqa_ratio: tl.constexpr,  # int (Hq // Hk)
    sm_scale,         # f32
    num_tokens,       # i32
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    q_base = (b * Hq + h) * D
    kv_head = h // gqa_ratio
    # Compute sum_exp for softmax across tokens
    sum_exp = 0.0
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + t)
        k_offset = tok_idx * (Hk * D) + kv_head * D
        q_vec = tl.load(q_ptr + q_base + tl.arange(0, D))
        k_vec = tl.load(k_ptr + k_offset + tl.arange(0, D))
        acc = 0.0
        i = 0
        while i < D:
            acc += q_vec[i] * k_vec[i]
            i += 1
        x = acc * sm_scale
        sum_exp += tl.exp(x)
        t += 1

    # Accumulate output[b, h, :]
    i = 0
    while i < D:
        # For each token, add attn * v[token, kv_head, i] to out[b, h, i]
        t = 0
        while t < num_tokens:
            tok_idx = tl.load(kv_indices_ptr + t)
            k_offset = tok_idx * (Hk * D) + kv_head * D
            q_vec = tl.load(q_ptr + q_base + tl.arange(0, D))
            k_vec = tl.load(k_ptr + k_offset + tl.arange(0, D))
            acc = 0.0
            j = 0
            while j < D:
                acc += q_vec[j] * k_vec[j]
                j += 1
            x = acc * sm_scale
            attn = tl.exp(x) / sum_exp
            v_offset = tok_idx * (Hk * D) + kv_head * D
            v_vec = tl.load(v_ptr + v_offset + tl.arange(0, D))
            # contribution for this i
            out_val = attn * v_vec[i]
            out_ptr_index = (b * Hq + h) * D + i
            tl.store(output_ptr + out_ptr_index, tl.load(output_ptr + out_ptr_index) + out_val)
            t += 1
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No __init__ args; this is a functional module

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, Hq, D], bfloat16
        k_cache, v_cache: [num_pages, 1, Hk, D], bfloat16 (Hq=32, Hk=8, D=128)
        kv_indptr: [B+1], int32
        kv_indices: [num_tokens], int32
        sm_scale: float (ignored, to match original behavior)
        Returns:
        - output: [B, Hq, D], bfloat16
        - lse: [B, Hq], float32
        """
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
        B, Hq, D = q.shape
        num_pages, _, Hk, _ = k_cache.shape
        assert Hq == 32 and Hk == 8 and D == 128  # original constraints
        assert kv_indptr.shape[0] == B + 1
        num_tokens = kv_indices.numel()
        # The original run ignores sm_scale; we keep behavior consistent
        # Prepare flattened pointers
        q_f32 = q.to(torch.float32).contiguous()                # [B, Hq, D] f32
        k_flat = k_cache.to(torch.float32).contiguous().view(-1)  # [num_pages*Hk*D]
        v_flat = v_cache.to(torch.float32).contiguous().view(-1)  # [num_pages*Hk*D]

        # Logits buffer: [B, Hq, num_tokens] f32
        logits = torch.empty((B, Hq, num_tokens), dtype=torch.float32, device=q.device)

        # lse: [B, Hq] f32
        lse = torch.empty((B, Hq), dtype=torch.float32, device=q.device)

        # Launch compute_logits_bh_kernel
        grid_logits = (B, Hq)
        _compute_logits_bh_kernel[grid_logits](
            q_f32, k_flat, kv_indptr, kv_indices, logits,
            B=B, Hq=Hq, D=D, Hk=Hk, gqa_ratio=Hq // Hk,
            sm_scale=float(sm_scale),  # pass sm_scale to kernel (even though we ignore it)
            num_tokens=num_tokens
        )

        # Launch lse per (b,h)
        grid_lse = (B, Hq)
        _lse_per_bh_kernel[grid_lse](
            logits, lse,
            B=B, Hq=Hq, num_tokens=num_tokens
        )

        # Accumulate output per (b,h)
        output = torch.zeros((B, Hq, D), dtype=torch.float32, device=q.device)
        grid_out = (B, Hq)
        _accumulate_output_bh_kernel[grid_out](
            q_f32, k_flat, kv_indptr, kv_indices, v_flat, output,
            B=B, Hq=Hq, D=D, Hk=Hk, gqa_ratio=Hq // Hk,
            sm_scale=float(sm_scale),
            num_tokens=num_tokens
        )

        # Cast output to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
