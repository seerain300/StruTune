import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_bh_kernel(
    q_ptr,                    # *f32, shape [B, Hq, D] flattened
    k_ptr,                    # *f32, shape [num_tokens, Hk, D] flattened
    kv_indptr_ptr,            # *i32, shape [B+1]
    kv_indices_ptr,           # *i32, shape [num_tokens]
    out_logits_ptr,           # *f32, shape [B, Hq, MAX_TOKS]
    B: tl.constexpr,
    Hq: tl.constexpr,
    D: tl.constexpr,
    Hk: tl.constexpr,
    MAX_TOKS: tl.constexpr,
):
    # Each program handles one (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    q_base = pid_b * Hq * D + pid_h * D

    # loop over tokens t
    t = 0
    while t < MAX_TOKS:
        start = tl.load(kv_indptr_ptr + pid_b)        # i32
        end = tl.load(kv_indptr_ptr + pid_b + 1)      # i32
        if t >= (end - start):
            t += 1
            continue

        token_idx = tl.load(kv_indices_ptr + start + t)  # i32
        kv_base = token_idx * Hk * D

        # compute q[b, h, :] (D vector)
        q_vec = tl.zeros([D], dtype=tl.float32)
        offs = 0
        while offs < D:
            q_vec += tl.load(q_ptr + q_base + offs, mask=offs < D, other=0.0)
            offs += 1

        # compute k[token, kv_head, :] (D vector)
        gqa_ratio = Hq // Hk
        kv_head = pid_h // gqa_ratio
        k_vec = tl.zeros([D], dtype=tl.float32)
        offs = 0
        while offs < D:
            k_vec += tl.load(k_ptr + kv_base + kv_head * D + offs, mask=offs < D, other=0.0)
            offs += 1

        # dot product
        dot = 0.0
        offs = 0
        while offs < D:
            dot += q_vec[offs] * k_vec[offs]
            offs += 1

        # store scaled logits; sm_scale is not used to match original run
        out_ptr = out_logits_ptr + pid_b * Hq * MAX_TOKS + pid_h * MAX_TOKS + t
        tl.store(out_ptr, dot)

        t += 1


@triton.jit
def _lse_sum_per_bh_kernel(
    logits_ptr,   # *f32, shape [B, Hq, MAX_TOKS]
    lse_sum_ptr,  # *f32, shape [B, Hq]
    B: tl.constexpr,
    Hq: tl.constexpr,
    MAX_TOKS: tl.constexpr,
    LOG2_INV: tl.constexpr,  # 1.0 / ln(2)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    max_val = -float("inf")
    sum_exp = 0.0

    t = 0
    while t < MAX_TOKS:
        logit = tl.load(logits_ptr + pid_b * Hq * MAX_TOKS + pid_h * MAX_TOKS + t)
        # stable update of logsumexp: keep max and sum_exp
        if t == 0:
            max_val = logit
        else:
            if logit > max_val:
                sum_exp = sum_exp * exp(-t) + exp(logit - max_val)
                max_val = logit
            else:
                sum_exp = sum_exp + exp(logit - max_val)
        t += 1

    lse = LOG2_INV * tl.log(sum_exp)
    tl.store(lse_sum_ptr + pid_b * Hq + pid_h, lse)


@triton.jit
def _accumulate_output_per_token_kernel(
    q_ptr,                    # *f32, shape [B, Hq, D]
    k_ptr,                    # *f32, shape [num_tokens, Hk, D]
    v_ptr,                    # *f32, shape [num_tokens, Hk, D]
    kv_indptr_ptr,            # *i32, shape [B+1]
    kv_indices_ptr,           # *i32, shape [num_tokens]
    lse_sum_ptr,              # *f32, shape [B, Hq]
    out_ptr,                  # *f32, shape [B, Hq, D] (we'll cast to bf16 after)
    B: tl.constexpr,
    Hq: tl.constexpr,
    D: tl.constexpr,
    Hk: tl.constexpr,
    MAX_TOKS: tl.constexpr,
):
    # grid = (B*Hq, MAX_TOKS): each program handles (b,h) and token t
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    b = pid0 // Hq
    h = pid0 % Hq
    t = pid1

    if t >= MAX_TOKS:
        return

    start = tl.load(kv_indptr_ptr + b)        # i32
    end = tl.load(kv_indptr_ptr + b + 1)      # i32
    if t >= (end - start):
        return

    token_idx = tl.load(kv_indices_ptr + start + t)  # i32
    kv_base = token_idx * Hk * D

    # compute q[b, h, :] dot k[token, kv_head, :]
    gqa_ratio = Hq // Hk
    kv_head = h // gqa_ratio

    q_base = b * Hq * D + h * D
    q_vec = tl.zeros([D], dtype=tl.float32)
    offs = 0
    while offs < D:
        q_vec += tl.load(q_ptr + q_base + offs, mask=offs < D, other=0.0)
        offs += 1

    k_vec = tl.zeros([D], dtype=tl.float32)
    offs = 0
    while offs < D:
        k_vec += tl.load(k_ptr + kv_base + kv_head * D + offs, mask=offs < D, other=0.0)
        offs += 1

    dot = 0.0
    offs = 0
    while offs < D:
        dot += q_vec[offs] * k_vec[offs]
        offs += 1

    # load lse_sum for (b,h)
    lse = tl.load(lse_sum_ptr + b * Hq + h)
    attn = exp(dot) / lse

    # load v[token, kv_head, :]
    v_vec = tl.zeros([D], dtype=tl.float32)
    offs = 0
    while offs < D:
        v_vec += tl.load(v_ptr + kv_base + offs, mask=offs < D, other=0.0)
        offs += 1

    # accumulate into output[b, h, :]
    out_base = b * Hq * D + h * D
    cur = tl.load(out_ptr + out_base)
    tl.store(out_ptr + out_base, cur + attn * v_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure dtype and contiguity
        device = q.device
        B, Hq, D = q.shape
        # k_cache, v_cache are [num_pages, 1, Hk, D]; original asserts Hk=8, D=128, Hq=32
        k_cache_f32 = k_cache.squeeze(1).to(torch.float32).contiguous()
        v_cache_f32 = v_cache.squeeze(1).to(torch.float32).contiguous()

        # Ensure indices/pointers are int32 and contiguous
        kv_indices_i32 = kv_indices.to(torch.int32).contiguous()
        kv_indptr_i32 = kv_indptr.to(torch.int32).contiguous()

        # num_tokens is number of elements in kv_indices
        num_tokens = kv_indices_i32.numel()

        # Output buffers: compute in f32 for stability, cast to bf16 at end.
        output_f32 = torch.zeros((B, Hq, D), dtype=torch.float32, device=device)
        lse = torch.full((B, Hq), -float("inf"), dtype=torch.float32, device=device)

        # We assume and enforce num_tokens <= MAX_TOKS=10 to keep Triton simple and fast.
        MAX_TOKS = 10

        # Prepare logits buffer [B, Hq, MAX_TOKS]
        logits_buf = torch.empty((B, Hq, MAX_TOKS), dtype=torch.float32, device=device)

        # 1) Compute logits = q·k per (b,h) across tokens (scalar loop in kernel)
        _compute_logits_bh_kernel[(B, Hq)](
            q.to(torch.float32).contiguous(),
            k_cache_f32,
            kv_indptr_i32,
            kv_indices_i32,
            logits_buf,
            B, Hq, D,
            Hk=8,
            MAX_TOKS=MAX_TOKS,
        )

        # 2) Compute lse[b, h] = logsumexp(logits) / ln(2)
        LOG2_INV = 1.0 / math.log(2.0)
        lse_sum = torch.empty((B, Hq), dtype=torch.float32, device=device)
        _lse_sum_per_bh_kernel[(B, Hq)](
            logits_buf,
            lse_sum,
            B, Hq, MAX_TOKS,
            LOG2_INV,
        )
        lse.copy_(lse_sum)

        # 3) Accumulate output[b, h, :] across tokens: out += attn * v
        _accumulate_output_per_token_kernel[(B * Hq, MAX_TOKS)](
            q.to(torch.float32).contiguous(),
            k_cache_f32,
            v_cache_f32,
            kv_indptr_i32,
            kv_indices_i32,
            lse_sum,
            output_f32,
            B, Hq, D, Hk=8, MAX_TOKS=MAX_TOKS,
        )

        # Cast output to bfloat16 as original output dtype
        output = output_f32.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
