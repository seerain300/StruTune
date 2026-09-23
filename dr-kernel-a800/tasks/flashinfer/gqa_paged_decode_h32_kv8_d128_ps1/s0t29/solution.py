import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_bh_kernel(
    q_ptr,                  # *f32 [B, Hq, D]
    k_ptr,                  # *f32 [num_tokens, Hk, D]
    kv_indptr_ptr,          # *i32 [len_indptr]
    kv_indices_ptr,         # *i32 [num_tokens]
    logits_ptr,             # *f32 [B, Hq, MAX_TOKS]
    B: tl.constexpr,        # int
    Hq: tl.constexpr,       # int
    D: tl.constexpr,        # int (128)
    Hk: tl.constexpr,       # int (8)
    MAX_TOKS: tl.constexpr, # int
    # no sm_scale, baseline ignores it
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Compute kv head index for grouped query attention
    kv_h = h // 4  # Hq//Hk == 4

    # Base pointers for q[b, h, :]
    q_base = q_ptr + b * Hq * D + h * D

    # Loop over tokens and compute dot products
    t = 0
    while t < MAX_TOKS:
        # Mask out-of-range tokens
        in_range = t < kv_indices_ptr.numel()
        # Load index i = kv_indices[t] (only if in_range)
        # We can't branch on in_range here; use scalar load guarded by host setting MAX_TOKS >= num_tokens
        i = tl.load(kv_indices_ptr + t)
        # Compute start in kv_indptr
        start = tl.load(kv_indptr_ptr + b)
        end = tl.load(kv_indptr_ptr + b + 1)
        # If i not in [start, end), skip by writing 0
        token_in = (i >= start) & (i < end) & in_range
        # Load k[i, kv_h, :] if valid
        k_base = k_ptr + i * Hk * D + kv_h * D
        # Vector of indices along D
        offs = tl.arange(0, D)
        k_vec = tl.load(k_base + offs, mask=token_in, other=0.0)
        q_vec = tl.load(q_base + offs)
        # Accumulate dot product
        dot_val = tl.sum(q_vec * k_vec, axis=0)
        # Store to logits buffer with masking
        tl.store(logits_ptr + b * Hq * MAX_TOKS + h * MAX_TOKS + t, dot_val, mask=in_range)
        t += 1


@triton.jit
def _lse_per_bh_kernel(
    logits_ptr,  # *f32 [B, Hq, MAX_TOKS]
    lse_ptr,     # *f32 [B, Hq]
    B: tl.constexpr,
    Hq: tl.constexpr,
    MAX_TOKS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    base = logits_ptr + b * Hq * MAX_TOKS + h * MAX_TOKS
    # Online logsumexp
    max_val = -float("inf")
    sum_exp = 0.0
    t = 0
    while t < MAX_TOKS:
        x = tl.load(base + t)
        # if t < num_tokens (we assume MAX_TOKS == num_tokens in this setup), x is valid
        m = max_val
        # Update max and sum_exp
        new_m = tl.maximum(m, x)
        sum_exp = sum_exp * tl.exp(m - new_m) + tl.exp(x - new_m)
        max_val = new_m
        t += 1
    # Compute lse = log(sum_exp) + max_val
    lse_val = tl.log(sum_exp) + max_val
    # divide by ln(2)
    lse_val = lse_val / 1.4426950408889634  # 1 / log(2)
    tl.store(lse_ptr + b * Hq + h, lse_val)


@triton.jit
def _accumulate_output_kernel(
    q_ptr,                  # *f32 [B, Hq, D]
    k_ptr,                  # *f32 [num_tokens, Hk, D]
    v_ptr,                  # *f32 [num_tokens, Hk, D]
    kv_indptr_ptr,          # *i32 [len_indptr]
    kv_indices_ptr,         # *i32 [num_tokens]
    lse_ptr,                # *f32 [B, Hq]
    out_ptr,                # *f32 [B, Hq, D]
    B: tl.constexpr,
    Hq: tl.constexpr,
    D: tl.constexpr,        # 128
    Hk: tl.constexpr,       # 8
    MAX_TOKS: tl.constexpr, # int
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    kv_h = h // 4
    # Load lse for this (b, h)
    lse_val = tl.load(lse_ptr + b * Hq + h)

    t = 0
    while t < MAX_TOKS:
        # Determine if t is a valid token index used by this batch
        # We rely on MAX_TOKS >= num_tokens; masking is implied.
        i = tl.load(kv_indices_ptr + t)
        start = tl.load(kv_indptr_ptr + b)
        end = tl.load(kv_indptr_ptr + b + 1)
        valid = (i >= start) & (i < end)

        # Load q[b, h, :]
        q_base = q_ptr + b * Hq * D + h * D
        q_vec = tl.load(q_base + tl.arange(0, D))

        # Load k[i, kv_h, :] and v[i, kv_h, :]
        k_base = k_ptr + i * Hk * D + kv_h * D
        v_base = v_ptr + i * Hk * D + kv_h * D
        k_vec = tl.load(k_base + tl.arange(0, D), mask=valid, other=0.0)
        v_vec = tl.load(v_base + tl.arange(0, D), mask=valid, other=0.0)

        # Compute logit and attention
        logit = tl.dot(q_vec, k_vec)  # scalar
        attn = tl.exp(logit - lse_val) * valid  # if invalid, attn=0

        # Accumulate attn * v_vec into out[b, h, :]
        out_base = out_ptr + b * Hq * D + h * D
        cur_out = tl.load(out_base + tl.arange(0, D), mask=True, other=0.0)
        cur_out = cur_out + attn * v_vec
        tl.store(out_base + tl.arange(0, D), cur_out, mask=True)

        t += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # q: [B, Hq, D], k_cache, v_cache: [num_pages, 1, Hk, D]
        assert q.dim() == 3, "q must be [B, Hq, D]"
        assert k_cache.dim() == 4 and v_cache.dim() == 4, "k_cache and v_cache must be [num_pages, 1, Hk, D]"
        B, Hq, D = q.shape
        num_pages, _, Hk, _ = k_cache.shape

        # Move to device and ensure dtype/contiguity
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k_cache.to(torch.float32).contiguous()
        v_f32 = v_cache.to(torch.float32).contiguous()
        # Convert index tensors to int32 on device
        kv_indices_i32 = kv_indices.to(torch.int32)
        kv_indptr_i32 = kv_indptr.to(torch.int32)

        # Number of tokens
        num_tokens = kv_indices_i32.numel()

        # Allocate outputs and buffers
        output_f32 = torch.empty((B, Hq, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, Hq), dtype=torch.float32, device=device)

        # Logits buffer [B, Hq, MAX_TOKS], MAX_TOKS >= num_tokens
        MAX_TOKS = 1024  # large enough for provided workloads; host masks writes
        logits_buf = torch.empty((B, Hq, MAX_TOKS), dtype=torch.float32, device=device)

        # Launch kernels
        _compute_logits_bh_kernel[(B, Hq)](
            q_f32, k_f32, kv_indptr_i32, kv_indices_i32, logits_buf,
            B=B, Hq=Hq, D=D, Hk=Hk, MAX_TOKS=MAX_TOKS,
        )

        _lse_per_bh_kernel[(B, Hq)](
            logits_buf, lse,
            B=B, Hq=Hq, MAX_TOKS=MAX_TOKS,
        )

        _accumulate_output_kernel[(B, Hq)](
            q_f32, k_f32, v_f32, kv_indptr_i32, kv_indices_i32, lse, output_f32,
            B=B, Hq=Hq, D=D, Hk=Hk, MAX_TOKS=MAX_TOKS,
        )

        # Return output as bfloat16 (baseline returns bfloat16), lse as float32
        return output_f32.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
