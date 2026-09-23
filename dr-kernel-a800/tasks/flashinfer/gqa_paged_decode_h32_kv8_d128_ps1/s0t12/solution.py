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
#   num_tokens:  i32
#   D:           i32
#   sm_scale:    f32
@triton.jit
def _compute_logits_per_token_kernel(
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
    for offs in range(0, D, 128):
        idx = offs + tl.arange(0, 128)
        mask = idx < D
        q_vec = tl.load(q_ptr + idx, mask=mask, other=0.0)
        k_vec = tl.load(k_ptr + t * D + idx, mask=mask, other=0.0)
        acc += tl.sum(q_vec * k_vec, axis=0)
    x = acc * sm_scale
    tl.store(logits_ptr + t, x)


# Triton kernel: accumulate output[b, h, :] across tokens using softmax of scaled logits
# Grid: (B, Hq)
# Inputs:
#   q_ptr:     *f32, [B*Hq*D] contiguous (we index via pid, h, i)
#   k_ptr:     *f32, [num_tokens, D] contiguous
#   v_ptr:     *f32, [num_tokens, D] contiguous
#   out_ptr:   *f32, [B*Hq*D] contiguous view of output
#   num_tokens:i32
#   D:         i32
#   sm_scale:  f32
@triton.jit
def _accumulate_output_atomic_kernel(
    q_ptr,               # *f32, [B*Hq*D] but we will index via pid
    k_ptr,               # *f32, [num_tokens, D]
    v_ptr,               # *f32, [num_tokens, D]
    out_ptr,             # *f32, [B*Hq*D]
    num_tokens,          # i32
    D,                   # i32
    sm_scale,            # f32
):
    pid = tl.program_id(0)
    b = pid // 32
    h = pid % 32
    base_out = b * 32 * 128 + h * 128
    # Online logsumexp over scaled logits for this (b, h)
    max_val = -float('inf')
    sum_exp = 0.0
    t = 0
    # First pass: compute max and sum_exp
    while t < num_tokens:
        kv_head = h // 4  # GQA mapping
        acc = 0.0
        for offs in range(0, D, 128):
            idx = offs + tl.arange(0, 128)
            mask = idx < D
            # q[b, h, :]
            q_vec = tl.load(q_ptr + b * 32 * 128 + h * 128 + idx, mask=mask, other=0.0)
            # k[t, kv_head, :]
            k_vec = tl.load(k_ptr + t * D + idx, mask=mask, other=0.0)
            acc += tl.sum(q_vec * k_vec, axis=0)
        x = acc * sm_scale
        if x > max_val:
            sum_exp = sum_exp * tl.exp(max_val - x) + 1.0
            max_val = x
        else:
            sum_exp = sum_exp + tl.exp(x - max_val)
        t += 1

    # Second pass: accumulate output
    for t in range(0, num_tokens):
        kv_head = h // 4
        acc = 0.0
        for offs in range(0, D, 128):
            idx = offs + tl.arange(0, 128)
            mask = idx < D
            q_vec = tl.load(q_ptr + b * 32 * 128 + h * 128 + idx, mask=mask, other=0.0)
            k_vec = tl.load(k_ptr + t * D + idx, mask=mask, other=0.0)
            acc += tl.sum(q_vec * k_vec, axis=0)
        x = acc * sm_scale
        p = tl.exp(x - max_val) / sum_exp  # softmax probability for this token
        v_vec = tl.load(v_ptr + t * D + tl.arange(0, 128), mask=tl.arange(0, 128) < D, other=0.0)
        # Atomic add p * v_vec to out[b, h, :]
        for offs in range(0, D, 128):
            idx = offs + tl.arange(0, 128)
            mask = idx < D
            tl.atomic_add(out_ptr + base_out + idx, p * v_vec[idx], mask=mask)


# Triton kernel: compute lse[b, h] = logsumexp(logits_scaled) / ln(2)
# Grid: (B, Hq)
# Inputs:
#   q_ptr:    *f32, [B*Hq*D] indexed via pid
#   k_ptr:    *f32, [num_tokens, D]
#   lse_ptr:  *f32, [B*Hq]
#   num_tokens: i32
#   D:        i32
#   sm_scale: f32
@triton.jit
def _lse_per_bh_kernel(
    q_ptr,               # *f32, [B*Hq*D]
    k_ptr,               # *f32, [num_tokens, D]
    lse_ptr,             # *f32, [B*Hq]
    num_tokens,          # i32
    D,                   # i32
    sm_scale,            # f32
):
    pid = tl.program_id(0)
    b = pid // 32
    h = pid % 32
    base_q = b * 32 * 128 + h * 128
    max_val = -float('inf')
    sum_exp = 0.0
    t = 0
    while t < num_tokens:
        kv_head = h // 4
        acc = 0.0
        for offs in range(0, D, 128):
            idx = offs + tl.arange(0, 128)
            mask = idx < D
            q_vec = tl.load(q_ptr + base_q + idx, mask=mask, other=0.0)
            k_vec = tl.load(k_ptr + t * D + idx, mask=mask, other=0.0)
            acc += tl.sum(q_vec * k_vec, axis=0)
        x = acc * sm_scale
        if x > max_val:
            sum_exp = sum_exp * tl.exp(max_val - x) + 1.0
            max_val = x
        else:
            sum_exp = sum_exp + tl.exp(x - max_val)
        t += 1
    lse = max_val + tl.log(sum_exp)  # logsumexp of scaled logits
    # Divide by ln(2) to match original
    lse = lse / tl.log(2.0)
    tl.store(lse_ptr + pid, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, _unused_param=None):
        # Ensure CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All inputs must be CUDA tensors."
        device = q.device
        B, Hq, D = q.shape
        assert Hq == 32, "num_qo_heads must be 32"
        assert v_cache.shape[2] == 8, "num_kv_heads must be 8"
        assert D == 128, "head_dim must be 128"

        num_tokens = kv_indices.shape[0]

        # Output and lse buffers (float32 for compute)
        output = torch.zeros((B, Hq, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, Hq), dtype=torch.float32, device=device)

        # Convert q to float32 for compute
        q_f32 = q.to(torch.float32)

        # 1) Compute logits per token
        logits = torch.empty((num_tokens,), dtype=torch.float32, device=device)
        _compute_logits_per_token_kernel[(num_tokens,)](
            q_f32.view(B * Hq, D)[0],  # we don't use b/h split here; q per token comes from q_f32
            k_cache,                   # *f32, [num_tokens, D]
            logits,                    # *f32, [num_tokens]
            num_tokens, D, float(sm_scale),
        )

        # 2) Accumulate output using Triton
        output_flat = output.view(-1)
        _accumulate_output_atomic_kernel[(B * Hq,)](
            q_f32.view(B * Hq, D),     # q_ptr
            k_cache,                   # *f32, [num_tokens, D]
            v_cache,                   # *f32, [num_tokens, D]
            output_flat,               # *f32, [B*Hq*D]
            num_tokens, D, float(sm_scale),
        )

        # 3) Compute lse per (b, h)
        _lse_per_bh_kernel[(B * Hq,)](
            q_f32.view(B * Hq, D),     # q_ptr
            k_cache,                   # *f32, [num_tokens, D]
            lse,                       # *f32, [B*Hq]
            num_tokens, D, float(sm_scale),
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
