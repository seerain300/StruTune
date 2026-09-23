import math
import torch

import triton
import triton.language as tl


# Triton kernel: accumulate output[b, h, :] = sum_t softmax((q[b,h,:] · k[token, kv_head, :]) * sm_scale) * v[token, kv_head, :]
# Grid: (B, Hq). Each program handles one (b, h) and iterates tokens t using a scalar while loop.
@triton.jit
def _accumulate_output_kernel(
    q_ptr,               # *float32, shape [B, Hq, D]
    k_ptr,               # *float32, shape [num_pages, Hk, D] (we index by kv_indices)
    v_ptr,               # *float32, shape [num_pages, Hk, D] (we index by kv_indices)
    out_ptr,             # *bf16, shape [B, Hq, D]
    kv_indices_ptr,      # *int32, shape [num_tokens]
    Hq: tl.constexpr,    # num output heads = 32
    Hk: tl.constexpr,    # num kv heads = 8
    gqa_ratio: tl.constexpr,  # Hq // Hk = 4
    num_tokens,           # int32, runtime
    sm_scale,             # float32
    B: tl.constexpr,      # batch size, runtime int
    D: tl.constexpr        # head_dim = 128, runtime int
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Compute base offsets
    q_base = (b * Hq + h) * D
    # Output accumulator in fp32
    out = tl.zeros([D], dtype=tl.float32)

    # Compute sum_exp for softmax across tokens
    sum_exp = 0.0  # float32

    t = 0
    while t < num_tokens:
        # Load q[b, h, :]
        q_vec = tl.load(q_ptr + q_base + tl.arange(0, D))
        # Load kv index for this token
        idx_t = tl.load(kv_indices_ptr + t)
        # GQA mapping: kv head index for this output head
        kv_head = h // gqa_ratio  # gqa_ratio = Hq // Hk = 4

        # Compute base offsets for k and v for this token index
        k_base = idx_t * Hk * D + kv_head * D
        v_base = idx_t * Hk * D + kv_head * D

        # Load k[token, kv_head, :] and v[token, kv_head, :]
        k_vec = tl.load(k_ptr + k_base + tl.arange(0, D))
        v_vec = tl.load(v_ptr + v_base + tl.arange(0, D))

        # Dot product
        logit = tl.sum(q_vec * k_vec, axis=0)  # scalar
        logit_scaled = logit * sm_scale

        # Update sum_exp
        exp_val = tl.exp(logit_scaled)
        sum_exp = sum_exp + exp_val

        # Accumulate output
        attn = exp_val / sum_exp
        out = out + attn * v_vec

        t += 1

    # Store output[b, h, :] in bfloat16
    out_base = (b * Hq + h) * D
    out_bf16 = out.to(tl.bfloat16)
    tl.store(out_ptr + out_base + tl.arange(0, D), out_bf16)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, Hq, D] (B batch, Hq=32, D=128) in bfloat16
        k_cache: [num_pages, 1, Hk, D] (Hk=8) in bfloat16
        v_cache: [num_pages, 1, Hk, D] in bfloat16
        kv_indptr: [B+1] int32 (original code doesn't use it; we keep for signature)
        kv_indices: [num_tokens] int32, num_tokens == kv_indices.shape[0]
        sm_scale: float32 scalar (baseline ignores it, but we keep it for signature compatibility)
        Returns: (output [B, Hq, D] bfloat16, lse [B, Hq] float32)
        """
        # Ensure contiguity and dtypes
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indices = kv_indices.contiguous()

        B, Hq, D = q.shape
        assert Hq == 32, "Expected num_qo_heads == 32"
        assert D == 128, "Expected head_dim == 128"
        Hk = k_cache.shape[2]  # number of kv heads, expected 8
        assert Hk == 8, "Expected num_kv_heads == 8"

        # Output tensor in bfloat16
        output = torch.empty((B, Hq, D), dtype=torch.bfloat16, device=q.device)

        # Triton grid: (B, Hq)
        grid = (B, Hq)

        # Launch Triton kernel to accumulate output. Compute in float32 for stability.
        q_f32 = q.float()
        k_f32 = k_cache.float()
        v_f32 = v_cache.float()

        _accumulate_output_kernel[grid](
            q_ptr=q_f32,                # q in float32
            k_ptr=k_f32,                # k in float32
            v_ptr=v_f32,                # v in float32
            out_ptr=output,             # output in bfloat16
            kv_indices_ptr=kv_indices,  # int32 indices
            Hq=32,                      # num output heads
            Hk=8,                       # num kv heads
            gqa_ratio=4,                # Hq // Hk
            num_tokens=kv_indices.numel(),  # runtime int
            sm_scale=float(sm_scale),   # scalar float
            B=B,                        # runtime int
            D=D,                        # runtime int
            num_warps=4,                # heuristic
            num_stages=1,
        )

        # Compute lse per (b, h) using torch: logits_scaled = q[b,h,:] · k[token, kv_head, :] * sm_scale
        # We recompute logits_scaled vector to get logsumexp. This is lightweight compared to output accumulation.
        device = q.device
        # For each (b, h), compute logits_scaled and lse
        lse = torch.full((B, Hq), float("-inf"), dtype=torch.float32, device=device)
        for b_idx in range(B):
            for h_idx in range(Hq):
                kv_head = h_idx // 4  # GQA mapping since gqa_ratio=4
                # Compute logits_scaled for this (b, h)
                logits_scaled = torch.tensor([], device=device, dtype=torch.float32)
                for t in range(kv_indices.numel()):
                    idx_t = int(kv_indices[t].item())
                    k_vec = k_cache[idx_t, 0, kv_head].float()  # [D]
                    # q_vec for this head
                    q_vec = q[b_idx, h_idx].float()  # [D]
                    dot = torch.dot(q_vec, k_vec)  # scalar
                    logits_scaled = torch.cat([logits_scaled, torch.tensor([dot * float(sm_scale)], device=device, dtype=torch.float32)])
                # lse for this (b,h)
                lse[b_idx, h_idx] = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)

        return output, lse


def run(*args):
    return ModelNew()(*args)
