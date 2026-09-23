import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute sum_exp = sum_t exp(logits[b,h,t]) over tokens for each (b,h)
# We don't use sm_scale (baseline ignores it); we keep it accepted but unused.
@triton.jit
def _lse_kernel(
    q_ptr,             # *f32, shape [B*Hq*D] contiguous
    k_ptr,             # *f32, shape [num_tokens*Hk*D] contiguous
    out_lse_ptr,       # *f32, shape [B*Hq]
    num_tokens: tl.constexpr,
    MAX_TOKS: tl.constexpr,
    Hq: tl.constexpr,
    D: tl.constexpr,
    Hk: tl.constexpr,
    gqa_ratio: tl.constexpr,
    B: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    sum_exp = 0.0
    t = 0
    while t < MAX_TOKS:
        if t >= num_tokens:
            t += 1
        # Compute dot(q[b,h,:], k[t, kv_head, :])
        kv_head = h // gqa_ratio
        dot = 0.0
        d = 0
        while d < D:
            q_off = (b * Hq + h) * D + d
            q_val = tl.load(q_ptr + q_off)
            k_off = t * (Hk * D) + kv_head * D + d
            k_val = tl.load(k_ptr + k_off)
            dot += q_val * k_val
            d += 1
        sum_exp += tl.exp(dot)  # baseline ignores sm_scale, so dot = logits
        t += 1

    # lse = log(sum_exp) / ln(2)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)
    tl.store(out_lse_ptr + (b * Hq + h), lse_val)


# Triton kernel: accumulate output[b,h,:] over tokens:
# For each token t, add attn * v[t, kv_head, :] to out_output[b*Hq*128 + h*128]
@triton.jit
def _accumulate_output_kernel(
    q_ptr,               # *f32, shape [B*Hq*D] contiguous
    k_ptr,               # *f32, shape [num_tokens*Hk*D] contiguous
    v_ptr,               # *f32, shape [num_tokens*Hk*D] contiguous
    out_output_ptr,      # *f32, shape [B*Hq*D] contiguous (flattened)
    out_lse_ptr,         # *f32, shape [B*Hq]
    num_tokens: tl.constexpr,
    MAX_TOKS: tl.constexpr,
    Hq: tl.constexpr,
    D: tl.constexpr,
    Hk: tl.constexpr,
    gqa_ratio: tl.constexpr,
    B: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load sum_exp = exp(lse * ln(2)) for normalization
    lse_val = tl.load(out_lse_ptr + (b * Hq + h))
    sum_exp = tl.exp(lse_val * 1.4426950408889634)

    # Accumulate output across tokens
    out_base = (b * Hq + h) * D
    t = 0
    while t < MAX_TOKS:
        if t >= num_tokens:
            t += 1
        kv_head = h // gqa_ratio
        # Recompute dot = q[b,h,:] · k[t, kv_head, :]
        dot = 0.0
        d = 0
        while d < D:
            q_off = (b * Hq + h) * D + d
            q_val = tl.load(q_ptr + q_off)
            k_off = t * (Hk * D) + kv_head * D + d
            k_val = tl.load(k_ptr + k_off)
            dot += q_val * k_val
            d += 1
        attn = tl.exp(dot) / sum_exp  # baseline ignores sm_scale

        # Add attn * v[t, kv_head, :] to out_output[b,h,:]
        v_off = t * (Hk * D) + kv_head * D
        d2 = 0
        while d2 < D:
            v_val = tl.load(v_ptr + v_off + d2)
            tl.store(out_output_ptr + out_base + d2, tl.load(out_output_ptr + out_base + d2) + attn * v_val)
            d2 += 1
        t += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # We accept 6 inputs as in original: (q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale)
        # The original run ignores sm_scale; we keep it for signature compatibility but do not use it.

        # Ensure inputs are on a device where Triton can run
        device = q.device
        assert device.type == "cuda", "This Triton implementation requires CUDA tensors."
        assert TRITON_AVAILABLE, "Triton is not available."

        # Shapes and constants (match original assumptions)
        B, Hq, D = q.shape
        _, num_pages, Hk, _ = k_cache.shape
        assert Hq == 32, "num_qo_heads must be 32"
        assert Hk == 8, "num_kv_heads must be 8"
        assert D == 128, "head_dim must be 128"

        # Prepare flattened pointers in float32
        q_f32 = q.to(torch.float32).contiguous()                 # [B, 32, 128]
        k_f32 = k_cache.to(torch.float32).contiguous()          # [num_pages, 1, 8, 128]
        v_f32 = v_cache.to(torch.float32).contiguous()          # [num_pages, 1, 8, 128]

        # Compute num_tokens for this batch (assumes kv_indptr[0] == 0 and kv_indptr[B] has total tokens)
        # The original run enforces num_tokens == kv_indices.shape[0]; we do the same here.
        num_tokens = kv_indices.numel()
        assert kv_indptr[-1].item() - kv_indptr[0].item() == num_tokens, "kv_indptr and kv_indices mismatch."

        # Prepare a buffer for lse[B*Hq]
        out_lse = torch.empty((B, Hq), dtype=torch.float32, device=device)

        # Output buffer [B, 32, 128] in float32 (baseline returns float tensors)
        out_output = torch.zeros((B, Hq, D), dtype=torch.float32, device=device)

        # Launch Triton kernels: grid over (B, Hq)
        grid = (B, Hq)
        # Choose MAX_TOKS as an upper bound; here we can set it to a reasonably large value like 1024.
        # However, we can set it equal to the actual num_tokens to reduce unnecessary iterations.
        # Since Triton requires compile-time for loop bounds, we pass MAX_TOKS = num_tokens (it's fine as constexpr for small sizes).
        MAX_TOKS = int(num_tokens)

        # Kernel 1: compute lse per (b, h)
        _lse_kernel[grid](
            q_f32, k_f32, out_lse,
            num_tokens=num_tokens,
            MAX_TOKS=MAX_TOKS,
            Hq=Hq,
            D=D,
            Hk=Hk,
            gqa_ratio=Hq // Hk,  # 32 // 8 = 4
            B=B,
        )

        # Kernel 2: accumulate output per (b, h)
        _accumulate_output_kernel[grid](
            q_f32, k_f32, v_f32, out_output.view(-1), out_lse,
            num_tokens=num_tokens,
            MAX_TOKS=MAX_TOKS,
            Hq=Hq,
            D=D,
            Hk=Hk,
            gqa_ratio=Hq // Hk,  # 32 // 8 = 4
            B=B,
        )

        # Return output and lse, matching original run: output is float32, lse is float32
        # Original run returns (output, lse) as a tuple
        return out_output, out_lse


def run(*args):
    return ModelNew()(*args)
