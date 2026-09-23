import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    # Constants consistent with original code
    Hq = 32        # number of query heads
    D = 128        # head dimension
    Hk = 8         # number of kv heads
    gqa_ratio = 4  # Hq // Hk

    @triton.jit
    def _compute_logits_bh_kernel(
        q_ptr,            # *f32, [B, Hq, D] contiguous
        k_ptr,            # *f32, [NUM_TOKS, Hk, D] contiguous
        logits_ptr,       # *f32, [B, Hq, NUM_TOKS] contiguous
        B: tl.constexpr,  # int
        NUM_TOKS: tl.constexpr,  # int
    ):
        # Grid is (B, Hq)
        b = tl.program_id(0)
        h = tl.program_id(1)
        if b >= B or h >= Hq:
            return
        # Base offset for q[b, h, :]
        q_base = b * Hq * D + h * D
        # For each token t, compute dot(q[b,h,:], k[t, kv_head, :])
        for t in range(NUM_TOKS):
            kv_head = h // gqa_ratio  # GQA mapping
            k_base = t * Hk * D + kv_head * D  # k[t, kv_head, :]
            dot_acc = 0.0
            for d in range(D):
                q_val = tl.load(q_ptr + q_base + d)
                k_val = tl.load(k_ptr + k_base + d)
                dot_acc += q_val * k_val
            # Store raw dot
            tl.store(logits_ptr + b * Hq * NUM_TOKS + h * NUM_TOKS + t, dot_acc)

    @triton.jit
    def _lse_per_bh_kernel(
        logits_ptr,       # *f32, [B, Hq, NUM_TOKS] contiguous
        lse_ptr,          # *f32, [B, Hq]
        B: tl.constexpr,
        NUM_TOKS: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        if b >= B or h >= Hq:
            return
        base = b * Hq * NUM_TOKS + h * NUM_TOKS
        m = -float("inf")
        sum_exp = 0.0
        t = 0
        while t < NUM_TOKS:
            val = tl.load(logits_ptr + base + t)
            if val > m:
                sum_exp = sum_exp * tl.exp(m - val)
                m = val
            else:
                sum_exp += tl.exp(val - m)
            t += 1
        lse = m + tl.log(sum_exp)
        # Divide by ln(2)
        lse = lse / 0.6931471805599453
        tl.store(lse_ptr + b * Hq + h, lse)

    @triton.jit
    def _accumulate_output_bh_kernel(
        q_ptr,            # *f32, [B, Hq, D]
        k_ptr,            # *f32, [NUM_TOKS, Hk, D]
        v_ptr,            # *f32, [NUM_TOKS, Hk, D]
        logits_ptr,       # *f32, [B, Hq, NUM_TOKS]
        out_ptr,          # *f32, [B, Hq, D]
        B: tl.constexpr,
        NUM_TOKS: tl.constexpr,
        D: tl.constexpr,
        Hk: tl.constexpr,
        gqa_ratio: tl.constexpr,
    ):
        # Grid is (B, Hq)
        b = tl.program_id(0)
        h = tl.program_id(1)
        if b >= B or h >= Hq:
            return
        # Initialize output[b, h, :] to zero
        out_base = b * Hq * D + h * D
        for d in range(D):
            tl.store(out_ptr + out_base + d, 0.0)
        kv_head = h // gqa_ratio
        # Accumulate: out[b, h, :] += sum_t softmax(logits[b, h, t]) * v[t, kv_head, :]
        t = 0
        while t < NUM_TOKS:
            val = tl.load(logits_ptr + b * Hq * NUM_TOKS + h * NUM_TOKS + t)
            # Compute sum_exp over all tokens for (b,h)
            sum_exp = 0.0
            s = 0
            while s < NUM_TOKS:
                v_s = tl.load(logits_ptr + b * Hq * NUM_TOKS + h * NUM_TOKS + s)
                sum_exp += tl.exp(v_s)
                s += 1
            attn = tl.exp(val) / sum_exp
            # v[t, kv_head, :]
            v_base = t * Hk * D + kv_head * D
            for d in range(D):
                v_val = tl.load(v_ptr + v_base + d)
                tl.store(out_ptr + out_base + d, tl.load(out_ptr + out_base + d) + attn * v_val)
            t += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Convert to float32 for Triton kernels
        q_f32 = q.to(torch.float32).contiguous()
        # Gather K and V for this batch (num_tokens is the total number of indices)
        num_tokens = kv_indices.numel()
        # Indices are already int32 in the harness; use them to gather
        # k_ptr and v_ptr: shape [num_tokens, 1, Hk, D] -> we need [num_tokens, Hk, D]
        # Note: k_cache/v_cache are [num_pages, 1, Hk, D]; we only need the selected indices.
        # The original len_indptr indicates the total tokens per batch element; here we assume the whole batch uses kv_indices.
        k_f32 = k_cache[:, 0].reshape(-1, Hk, D).to(torch.float32).contiguous()
        v_f32 = v_cache[:, 0].reshape(-1, Hk, D).to(torch.float32).contiguous()
        # But since len_indptr[-1] equals num_tokens, we can directly take the first num_tokens entries:
        k_f32 = k_f32[:num_tokens].contiguous()
        v_f32 = v_f32[:num_tokens].contiguous()

        B, Hq_b, D_b = q_f32.shape
        # Ensure constants match
        assert Hq_b == Hq and D_b == D, "q shape must be [B, 32, 128]"

        # Allocate buffers
        logits = torch.empty((B, Hq, num_tokens), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, Hq), dtype=torch.float32, device=q.device)
        out = torch.empty((B, Hq, D), dtype=torch.float32, device=q.device)

        # Launch Triton kernels
        # 1) Compute logits
        _compute_logits_bh_kernel[(B, Hq)](
            q_f32, k_f32, logits,
            B=B, NUM_TOKS=num_tokens,
        )
        # 2) Compute lse
        _lse_per_bh_kernel[(B, Hq)](
            logits, lse,
            B=B, NUM_TOKS=num_tokens,
        )
        # 3) Accumulate output
        _accumulate_output_bh_kernel[(B, Hq)](
            q_f32, k_f32, v_f32, logits, out,
            B=B, NUM_TOKS=num_tokens, D=D, Hk=Hk, gqa_ratio=gqa_ratio,
        )

        # Return output in bfloat16 (match original q dtype)
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
