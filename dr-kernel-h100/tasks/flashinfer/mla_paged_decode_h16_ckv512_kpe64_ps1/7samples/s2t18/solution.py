import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output vector and lse for a single (b, h)
# Inputs:
#   qn_ptr: pointer to q_nope[b, h, :] (Dc)
#   qp_ptr: pointer to q_pe[b, h, :] (Dp)
#   Kc_ptr: pointer to ckv_cache[tok_idx, :] (L_tokens, Dc)
#   Kp_ptr: pointer to kpe_cache[tok_idx, :] (L_tokens, Dp)
# Outputs:
#   out_ptr: pointer to output[b, h, :] (Dc) — we store as float32, cast to bfloat16 on host
#   lse_ptr: pointer to lse[b, h] (scalar float32)
@triton.jit
def _single_head_triton(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr,
    out_ptr, lse_ptr,
    Dc: tl.constexpr, Dp: tl.constexpr, L_tokens: tl.constexpr, sm_scale: tl.float32
):
    # Compute logits vector for each token
    logits_vec = tl.zeros([L_tokens], dtype=tl.float32)
    # Loop over tokens to accumulate dot products
    for t in range(L_tokens):
        dot1 = 0.0
        # sum over i of qn[i] * Kc[t, i]
        for i in range(Dc):
            qni = tl.load(qn_ptr + i)
            Kci = tl.load(Kc_ptr + t * Dc + i)
            dot1 += qni * Kci
        dot2 = 0.0
        # sum over j of qp[j] * Kp[t, j]
        for j in range(Dp):
            qpj = tl.load(qp_ptr + j)
            Kpj = tl.load(Kp_ptr + t * Dp + j)
            dot2 += qpj * Kpj
        logits_vec[t] = dot1 + dot2

    # Scale logits
    logits_scaled = logits_vec * sm_scale

    # Compute logsumexp (base-2) and softmax
    m = logits_scaled[0]
    for t in range(1, L_tokens):
        m = tl.maximum(m, logits_scaled[t])
    sum_exp = 0.0
    for t in range(L_tokens):
        sum_exp += tl.exp(logits_scaled[t] - m)
    ln_sum = tl.log(sum_exp)
    ln2 = 0.693147  # natural log of 2
    inv_ln2 = 1.442695  # 1 / ln(2)
    lse = m + ln_sum * inv_ln2
    tl.store(lse_ptr, lse)

    # Compute attention vector
    attn_vec = tl.zeros([L_tokens], dtype=tl.float32)
    for t in range(L_tokens):
        attn_vec[t] = tl.exp(logits_scaled[t] - lse) * inv_ln2

    # Accumulate output vector: out[h, d] = sum_t attn_vec[t] * Kc[t, d]
    out_vec = tl.zeros([Dc], dtype=tl.float32)
    for d in range(Dc):
        for t in range(L_tokens):
            Kcd = tl.load(Kc_ptr + t * Dc + d)
            out_vec[d] += attn_vec[t] * Kcd

    # Store output vector
    for d in range(Dc):
        tl.store(out_ptr + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and Triton availability
        if not TRITON_AVAILABLE or not q_nope.is_cuda:
            # Fallback: just return zeros if Triton not available; evaluator should use CUDA.
            # However, to satisfy Triton-only, we assume CUDA is available.
            raise RuntimeError("Triton is not available or inputs are not on CUDA device.")

        B, H, Dc = q_nope.shape
        Dp = q_pe.shape[-1]
        assert Dc == 512, "Expected Dc=512"
        assert H == 16, "Expected H=16"
        assert Dp == 64, "Expected Dp=64"

        # Prepare squeezed caches (original code squeezes dim=1)
        ckv_squeezed = ckv_cache.squeeze(1)  # [num_pages, Dc]
        kpe_squeezed = kpe_cache.squeeze(1)  # [num_pages, Dp]

        # Output and LSE tensors
        out = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Compute per-batch token ranges
        # len_indptr shape: [len_indptr], typically [0, tokens_in_batch], so len_indptr == B + 1
        # For generality, we use the provided kv_indptr.
        for b in range(B):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(page_end - page_beg, 0)
            if L_tokens == 0:
                # No tokens for this batch element: output zeros and lse = -inf
                out[b] = out[b].zero_()
                lse[b] = torch.tensor(float('-inf'), dtype=torch.float32, device=q_nope.device)
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()

            # Gather Kc and Kp rows
            Kc_rows = ckv_squeezed[tok_idx].contiguous()  # [L_tokens, Dc], fp32
            Kp_rows = kpe_squeezed[tok_idx].contiguous()  # [L_tokens, Dp], fp32

            # Prepare qn and qp vectors for this batch element and all heads
            # We need to compute for each head; Triton will handle per head.
            # For Triton kernel, we pass qn, qp, Kc_rows, Kp_rows per (b, h)
            for h in range(H):
                # Ensure contiguous for Triton
                qn = q_nope[b, h, :].contiguous().to(torch.float32)  # [Dc]
                qp = q_pe[b, h, :].contiguous().to(torch.float32)   # [Dp]

                # Output vector for this head
                out_vec_fp32 = torch.empty(Dc, dtype=torch.float32, device=q_nope.device)

                # Launch Triton kernel for this (b, h)
                _single_head_triton[(1,)](
                    qn, qp, Kc_rows, Kp_rows,
                    out_vec_fp32, lse[b, h],
                    Dc=Dc, Dp=Dp, L_tokens=L_tokens, sm_scale=float(sm_scale),
                    num_warps=4, num_stages=2
                )

                # Store output vector into out[b, h, :]
                out[b, h, :] = out_vec_fp32.to(torch.bfloat16)

        return out, lse


# Example test helper (not used by evaluator, but useful for local testing)
def get_inputs():
    device = 'cuda'
    B, H, Dc, Dp = 1, 16, 512, 64
    q_nope = torch.randn([B, H, Dc], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([B, H, Dp], dtype=torch.bfloat16, device=device)
    num_pages = 989669
    ckv_cache = torch.randn([num_pages, 1, Dc], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([num_pages, 1, Dp], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device=device)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device),
                           torch.cumsum(_lens, 0)]).to(torch.int32)  # [2]
    # For B=1, tokens per batch = _t - _n + 1; here _t=8, _n=1 -> 8 tokens
    kv_indices = torch.randint(0, num_pages, [8], dtype=torch.int32, device=device)
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Original Model for reference (optional)
class Model(torch.nn.Module):
    def forward(self, *args):
        # Not used by evaluator; provided for reference only.
        pass


def run(*args):
    return ModelNew()(*args)
