import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_base2_kernel(
    qnh_ptr, qph_ptr,
    Kc_ptr, Kp_ptr,
    lse_ptr,
    sm_scale: tl.constexpr,
    D: tl.constexpr,  # head_dim_ckv = 512
    DP: tl.constexpr,  # head_dim_kpe = 64
    L_TOKENS: tl.constexpr,
):
    # One program per head h; h is implicit since grid is (num_qo_heads,).
    # We directly read qnh_ptr and qph_ptr; lse_ptr is scalar per head.
    # Compute logits vector for all tokens and lse.
    logits = tl.zeros((L_TOKENS,), dtype=tl.float32)
    # Loop over tokens; Triton will unroll due to constexpr
    for t in tl.static_range(L_TOKENS):
        # Load Kc row t: offset t * D + i for i in [0..D-1] -> use t*D + tl.arange(0, D)
        Kc_row = tl.load(Kc_ptr + t * D + tl.arange(0, D))
        # Load qnh vector: tl.load(qnh_ptr + tl.arange(0, D))
        qnh_vec = tl.load(qnh_ptr + tl.arange(0, D))
        dot_qnh = tl.sum(Kc_row * qnh_vec, axis=0)
        # Load Kp row t: offset t * DP + j for j in [0..DP-1]
        Kp_row = tl.load(Kp_ptr + t * DP + tl.arange(0, DP))
        qph_vec = tl.load(qph_ptr + tl.arange(0, DP))
        dot_qph = tl.sum(Kp_row * qph_vec, axis=0)
        logits[t] = dot_qnh + dot_qph
    # Apply scaling
    logits_scaled = logits * sm_scale
    m = tl.max(logits_scaled, axis=0)
    # Numerical stability: subtract m*sm_scale
    shifted = logits_scaled - m
    exp_vals = tl.exp(shifted)
    s = tl.sum(exp_vals, axis=0)
    lse_val = m + tl.log(s) / 0.6931471805599453  # log2(e)
    tl.store(lse_ptr, lse_val)


@triton.jit
def attention_softmax_kernel(
    qnh_ptr, qph_ptr,
    Kc_ptr, Kp_ptr,
    attn_ptr,
    lse_val,  # scalar float32
    sm_scale: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    L_TOKENS: tl.constexpr,
):
    # Compute logits, softmax, and store attn[t] for all tokens
    logits = tl.zeros((L_TOKENS,), dtype=tl.float32)
    for t in tl.static_range(L_TOKENS):
        Kc_row = tl.load(Kc_ptr + t * D + tl.arange(0, D))
        qnh_vec = tl.load(qnh_ptr + tl.arange(0, D))
        dot_qnh = tl.sum(Kc_row * qnh_vec, axis=0)
        Kp_row = tl.load(Kp_ptr + t * DP + tl.arange(0, DP))
        qph_vec = tl.load(qph_ptr + tl.arange(0, DP))
        dot_qph = tl.sum(Kp_row * qph_vec, axis=0)
        logits[t] = dot_qnh + dot_qph
    logits_scaled = logits * sm_scale
    m = tl.max(logits_scaled, axis=0)
    shifted = logits_scaled - m
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals, axis=0)
    # Write attn[t] = exp(logits_scaled[t] - m) / sum_exp
    for t in tl.static_range(L_TOKENS):
        attn_t = exp_vals[t] / sum_exp
        tl.store(attn_ptr + t, attn_t)


@triton.jit
def attention_output_kernel(
    attn_ptr,  # 1D buffer [L_TOKENS]
    Kc_ptr,    # [L_TOKENS, D]
    out_ptr,   # [D]
    D: tl.constexpr,
    L_TOKENS: tl.constexpr,
):
    # Accumulate output vector: out = sum_t attn[t] * Kc[t]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in tl.static_range(L_TOKENS):
        attn_t = tl.load(attn_ptr + t)  # scalar
        Kc_row = tl.load(Kc_ptr + t * D + tl.arange(0, D))  # [D]
        out_vec += attn_t * Kc_row
    tl.store(out_ptr + tl.arange(0, D), out_vec)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-optimized forward:
    - Computes output [B, 16, 512] and lse [B, 16] using Triton kernels.
    """
    device = q_nope.device  # ensure CUDA device
    B = q_nope.shape[0]
    H = q_nope.shape[1]
    D = q_nope.shape[2]  # 512
    DP = q_pe.shape[2]   # 64
    # Prepare outputs
    output = torch.zeros((B, H, D), dtype=torch.float32, device=device)
    lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

    for b in range(B):
        # Compute L_tokens and gather token indices
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            # No KV entries for this batch, output zeros and lse remains -inf
            continue

        tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(device).to(torch.int32)

        # Gather cached keys (cast to float32 for compute)
        Kc_selected = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, D]
        Kp_selected = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, DP]

        # Ensure q_nope[b] and q_pe[b] are float32
        qnh = q_nope[b].contiguous().to(torch.float32)  # [H, D]
        qph = q_pe[b].contiguous().to(torch.float32)    # [H, DP]
        # We will process each head h: grid is (H,)
        for h in range(H):
            # Prepare pointers for this head
            qnh_ptr = qnh[h]  # [D]
            qph_ptr = qph[h]  # [DP]

            # Launch lse_base2_kernel: compute lse[b, h]
            lse[b, h] = lse_base2_kernel[(1,)](
                qnh_ptr, qph_ptr,
                Kc_selected, Kp_selected,
                lse[b, h],
                sm_scale,
                D=D, DP=DP, L_TOKENS=L_tokens,
                num_warps=4,
            )

            # Allocate attn buffer [L_tokens]
            attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
            # Launch attention_softmax_kernel to compute attn
            attn = attention_softmax_kernel[(1,)](
                qnh_ptr, qph_ptr,
                Kc_selected, Kp_selected,
                attn,
                lse[b, h],
                sm_scale,
                D=D, DP=DP, L_TOKENS=L_tokens,
                num_warps=4,
            )

            # Launch attention_output_kernel to compute output[b, h, :]
            out_vec = torch.empty(D, dtype=torch.float32, device=device)
            attention_output_kernel[(1,)](
                attn, Kc_selected, out_vec,
                D=D, L_TOKENS=L_tokens,
                num_warps=1,
            )
            # Store the output vector
            output[b, h, :] = out_vec

    # Cast output to bfloat16 as original returns
    output = output.to(torch.bfloat16)
    return output, lse


# Optional: provide get_inputs that returns CUDA tensors
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    num_pages = 989669
    ckv_cache = torch.randn([num_pages, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([num_pages, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, num_pages, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Entry point required by the evaluator
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA device; Triton requires CUDA tensors
        if not q_nope.is_cuda:
            q_nope = q_nope.to('cuda')
        if not q_pe.is_cuda:
            q_pe = q_pe.to('cuda')
        if not ckv_cache.is_cuda:
            ckv_cache = ckv_cache.to('cuda')
        if not kpe_cache.is_cuda:
            kpe_cache = kpe_cache.to('cuda')
        if not kv_indptr.is_cuda:
            kv_indptr = kv_indptr.to('cuda')
        if not kv_indices.is_cuda:
            kv_indices = kv_indices.to('cuda')

        # Run with Triton kernels
        output, lse = run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
        return output, lse