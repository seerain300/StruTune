import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Single Triton kernel: per head h, compute:
# - lse = logsumexp_base2 over logits = (qnh @ Kc_selected.T) + (qph @ Kp_selected.T)
# - output = sum_t softmax(logits * sm_scale)[:, None] * Kc_selected[t, :]
# It assumes D = head_dim_ckv is a tl.constexpr meta-parameter.
@triton.jit
def _compute_lse_and_output_kernel(
    qnh_ptr, qph_ptr, Kc_ptr, Kp_ptr,
    out_vec_ptr, lse_ptr,
    L_TOKENS: tl.constexpr, D: tl.constexpr, sm_scale: tl.float32
):
    # Vector for feature dimension
    idx = tl.arange(0, D)

    # Load qnh and qph
    qnh = tl.load(qnh_ptr + idx)
    qph = tl.load(qph_ptr + idx)

    # Initialize vectors for logits and output
    logits = tl.zeros([L_TOKENS], dtype=tl.float32)
    out_vec = tl.zeros([D], dtype=tl.float32)

    # Compute logits for each token t
    for t in tl.static_range(0, L_TOKENS):
        Kc_row_ptr = Kc_ptr + t * D + idx
        Kp_row_ptr = Kp_ptr + t * D + idx
        Kc_row = tl.load(Kc_row_ptr)
        Kp_row = tl.load(Kp_row_ptr)
        logits[t] = tl.sum(qnh * Kc_row, axis=0) + tl.sum(qph * Kp_row, axis=0)

    # Compute lse: logsumexp_base2(logits * sm_scale)
    scaled = logits * sm_scale
    m = tl.max(scaled, axis=0)
    s = tl.sum(tl.exp(scaled - m), axis=0)
    lse = m + tl.log(s) * 1.0 / math.log(2.0)
    tl.store(lse_ptr, lse)

    # Compute softmax and accumulate output
    # Note: recompute scaled for each t to get its softmax contribution
    for t in tl.static_range(0, L_TOKENS):
        Kc_row_ptr = Kc_ptr + t * D + idx
        Kp_row_ptr = Kp_ptr + t * D + idx
        Kc_row = tl.load(Kc_row_ptr)
        Kp_row = tl.load(Kp_row_ptr)
        scaled_t = (tl.sum(qnh * Kc_row, axis=0) + tl.sum(qph * Kp_row, axis=0)) * sm_scale
        # attn[t] = exp(scaled_t - lse)
        attn_t = tl.exp(scaled_t - lse)
        # Normalize attn_t by the same denominator we computed via s (since lse is log(denom)/log(2))
        # s = sum exp(scaled) = sum exp(scaled - lse + lse) = sum exp(scaled - lse + lse) -> we can normalize using s
        # But here we can directly use s computed above:
        norm = tl.log(s) * 1.0 / math.log(2.0)  # not needed, we normalize using s directly
        # To normalize, we need denominator = sum_t exp(scaled_t - lse) = s (since lse is log(denom)/log(2)).
        # Therefore: attn_t = attn_t * 1 / s. But s is per row vector? No: s is scalar for logits. Recompute denom for this batch:
        # Simpler: compute denom = sum exp(scaled - lse) for all t. Since we already have s from logits, and scaled_t differs by constants,
        # we can use s as the sum for this head. However, s was computed from logits, not from scaled_t. To fix, recompute denom_t = sum exp(scaled - lse).
        # For small L_TOKENS, this is fine: we loop and compute denom.
        denom = tl.zeros([], dtype=tl.float32)
        for u in tl.static_range(0, L_TOKENS):
            scaled_u = (tl.sum(qnh * Kc_ptr + u * D + idx, axis=0) + tl.sum(qph * Kp_ptr + u * D + idx, axis=0)) * sm_scale
            denom += tl.exp(scaled_u - lse)
        attn_t = attn_t / denom
        out_vec += attn_t * Kc_row

    # Store output vector
    tl.store(out_vec_ptr, out_vec)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-orchestrated forward that computes:
      - output: [B, 16, 512] float32, then cast to bfloat16
      - lse:    [B, 16] float32
    Assumes Triton is available. All heavy math is in Triton kernels.
    """
    assert TRITON_AVAILABLE, "Triton is required but not available"
    device = q_nope.device
    B = q_nope.shape[0]
    num_heads = q_nope.shape[1]
    D = q_nope.shape[2]
    assert D == 512, "head_dim_ckv must be 512 for this implementation"

    # Output and lse buffers
    output = torch.empty((B, num_heads, D), dtype=torch.float32, device=device)
    lse = torch.full((B, num_heads), float("-inf"), dtype=torch.float32, device=device)

    # Ensure inputs are contiguous and float32 for compute
    q_nope_f32 = q_nope.to(torch.float32)
    q_pe_f32 = q_pe.to(torch.float32)
    ckv_cache_f32 = ckv_cache.to(torch.float32)
    kpe_cache_f32 = kpe_cache.to(torch.float32)
    kv_indptr_i32 = kv_indptr.to(torch.int32)
    kv_indices_i32 = kv_indices.to(torch.int32)

    # For each batch and head, process tokens if any
    for b in range(B):
        # Compute L_tokens and indices
        L_tokens = int(kv_indptr_i32[b + 1].item() - kv_indptr_i32[b].item())
        if L_tokens <= 0:
            # No KV entries for this batch; set lse to -inf and output zeros
            lse[b, :] = -float("inf")
            output[b] = 0.0
            continue

        tok_idx = kv_indices_i32[kv_indptr_i32[b]: kv_indptr_i32[b + 1]]
        # Gather selected keys
        Kc_selected = ckv_cache_f32[tok_idx]  # [L_tokens, D]
        Kp_selected = kpe_cache_f32[tok_idx]  # [L_tokens, D_kp] but we don't use D_kp here since qph is 64, so we only need Kp_row (64)

        # For Triton, we pass pointers to qnh, qph, and per-row pointers. Since Kp is 64-dim, we only need first 64 dims of Kp; however, Kp is irrelevant for output accumulation, so we pass Kp as Kc_selected but only use qph*Kp_selected. To keep correctness, we pass Kp_selected as provided.

        # Launch Triton kernel per (b, h)
        for h in range(num_heads):
            qnh_ptr = q_nope_f32[b, h]  # [D]
            qph_ptr = q_pe_f32[b, h]    # [64]
            # Kc_selected rows are contiguous per token; pass as 1D pointers
            # Kp_selected similarly, though not needed for final output vector accumulation (we only use qph*Kp_selected to compute logits).
            out_vec_ptr = output[b, h]
            lse_ptr = lse[b, h]
            _compute_lse_and_output_kernel[(1,)](
                qnh_ptr, qph_ptr,
                Kc_selected, Kp_selected,
                out_vec_ptr, lse_ptr,
                L_TOKENS=L_tokens, D=D, sm_scale=sm_scale,
                num_warps=1
            )

    # Cast output to bfloat16 as in original
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


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


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton kernels are invoked from ModelNew.forward; no recursion
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)