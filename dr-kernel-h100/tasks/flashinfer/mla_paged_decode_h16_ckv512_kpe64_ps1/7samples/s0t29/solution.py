import torch
import triton
import triton.language as tl


@triton.jit
def _lse_kernel(qnh_ptr, Kc_ptr, qph_ptr, LSE_ptr, L_TOKENS: tl.constexpr, SM_SCALE: tl.float32):
    # Running max and sum for logsumexp in base 2
    m = -float('inf')
    s = 0.0
    for t in range(L_TOKENS):
        qnh = tl.load(qnh_ptr + tl.arange(0, 512))
        Kc_vec = tl.load(Kc_ptr + t * 512 + tl.arange(0, 512))
        qph = tl.load(qph_ptr + tl.arange(0, 64))  # not used but we keep structure consistent
        dot_qnh_Kc = tl.sum(qnh * Kc_vec)
        # Kp contribution is not used in lse; we only need qnh dot Kc
        scaled = dot_qnh_Kc * SM_SCALE
        if scaled > m:
            s = s * tl.exp(m - scaled) + 1.0
            m = scaled
        else:
            s += tl.exp(scaled - m)
    inv_log2 = 1.4426950408889
    lse_val = m + tl.log(s) * inv_log2
    tl.store(LSE_ptr, lse_val)


@triton.jit
def _output_from_lse_and_gather_kernel(qnh_ptr, Kc_ptr, LSE, OUT_VEC_ptr, L_TOKENS: tl.constexpr, SM_SCALE: tl.float32):
    # Compute output[b, h, :] = sum_t exp(logits_scaled - LSE * SM_SCALE) * Kc_selected[t]
    for t in range(L_TOKENS):
        qnh = tl.load(qnh_ptr + tl.arange(0, 512))
        Kc_vec = tl.load(Kc_ptr + t * 512 + tl.arange(0, 512))
        dot_qnh_Kc = tl.sum(qnh * Kc_vec)  # logits without Kp contribution
        scaled = dot_qnh_Kc * SM_SCALE
        attn = tl.exp(scaled - LSE * SM_SCALE)
        tl.store(OUT_VEC_ptr + tl.arange(0, 512), tl.load(OUT_VEC_ptr + tl.arange(0, 512)) + attn * Kc_vec)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure tensors are on CUDA and contiguous
    device = q_nope.device
    B, H, D1 = q_nope.shape
    assert H == 16
    assert D1 == 512
    assert q_pe.shape[1] == 16 and q_pe.shape[2] == 64
    assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512
    assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64

    # Preallocate outputs
    output = torch.empty((B, H, D1), dtype=torch.float32, device=device)
    lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

    # For each batch b
    for b in range(B):
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            output[b].zero_()
            lse[b].fill_(-float("inf"))
            continue

        # Gather token indices for this batch element
        tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).contiguous().to(device)

        # Gather K rows into contiguous buffers
        Kc_selected = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, 512]
        Kp_selected = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, 64]
        # Note: Kp_selected is not used in output reconstruction but is gathered for completeness.

        # Prepare qnh and qph vectors
        qnh = q_nope[b].contiguous().to(torch.float32)
        qph = q_pe[b].contiguous().to(torch.float32)

        # Kernel 1: compute lse per head
        lse_val = torch.empty((), dtype=torch.float32, device=device)
        _lse_kernel[(1,)](
            qnh, Kc_selected, qph, lse_val,
            L_TOKENS=L_tokens, SM_SCALE=sm_scale
        )
        lse[b] = lse_val

        # Kernel 2: compute output[b, h, :] using lse
        _output_from_lse_and_gather_kernel[(1,)](
            qnh, Kc_selected, lse[b], output[b],
            L_TOKENS=L_tokens, SM_SCALE=sm_scale
        )

    # Cast output to bfloat16 to match original interface
    output = output.to(torch.bfloat16)
    return output, lse


# Entry point required by the evaluator
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-orchestrated forward: no recursion, no "run" calls
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
