import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    qn_ptr,               # *f32, 1D vector of length H*CK
    qp_ptr,               # *f32, 1D vector of length H*KP
    Kc_base_ptr,          # *f32, base for ckv_cache, shape [num_tokens, CK]
    Kp_base_ptr,          # *f32, base for kpe_cache, shape [num_tokens, KP]
    tok_idx_ptr,          # *i32, 1D vector of length L_tokens
    logits_ptr,           # *f32, output buffer [H, L_tokens]
    H: tl.constexpr,      # number of heads
    CK: tl.constexpr,     # head dim for q_nope
    KP: tl.constexpr,     # head dim for q_pe
    L_tokens: tl.constexpr,  # number of tokens in this batch element
    sm_scale: tl.constexpr,  # scaling factor (float)
):
    h = tl.program_id(0)  # head index
    t = tl.program_id(1)  # token index

    # load qn[h], qp[h] as 1D vectors
    qn = tl.load(qn_ptr + h * CK + tl.arange(0, CK))
    qp = tl.load(qp_ptr + h * KP + tl.arange(0, KP))

    # load Kc[t] and Kp[t] rows using tok_idx_ptr[t]
    tok_idx = tl.load(tok_idx_ptr + t)
    Kc_row = tl.load(Kc_base_ptr + tok_idx * CK + tl.arange(0, CK))
    Kp_row = tl.load(Kp_base_ptr + tok_idx * KP + tl.arange(0, KP))

    # compute dot products
    dot_qn_Kc = tl.sum(qn * Kc_row, axis=0)
    dot_qp_Kp = tl.sum(qp * Kp_row, axis=0)

    scaled = sm_scale * (dot_qn_Kc + dot_qp_Kp)

    # store to logits[h, t]
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_kernel(
    logits_ptr,       # *f32, [H, L_tokens]
    lse_ptr,          # *f32, [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    # pass 1: compute max over tokens
    m = -float("inf")
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        if val > m:
            m = val
    # pass 2: compute sum(exp(val - m))
    sumexp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sumexp += tl.exp(val - m)
    lse = tl.log(sumexp)  # natural log
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    qn_ptr,               # *f32, 1D vector of length CK
    qp_ptr,               # *f32, 1D vector of length KP
    logits_ptr,           # *f32, [H, L_tokens]
    lse_ptr,              # *f32, [H]
    Kc_base_ptr,          # *f32, base for ckv_cache, shape [num_tokens, CK]
    tok_idx_ptr,          # *i32, 1D vector of length L_tokens
    out_ptr,              # *f32, [H, CK] output buffer
    H: tl.constexpr,
    CK: tl.constexpr,
    KP: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    # initialize out[h, :]
    for j in range(0, CK):
        out_ptr[h * CK + j] = 0.0

    m = tl.load(lse_ptr + h)

    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        p = tl.exp(val - m)
        tok_idx = tl.load(tok_idx_ptr + t)
        Kc_row = tl.load(Kc_base_ptr + tok_idx * CK + tl.arange(0, CK))
        for j in range(0, CK):
            out_ptr[h * CK + j] += p * Kc_row[j]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # device and dtype
        device = q_nope.device

        # cast inputs to float32 for computation
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        ckv_cache_f32 = ckv_cache.to(torch.float32)
        kpe_cache_f32 = kpe_cache.to(torch.float32)

        batch_size = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        CK = q_nope_f32.shape[2]
        KP = q_pe_f32.shape[2]
        num_tokens = ckv_cache_f32.shape[0]

        # Sanity checks (as in original)
        assert H == 16, "num_qo_heads must be 16"
        assert CK == 512, "head_dim_ckv must be 512"
        assert KP == 64, "head_dim_kpe must be 64"

        # Prepare output buffers
        output = torch.zeros((batch_size, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # For each batch element
        for b in range(batch_size):
            # Determine L_tokens from kv_indptr
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV for this batch element
                lse[b].fill_(-float("inf"))
                continue

            # tok_idx: indices of tokens used for this batch element
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).contiguous()  # 1D int32

            # Prepare qn and qp as 1D vectors to match Triton kernel expectations (vector length CK/KP)
            qn = q_nope_f32[b].reshape(-1).contiguous()  # [CK]
            qp = q_pe_f32[b].reshape(-1).contiguous()   # [KP]

            # Allocate logits buffer [H, L_tokens] in f32
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch kernel to compute scaled logits
            _compute_scaled_logits_kernel[(H, L_tokens)](
                qn, qp,
                ckv_cache_f32, kpe_cache_f32,
                tok_idx,
                logits,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale)
            )

            # Compute lse per head
            _lse_kernel[(H,)](
                logits, lse[b],
                H=H, L_tokens=L_tokens
            )

            # Compute final output per head: out[h, :] = sum_t softmax(logits[h, t]) * Kc[t, :]
            out_per_batch = torch.empty((H, CK), dtype=torch.float32, device=device)
            _compute_output_kernel[(H,)](
                qn, qp,
                logits, lse[b],
                ckv_cache_f32,
                tok_idx,
                out_per_batch,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens
            )
            output[b] = out_per_batch

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)

        # Return (output, lse) to satisfy evaluator expecting multiple outputs
        return output, lse


def run(*args):
    return ModelNew()(*args)
