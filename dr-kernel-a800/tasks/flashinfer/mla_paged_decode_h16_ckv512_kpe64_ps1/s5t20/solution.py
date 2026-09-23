import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    qn_ptr,        # *f32, base pointer to [H, CK]
    qp_ptr,        # *f32, base pointer to [H, KP]
    Kc_all_ptr,    # *f32, base pointer to [P, CK]
    Kp_all_ptr,    # *f32, base pointer to [P, KP]
    tok_idx_ptr,   # *i32, base pointer to [L_tokens]
    logits_ptr,    # *f32, base pointer to [H, L_tokens]
    H: tl.constexpr,            # num_qo_heads
    CK: tl.constexpr,           # head_dim_ckv (512)
    KP: tl.constexpr,           # head_dim_kpe (64)
    L_tokens: tl.constexpr,     # number of tokens for this batch
    sm_scale: tl.constexpr,     # scaling factor
):
    # program ids
    h = tl.program_id(0)  # head id in [0, H)
    t = tl.program_id(1)  # token id in [0, L_tokens)

    if (h >= H) or (t >= L_tokens):
        return

    # load qn[h, :] and qp[h, :]
    dim_ck = tl.arange(0, CK)
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)  # [CK]

    dim_kp = tl.arange(0, KP)
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)  # [KP]

    # load token index
    idx = tl.load(tok_idx_ptr + t)  # int32

    # load corresponding rows from Kc_all and Kp_all
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

    # compute dot products
    dot_qn = tl.sum(qn_vec * Kc_row)  # scalar
    dot_qp = tl.sum(qp_vec * Kp_row)  # scalar

    scaled = (dot_qn + dot_qp) * sm_scale

    # store scaled logits at [h, t]
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_kernel(
    logits_ptr,   # *f32, base pointer to [H, L_tokens]
    lse_ptr,      # *f32, base pointer to [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    # First pass: compute max for numerical stability (natural log)
    max_val = -float("inf")
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        if val > max_val:
            max_val = val

    # Second pass: sum exp(val - max)
    sumexp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sumexp += tl.exp(val - max_val)

    lse = tl.log(sumexp)  # natural log
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    qn_ptr,        # *f32, base pointer to [H, CK]
    qp_ptr,        # *f32, base pointer to [H, KP]
    logits_ptr,    # *f32, base pointer to [H, L_tokens]
    lse_ptr,       # *f32, base pointer to [H]
    Kc_all_ptr,    # *f32, base pointer to [P, CK]
    tok_idx_ptr,   # *i32, base pointer to [L_tokens]
    out_ptr,       # *f32, base pointer to [H, CK]
    H: tl.constexpr,
    CK: tl.constexpr,
    KP: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    lse = tl.load(lse_ptr + h)

    # Accumulator for output
    out_acc = tl.zeros((CK,), tl.float32)

    # Loop over tokens, compute softmax and accumulate
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)  # scaled logits[h, t]
        prob = tl.exp(val - lse)  # softmax probability

        idx = tl.load(tok_idx_ptr + t)
        dim_ck = tl.arange(0, CK)
        Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)

        out_acc += prob * Kc_row

    # Store output for this head
    tl.store(out_ptr + h * CK + tl.arange(0, CK), out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, CK] bfloat16 (H must be 16)
        q_pe:   [B, H, KP] bfloat16
        ckv_cache: [P, 1, CK] bfloat16 -> use Kc_all = squeeze(1)
        kpe_cache: [P, 1, KP] bfloat16 -> use Kp_all = squeeze(1)
        kv_indptr: [B+1] int32
        kv_indices: [N_tokens] int32
        sm_scale: float32 scalar
        Returns: output [B, H, CK] bfloat16
        """
        # Cast to float32 for compute; Triton kernels operate on f32 pointers
        q_nope_f = q_nope.to(torch.float32)
        q_pe_f = q_pe.to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, CK]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, KP]

        B = q_nope_f.shape[0]
        H = q_nope_f.shape[1]
        assert H == 16, "num_qo_heads must be 16"
        CK = q_nope_f.shape[2]
        assert CK == 512, "head_dim_ckv must be 512"
        KP = q_pe_f.shape[2]
        assert KP == 64, "head_dim_kpe must be 64"

        device = q_nope_f.device

        # Output buffer in float32, then cast to bfloat16 at the end
        out = torch.empty((B, H, CK), dtype=torch.float32, device=device)

        # For each batch element b
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L = end - start
            if L <= 0:
                out[b] = torch.zeros((H, CK), dtype=torch.float32, device=device)
                continue

            tok_idx = kv_indices[start:end].to(torch.int32)

            # Per-batch q vectors
            qn = q_nope_f[b]  # [H, CK]
            qp = q_pe_f[b]    # [H, KP]

            # Temporary buffer for scaled_logits [H, L_tokens]
            logits = torch.empty((H, L), dtype=torch.float32, device=device)

            # Launch 1: compute scaled_logits
            grid = (H, L)
            _compute_scaled_logits_kernel[grid](
                qn, qp, Kc_all, Kp_all, tok_idx, logits,
                H=H, CK=CK, KP=KP, L_tokens=L, sm_scale=float(sm_scale)
            )

            # Launch 2: compute lse per head (natural log). We won't use lse in output since we don't return it,
            # but keeping the kernel ensures Triton-only pattern and can be reused if needed.
            lse = torch.empty((H,), dtype=torch.float32, device=device)
            _lse_kernel[(H,)](
                logits, lse,
                H=H, L_tokens=L
            )

            # Launch 3: compute final output per head
            out[b] = torch.zeros((H, CK), dtype=torch.float32, device=device)
            _compute_output_kernel[(H,)](
                qn, qp, logits, lse, Kc_all, tok_idx, out[b],
                H=H, CK=CK, KP=KP, L_tokens=L
            )

        # Return output cast to bfloat16 to match original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
