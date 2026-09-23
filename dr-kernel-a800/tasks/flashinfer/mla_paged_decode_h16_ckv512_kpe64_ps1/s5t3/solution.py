import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    qn_ptr,        # [H, CK], base pointer
    qp_ptr,        # [H, KP], base pointer
    Kc_all_ptr,    # [P, CK], base pointer
    Kp_all_ptr,    # [P, KP], base pointer
    tok_idx_ptr,   # [L_tokens], int32
    scaled_ptr,    # [H, L_tokens], float32, base pointer
    H: tl.constexpr,          # num_qo_heads
    CK: tl.constexpr,         # head_dim_ckv
    KP: tl.constexpr,         # head_dim_kpe
    L_tokens: tl.constexpr,   # number of tokens for this batch
    sm_scale: tl.constexpr,   # scaling factor
):
    # Grid: 1D over tokens; each program computes scaled_logits[h, t] for all h
    t = tl.program_id(0)
    if t >= L_tokens:
        return

    # Loop over heads and compute scaled logits
    for h in range(H):
        # Load qn[h, :] and qp[h, :]
        dim_ck = tl.arange(0, CK)
        dim_kp = tl.arange(0, KP)
        qn_vec = tl.load(qn_ptr + h * CK + dim_ck)   # [CK], float32
        qp_vec = tl.load(qp_ptr + h * KP + dim_kp)   # [KP], float32

        # Load Kc_row and Kp_row for this token
        idx = tl.load(tok_idx_ptr + t)               # int32
        Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
        Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

        # Compute dot-products
        dot_qn = tl.sum(qn_vec * Kc_row)
        dot_qp = tl.sum(qp_vec * Kp_row)
        scaled_val = (dot_qn + dot_qp) * sm_scale

        # Store to scaled_logits[h, t]
        tl.store(scaled_ptr + h * L_tokens + t, scaled_val)


@triton.jit
def _lse_kernel(
    scaled_ptr,    # [H, L_tokens], float32
    lse_ptr,       # [H], float32
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    # 1) compute max over tokens
    max_val = -float("inf")
    for t in range(0, L_tokens):
        val = tl.load(scaled_ptr + h * L_tokens + t)  # scalar
        if val > max_val:
            max_val = val

    # 2) sum exp(scaled - max)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(scaled_ptr + h * L_tokens + t)
        sum_exp += tl.exp(val - max_val)

    lse = tl.log(sum_exp)
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    scaled_ptr,    # [H, L_tokens], float32
    lse_ptr,       # [H], float32
    Kc_all_ptr,    # [P, CK], base pointer
    tok_idx_ptr,   # [L_tokens], int32
    out_ptr,       # [H, CK], float32, base pointer
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    # Accumulator for output vector [CK]
    out_acc = tl.zeros((CK,), tl.float32)

    # Loop over tokens; for each token, compute softmax contribution and accumulate
    for t in range(0, L_tokens):
        # Load softmax contribution for this token: exp(scaled[h, t] - lse[h])
        scaled_val = tl.load(scaled_ptr + h * L_tokens + t)  # scalar
        lse_val = tl.load(lse_ptr + h)                       # scalar
        soft = tl.exp(scaled_val - lse_val)                 # scalar in [0,1]

        # Load Kc_row for this token
        idx = tl.load(tok_idx_ptr + t)                      # int32
        dim_ck = tl.arange(0, CK)
        Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)   # [CK]

        # Accumulate: out[h, :] += soft * Kc_row
        out_acc += soft * Kc_row

    # Store result
    tl.store(out_ptr + h * CK + dim_ck, out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        H = num_qo_heads
        CK = head_dim_ckv
        KP = head_dim_kpe

        device = q_nope.device

        # Convert inputs to float32 for compute
        q_nope_f32 = q_nope.contiguous().to(torch.float32)      # [B, H, CK]
        q_pe_f32 = q_pe.contiguous().to(torch.float32)          # [B, H, KP]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [P, CK]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [P, KP]
        tok_idx = kv_indices.contiguous().to(torch.int32)        # [L_tokens]

        # Allocate outputs
        output_f32 = torch.empty((batch_size, H, CK), dtype=torch.float32, device=device)  # per batch
        # We'll compute lse per batch element in the loop
        # For now, create a per-batch lse buffer
        # But since we need per-batch lse, we'll allocate after we know b-specific L_tokens. We'll compute and store per b.
        lse_f32 = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Process each batch element b
        for b in range(batch_size):
            # Determine L_tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens for this batch element
                output_f32[b].zero_()
                lse_f32[b].zero_()
                continue

            # Prepare pointers for this batch
            qn_ptr = q_nope_f32[b]                 # [H, CK]
            qp_ptr = q_pe_f32[b]                   # [H, KP]

            # Allocate scaled_logits buffer for this batch element
            scaled_logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute scaled logits
            _compute_scaled_logits_kernel[(L_tokens,)](
                qn_ptr, qp_ptr, Kc_all, Kp_all, tok_idx, scaled_logits,
                H=H, CK=CK, KP_KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale),
            )

            # Launch Triton kernel to compute lse per head
            lse_b = torch.empty((H,), dtype=torch.float32, device=device)
            _lse_kernel[(H,)](
                scaled_logits, lse_b,
                H=H, L_tokens=L_tokens,
            )

            # Store lse for this batch
            lse_f32[b] = lse_b

            # Launch Triton kernel to compute final output for this batch
            output_f32[b] = torch.empty((H, CK), dtype=torch.float32, device=device)
            _compute_output_kernel[(H,)](
                scaled_logits, lse_b, Kc_all, tok_idx, output_f32[b],
                H=H, CK=CK, L_tokens=L_tokens,
            )

        # Cast outputs to bfloat16 to match original behavior
        output_bf16 = output_f32.to(torch.bfloat16)

        # The original divides lse by log(2). We compute 1/log(2) and multiply to avoid host .log() calls.
        inv_log2 = 1.4426950408889634  # 1 / ln(2)
        lse_div_bf32 = (lse_f32 * inv_log2).to(torch.float32)

        # Return: output tensor [B, H, CK] and lse tensor [B, H]
        return output_bf16, lse_div_bf32


def run(*args):
    return ModelNew()(*args)
