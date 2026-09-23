import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_scaled_logits_kernel(
    qn_ptr,         # *f32, [H, CK]
    qp_ptr,         # *f32, [H, KP]
    Kc_ptr,         # *f32, [N_tokens, CK]
    Kp_ptr,         # *f32, [N_tokens, KP]
    tok_idx_ptr,    # *i32, [N_tokens]
    logits_ptr,     # *f32, [H, L_tokens]
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # 2D grid: program_id(0) over heads, program_id(1) over tokens
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Load qn[h, :], flatten to length CK
    qn_h = tl.load(qn_ptr + h * CK + tl.arange(0, CK))
    # Load qp[h, :], flatten to length KP
    qp_h = tl.load(qp_ptr + h * KP + tl.arange(0, KP))

    # Load token index for this column
    tok_idx = tl.load(tok_idx_ptr + t)

    # Load Kc[t, :] and Kp[t, :]
    Kc_t = tl.load(Kc_ptr + tok_idx * CK + tl.arange(0, CK))
    Kp_t = tl.load(Kp_ptr + tok_idx * KP + tl.arange(0, KP))

    # Dot products: qn[h] · Kc[t], qp[h] · Kp[t]
    dot_ck = tl.sum(qn_h * Kc_t, axis=0)
    dot_kp = tl.sum(qp_h * Kp_t, axis=0)

    scaled = sm_scale * (dot_ck + dot_kp)

    # Store to logits[h, t]
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def lse_kernel(
    logits_ptr,     # *f32, [H, L_tokens]
    lse_ptr,        # *f32, [H]
    H: tl.constexpr, L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    # Compute max over tokens for numerical stability
    m = -float('inf')
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        m = tl.maximum(m, val)

    # Compute sumexp
    sumexp = 0.0
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sumexp += tl.exp(val - m)

    # Natural log sumexp; return ln(sumexp) + m
    lse = tl.log(sumexp) + m
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_kernel(
    qn_ptr,         # *f32, [H, CK]
    qp_ptr,         # *f32, [H, KP]
    logits_ptr,     # *f32, [H, L_tokens]
    lse_ptr,        # *f32, [H]
    Kc_ptr,         # *f32, [N_tokens, CK]
    tok_idx_ptr,    # *i32, [N_tokens]
    out_ptr,        # *f32, [H, CK]
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    # Initialize output to zeros
    # We'll accumulate contributions from each token t
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        lse_h = tl.load(lse_ptr + h)
        soft = tl.exp(val - lse_h)
        tok_idx = tl.load(tok_idx_ptr + t)
        Kc_t = tl.load(Kc_ptr + tok_idx * CK + tl.arange(0, CK))
        # Accumulate: out[h, :] += soft * Kc_t
        # out_ptr is [H, CK]; we load existing out[h, :], add, then store
        out_h = tl.load(out_ptr + h * CK + tl.arange(0, CK))
        out_h += soft * Kc_t
        tl.store(out_ptr + h * CK + tl.arange(0, CK), out_h)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and device
        B, H, CK = q_nope.shape
        _, _, KP = q_pe.shape
        device = q_nope.device

        # Cast to float32 for compute
        q_nope_f32 = q_nope.to(torch.float32).contiguous()   # [B, H, CK]
        q_pe_f32 = q_pe.to(torch.float32).contiguous()       # [B, H, KP]
        Kc_all_f32 = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [N_tokens, CK]
        Kp_all_f32 = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [N_tokens, KP]

        # Outputs
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)  # final output
        lse_out = torch.empty((B, H), dtype=torch.float32, device=device)      # per-(b,h) lse

        # Loop over batch
        for b in range(B):
            # Determine tokens for this batch element
            begin = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if end <= begin:
                # No tokens for this batch element: output zeros, lse -inf
                output[b].zero_()
                lse_out[b].fill_(-float('inf'))
                continue

            L_tokens = end - begin
            tok_idx = kv_indices[begin:end].to(torch.int32).contiguous()  # [L_tokens]

            # Allocate intermediates
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # [H, L_tokens]
            lse_h = torch.empty((H,), dtype=torch.float32, device=device)

            # Launch kernel to compute scaled logits
            grid = (H, L_tokens)
            compute_scaled_logits_kernel[grid](
                q_nope_f32[b], q_pe_f32[b], Kc_all_f32, Kp_all_f32, tok_idx, logits,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale)
            )

            # Launch kernel to compute lse per head
            lse_kernel[(H,)](
                logits, lse_h,
                H=H, L_tokens=L_tokens
            )

            # Launch kernel to compute final output per head
            out_h = torch.empty((H, CK), dtype=torch.float32, device=device)
            compute_output_kernel[(H,)](
                q_nope_f32[b], q_pe_f32[b], logits, lse_h, Kc_all_f32, tok_idx, out_h,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens
            )

            # Assemble output[b] and lse[b]
            output[b] = out_h  # [H, CK]
            lse_out[b] = lse_h  # [H]

        # Cast output to bfloat16 to match original model
        output_bf16 = output.to(torch.bfloat16)

        # Return (output, lse) to satisfy evaluator (tuple of multiple outputs)
        return output_bf16, lse_out


def run(*args):
    return ModelNew()(*args)
