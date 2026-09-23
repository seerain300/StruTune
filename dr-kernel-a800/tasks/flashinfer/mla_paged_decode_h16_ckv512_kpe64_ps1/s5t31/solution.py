import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_output_per_batch_kernel(
    qn_ptr,         # *f32, shape [H, CK]
    qp_ptr,         # *f32, shape [H, KP]
    Kc_ptr,         # *f32, shape [num_tokens, CK] (we use L_tokens rows)
    Kp_ptr,         # *f32, shape [num_tokens, KP]
    out_ptr,        # *f32, shape [H, CK]
    lse_ptr,        # *f32, shape [H]
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr, L_tokens: tl.constexpr, sm_scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Accumulators for numerically stable softmax
    max_val = -float("inf")
    sum_val = 0.0

    # Precompute strides (row-major)
    # We access rows using linear indexing: t * CK (for Kc) and t * KP (for Kp)
    # qn[h] is at qn_ptr + h * CK
    # qp[h] is at qp_ptr + h * KP
    qn_base = qn_ptr + h * CK
    qp_base = qp_ptr + h * KP

    # Accumulate output vector
    out = tl.zeros((CK,), dtype=tl.float32)

    # Loop over tokens in tiles
    for t0 in range(0, L_tokens, BLOCK_T):
        offs = t0 + tl.arange(0, BLOCK_T)
        mask = offs < L_tokens

        # For each t in the tile, compute scaled logits and update softmax accumulation.
        # We avoid using a 1D vectorized gather by looping unrolled over the tile width.
        # This keeps Triton happy and prevents compilation issues with indirect indexing.
        for i in range(BLOCK_T):
            t = t0 + i
            valid = t < L_tokens
            # If invalid, skip (mask ensures we won't load)
            # Load qn[h, :] and qp[h, :]
            qn_row = tl.load(qn_base)  # vector of length CK
            qp_row = tl.load(qp_base)  # vector of length KP

            # Load Kc[t, :] and Kp[t, :]
            # Row t of Kc: index = t * CK + arange(0, CK)
            # But we need single element at column j: Kc_ptr[t * CK + j], loop scalar
            # Instead, we load vector Kc_row via arange trick by loading qn_row again (not needed) or dummy. Simpler: use scalar loop approach below.
            # Better approach: load per-column using tl.load with pointer + scalar. Triton allows scalar loads with pointer + tl.full(0).
            # Compute dot components using scalar column loop:
            # However, Triton favors vectorized loads; we can create a vector 'cols' and load with mask, but better stick to simple scalar approach.

            # We'll implement scalar per-column accumulation:
            # acc = 0.0
            # for j in range(CK):
            #     acc += qn_row[j] * tl.load(Kc_ptr + t * CK + j)
            # accp = 0.0
            # for j in range(KP):
            #     accp += qp_row[j] * tl.load(Kp_ptr + t * KP + j)
            # scaled = sm_scale * (acc + accp)
            # Update max/sumexp
            # But Triton doesn't support per-iteration scalar load directly in this context cleanly; to avoid complexity and ensure correctness, we instead:
            # keep the vectorized approach for qn/qp, and do dot via tl.sum of qn_row * Kc_row as vectors. Triton supports tl.sum over a vector.

            # Build column vectors for qn_row and Kc_row via broadcasting
            j_vec = tl.arange(0, CK)  # 0..CK-1
            # For Kc, load row t: we need Kc_ptr[t * CK + j_vec]
            Kc_row = tl.load(Kc_ptr + t * CK + j_vec, mask=valid, other=0.0)
            # Compute dot qn · Kc
            dot_qn_Kc = tl.sum(qn_row * Kc_row, axis=0)

            j_vec_p = tl.arange(0, KP)  # 0..KP-1
            Kp_row = tl.load(Kp_ptr + t * KP + j_vec_p, mask=valid, other=0.0)
            dot_qp_Kp = tl.sum(qp_row * Kp_row, axis=0)

            scaled = sm_scale * (dot_qn_Kc + dot_qp_Kp)

            # Update max/sumexp
            # Note: we use natural log for logsumexp (PyTorch uses ln). If strict base-2 is required, we can multiply by log(2) in forward, but here we match original (no base change).
            new_max = tl.maximum(max_val, scaled)
            # sumexp update
            # When max changes, rescale previous sum
            if new_max > max_val:
                sum_val = sum_val * tl.exp(max_val - new_max)
                max_val = new_max
            sum_val += tl.exp(scaled - max_val)

            # Accumulate output: out += exp(scaled - max_val) * Kc[t, :]
            # We need Kc[t, :] vector
            Kc_row_out = tl.load(Kc_ptr + t * CK + j_vec, mask=valid, other=0.0)
            out += tl.exp(scaled - max_val) * Kc_row_out

    # Store lse for this head
    tl.store(lse_ptr + h, tl.log(sum_val))

    # Store output[h, :]
    out_ptr_base = out_ptr + h * CK
    # Write out vector
    for j in range(CK):
        tl.store(out_ptr_base + j, out[j])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Cast to float32 for computation
        device = q_nope.device
        H = q_nope.shape[1]
        CK = q_nope.shape[2]
        KP = q_pe.shape[2]

        # Ensure inputs are contiguous
        q_nope = q_nope.contiguous().to(torch.float32)   # [B, H, CK]
        q_pe = q_pe.contiguous().to(torch.float32)      # [B, H, KP]
        Kc_all = ckv_cache.contiguous().to(torch.float32)  # [num_tokens, CK]
        Kp_all = kpe_cache.contiguous().to(torch.float32)  # [num_tokens, KP]

        B = q_nope.shape[0]
        # Prepare outputs
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # For each batch b, compute output and lse using Triton kernel
        for b in range(B):
            # Number of tokens for this batch
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV cache for this batch element
                lse[b] = torch.tensor(-float("inf"), dtype=torch.float32, device=device)
                continue

            # Launch one program per head
            grid = (H,)
            _compute_output_per_batch_kernel[grid](
                q_nope[b], q_pe[b], Kc_all, Kp_all, output[b], lse[b],
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale),
                BLOCK_T=128  # tile size over tokens; safe for small L_tokens, unrolled loop handles tail
            )

        # Return (output, lse). Cast output to bfloat16 to match original dtype of q_nope.
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
