import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_ptr,  # *f32, [H, CK]
    qp_ptr,  # *f32, [H, KP]
    Kc_ptr,  # *f32, [L_tokens, CK]
    Kp_ptr,  # *f32, [L_tokens, KP]
    logits_ptr,  # *f32, [H, L_tokens]
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr, L_tokens: tl.constexpr, sm_scale: tl.float32
):
    # program ids
    h = tl.program_id(0)
    t = tl.program_id(1)
    # Compute dot products for CK and KP
    # qn[h, :] and Kc[t, :]
    offs_c = tl.arange(0, CK)
    qn_vec = tl.load(qn_ptr + h * CK + offs_c)  # [CK]
    Kc_vec = tl.load(Kc_ptr + t * CK + offs_c)  # [CK]
    dot_qn = tl.sum(qn_vec * Kc_vec, axis=0)  # scalar

    # qp[h, :] and Kp[t, :]
    offs_k = tl.arange(0, KP)
    qp_vec = tl.load(qp_ptr + h * KP + offs_k)  # [KP]
    Kp_vec = tl.load(Kp_ptr + t * KP + offs_k)  # [KP]
    dot_qp = tl.sum(qp_vec * Kp_vec, axis=0)  # scalar

    val = sm_scale * (dot_qn + dot_qp)  # scaled logits
    # store to logits[h, t]
    tl.store(logits_ptr + h * L_tokens + t, val)


@triton.jit
def compute_lse_kernel(
    logits_ptr,  # *f32, [H, L_tokens]
    lse_ptr,     # *f32, [H]
    H: tl.constexpr, L_tokens: tl.constexpr
):
    h = tl.program_id(0)
    max_val = -float("inf")
    sum_exp = 0.0
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        # max tracking
        max_val = tl.maximum(max_val, val)
        # sumexp tracking
        sum_exp += tl.exp(val - max_val)
    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_kernel(
    logits_ptr,   # *f32, [H, L_tokens]
    Kc_ptr,       # *f32, [L_tokens, CK]
    out_ptr,      # *f32, [H, CK]
    H: tl.constexpr, CK: tl.constexpr, L_tokens: tl.constexpr, sm_scale: tl.float32
):
    h = tl.program_id(0)
    # accumulate output[h, :]
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        prob = tl.exp(val * sm_scale)  # softmax probability per token
        # accumulate tiles along CK dimension
        for i in range(0, CK, 128):
            offs = i + tl.arange(0, 128)
            mask = offs < CK
            Kc_tile = tl.load(Kc_ptr + t * CK + offs, mask=mask, other=0.0)  # [128]
            out_tile = tl.load(out_ptr + h * CK + offs, mask=mask, other=0.0)  # [128]
            out_tile += prob * Kc_tile
            tl.store(out_ptr + h * CK + offs, out_tile, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA for Triton
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Triton requires CUDA tensors"

        # Cast to float32 for compute
        q_nope_f32 = q_nope.to(torch.float32)  # [B, H, CK] but in original H=16, CK=512, we handle general
        q_pe_f32 = q_pe.to(torch.float32)      # [B, H, KP] H=16, KP=64
        Kc_all_f32 = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, CK]
        Kp_all_f32 = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, KP]

        B = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        CK = q_nope_f32.shape[2]
        KP = q_pe_f32.shape[2]
        device = q_nope_f32.device

        output = []  # [B, H, CK]
        lse_list = []  # [B, H]

        # Loop over batch elements
        for b in range(B):
            # Determine L_tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV tokens for this batch element; output zeros, lse zeros
                out_b = torch.zeros((H, CK), dtype=torch.float32, device=device)
                lse_b = torch.zeros((H,), dtype=torch.float32, device=device)
                output.append(out_b)
                lse_list.append(lse_b)
                continue

            # Gather relevant token indices for this batch
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b] + L_tokens].to(torch.long)  # [L_tokens]
            Kc_subset = Kc_all_f32[tok_idx]  # [L_tokens, CK]
            Kp_subset = Kp_all_f32[tok_idx]  # [L_tokens, KP]

            # Prepare qn and qp as [H, CK] and [H, KP]
            qn = q_nope_f32[b]  # [H, CK]
            qp = q_pe_f32[b]    # [H, KP]

            # Allocate logits [H, L_tokens]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch compute_logits_kernel
            grid = (H, L_tokens)
            compute_logits_kernel[grid](
                qn, qp, Kc_subset, Kp_subset, logits,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale)
            )

            # Compute lse per head
            lse = torch.empty((H,), dtype=torch.float32, device=device)
            _ = compute_lse_kernel[(H,)](
                logits, lse,
                H=H, L_tokens=L_tokens
            )

            # Compute output per head: out[h, :] = sum_t softmax(logits[h, t] * sm_scale) * Kc[t, :]
            out_b = torch.zeros((H, CK), dtype=torch.float32, device=device)
            compute_output_kernel[(H,)](
                logits, Kc_subset, out_b,
                H=H, CK=CK, L_tokens=L_tokens, sm_scale=float(sm_scale)
            )

            output.append(out_b)
            lse_list.append(lse)

        # Cast output to bfloat16 to match original
        output_bf16 = [out.to(torch.bfloat16) for out in output]
        # Return tuple (output, lse). We keep lse as float32; the original returns lse as float32.
        return (output_bf16, lse_list)


def run(*args):
    return ModelNew()(*args)
