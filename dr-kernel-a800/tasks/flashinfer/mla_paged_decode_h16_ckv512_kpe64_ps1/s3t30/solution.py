import torch
import math
import triton
import triton.language as tl


# Single Triton kernel: compute logits[h, t] = qn[h] @ Kc[t].T + qp[h] @ Kp[t].T
# Inputs:
#   qn_ptr: *f32 [H, Dq]
#   qp_ptr: *f32 [H, Dp]
#   Kc_ptr: *f32 [T, Dq]
#   Kp_ptr: *f32 [T, Dp]
#   logits_ptr: *f32 [H, T]
# Meta-parameters:
#   H: number of heads (compile-time constant)
#   T: number of tokens (compile-time constant)
#   Dq: query/key feature dim for Kc (compile-time constant, 512)
#   Dp: position feature dim for Kp (compile-time constant, 64)
#   BLOCK_D: power-of-two tile size for feature dimension (e.g., 128)
@triton.jit
def fused_logits_kernel(
    qn_ptr,        # *f32 [H, Dq]
    qp_ptr,        # *f32 [H, Dp]
    Kc_ptr,        # *f32 [T, Dq]
    Kp_ptr,        # *f32 [T, Dp]
    logits_ptr,    # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,
    Dp: tl.constexpr,
    BLOCK_D: tl.constexpr,  # e.g., 128 (power-of-two)
):
    # program ids
    h = tl.program_id(0)  # head index
    t = tl.program_id(1)  # token index

    # accumulators for the two dot products
    acc_qn = tl.zeros((), dtype=tl.float32)
    acc_qp = tl.zeros((), dtype=tl.float32)

    # iterate over feature tiles
    for d_start in range(0, Dq, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)  # BLOCK_D must be power-of-two
        mask_d = offs < Dq

        # load qn[h, offs]
        qn_vals = tl.load(qn_ptr + h * Dq + offs, mask=mask_d, other=0.0)  # [BLOCK_D]
        # load Kc[t, offs]
        Kc_vals = tl.load(Kc_ptr + t * Dq + offs, mask=mask_d, other=0.0)  # [BLOCK_D]
        acc_qn += tl.sum(qn_vals * Kc_vals, axis=0)

    for d_start in range(0, Dp, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)  # BLOCK_D must be power-of-two
        mask_d = offs < Dp

        # load qp[h, offs]
        qp_vals = tl.load(qp_ptr + h * Dp + offs, mask=mask_d, other=0.0)  # [BLOCK_D]
        # load Kp[t, offs]
        Kp_vals = tl.load(Kp_ptr + t * Dp + offs, mask=mask_d, other=0.0)  # [BLOCK_D]
        acc_qp += tl.sum(qp_vals * Kp_vals, axis=0)

    # write the sum to logits[h, t]
    tl.store(logits_ptr + h * T + t, acc_qn + acc_qp)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton"

        # Extract shapes
        B, H, Dq = q_nope.shape  # H is 16, Dq is 512
        _, _, Dp = q_pe.shape    # Dp is 64
        # Process each batch item
        output = torch.empty((B, H, Dq), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Constants for Triton kernels
        BLOCK_D = 128  # power-of-two tile size for features

        for b in range(B):
            # If no tokens for this batch element, skip
            if kv_indptr[b + 1] == kv_indptr[b]:
                output[b].zero_()
                lse[b].fill_(0.0)
                continue

            # Token range and indices for this batch
            L_tokens = int(kv_indptr[b + 1] - kv_indptr[b])
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).contiguous()  # [L_tokens]

            # Load Kc and Kp for this batch; cast to f32 for compute
            # ckv_cache: [N, 1, Dq], kpe_cache: [N, 1, Dp]
            Kc_b = ckv_cache[tok_idx].squeeze(1).contiguous().to(torch.float32)  # [L_tokens, Dq]
            Kp_b = kpe_cache[tok_idx].squeeze(1).contiguous().to(torch.float32)  # [L_tokens, Dp]

            # Load qn and qp for this batch; cast to f32
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, Dq]
            qp = q_pe[b].to(torch.float32).contiguous()   # [H, Dp]

            # Allocate logits [H, L_tokens]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)

            # Launch Triton kernel to compute logits
            grid = (H, L_tokens)
            fused_logits_kernel[grid](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp, BLOCK_D=BLOCK_D,
                num_warps=4, num_stages=2
            )

            # Scale and perform softmax along tokens
            logits_scaled = logits * sm_scale
            attn = torch.softmax(logits_scaled, dim=1)  # [H, L_tokens]

            # Compute LSE per head
            lse[b] = torch.logsumexp(logits_scaled, dim=1) / math.log(2.0)  # [H]

            # Compute output per head: out[h, :] = attn[h, :] @ Kc[:, :]
            # Kc_b: [L_tokens, Dq]; attn[h, :]: [L_tokens]
            out_vec = attn @ Kc_b  # [H, Dq], float32
            output[b] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
