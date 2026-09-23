import torch
import math
import triton
import triton.language as tl


# Kernel 1: Compute logits_scaled[h, l] = qn[h]·Kc[l] + qp[h]·Kp[l], for h in [0..H), l in [0..L)
# Inputs:
#   qn_ptr: [H, Dc] float32
#   qp_ptr: [H, Dp] float32
#   Kc_ptr: [L, Dc] float32
#   Kp_ptr: [L, Dp] float32
#   out_ptr: [H, L] float32 (logits_scaled)
# Grid: (ceil(H/BLOCK_H), ceil(L/BLOCK_L))
@triton.jit
def compute_logits_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    H: tl.constexpr, L: tl.constexpr,
    Dc: tl.constexpr, Dp: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_L: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_l = tl.program_id(1)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_h = h_offsets < H
    mask_l = l_offsets < L

    # Accumulators for dot products
    acc1 = tl.zeros((BLOCK_H, BLOCK_L), dtype=tl.float32)  # qn[h] · Kc[l]
    acc2 = tl.zeros((BLOCK_H, BLOCK_L), dtype=tl.float32)  # qp[h] · Kp[l]

    # Loop over Dc and Dp in chunks
    for d in range(0, Dc, 128):
        d_vec = d + tl.arange(0, 128)
        mask_d = d_vec < Dc
        # qn[h, d] for all h in this block: shape [BLOCK_H, 128]
        qn = tl.load(
            qn_ptr + h_offsets[:, None] * Dc + d_vec[None, :],
            mask=mask_h[:, None] & mask_d[None, :],
            other=0.0
        )
        # Kc[l, d] for all l in this block: shape [BLOCK_L, 128]
        Kc = tl.load(
            Kc_ptr + l_offsets[:, None] * Dc + d_vec[None, :],
            mask=mask_l[:, None] & mask_d[None, :],
            other=0.0
        )
        acc1 += tl.dot(qn, Kc.T)  # [BLOCK_H, 128] @ [128, BLOCK_L] -> [BLOCK_H, BLOCK_L]

    for d in range(0, Dp, 128):
        d_vec = d + tl.arange(0, 128)
        mask_d = d_vec < Dp
        # qp[h, d] for all h in this block: shape [BLOCK_H, 128]
        qp = tl.load(
            qp_ptr + h_offsets[:, None] * Dp + d_vec[None, :],
            mask=mask_h[:, None] & mask_d[None, :],
            other=0.0
        )
        # Kp[l, d] for all l in this block: shape [BLOCK_L, 128]
        Kp = tl.load(
            Kp_ptr + l_offsets[:, None] * Dp + d_vec[None, :],
            mask=mask_l[:, None] & mask_d[None, :],
            other=0.0
        )
        acc2 += tl.dot(qp, Kp.T)  # [BLOCK_H, 128] @ [128, BLOCK_L] -> [BLOCK_H, BLOCK_L]

    logits_scaled = acc1 + acc2  # [BLOCK_H, BLOCK_L]

    # Store logits_scaled
    out_idx = h_offsets[:, None] * L + l_offsets[None, :]
    tl.store(
        out_ptr + out_idx,
        logits_scaled,
        mask=mask_h[:, None] & mask_l[None, :]
    )


# Kernel 2: For each head h, compute softmax along L for logits_scaled[h, :] and write attn[h, :]
# Also compute lse[h] = logsumexp(logits_scaled) / ln(2) and store in lse_ptr[h].
# Inputs:
#   logits_ptr: [H, L] float32
#   attn_ptr: [H, L] float32
#   lse_ptr: [H] float32
# Grid: (H, 1)
@triton.jit
def softmax_and_lse_kernel(
    logits_ptr, attn_ptr, lse_ptr,
    H: tl.constexpr, L: tl.constexpr,
    BLOCK_L: tl.constexpr
):
    h = tl.program_id(0)
    # If h >= H, guard; Triton will launch exactly H programs, so no need
    # We will process the entire L in chunks for numerical stability and performance
    # Compute row max
    max_val = -float('inf')
    for offs in range(0, L, BLOCK_L):
        l = offs + tl.arange(0, BLOCK_L)
        mask = l < L
        vals = tl.load(logits_ptr + h * L + l, mask=mask, other=-float('inf'))
        # Reduce max across this block
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum(exp(logits - max)) across L
    sum_exp = 0.0
    for offs in range(0, L, BLOCK_L):
        l = offs + tl.arange(0, BLOCK_L)
        mask = l < L
        vals = tl.load(logits_ptr + h * L + l, mask=mask, other=-float('inf'))
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)

    # lse = logsumexp / ln(2)
    ln2 = 1.4426950408889634
    lse_val = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + h, lse_val)

    # Write attn[h, :]
    for offs in range(0, L, BLOCK_L):
        l = offs + tl.arange(0, BLOCK_L)
        mask = l < L
        vals = tl.load(logits_ptr + h * L + l, mask=mask, other=-float('inf'))
        attn_vals = tl.exp(vals - max_val) / sum_exp
        tl.store(attn_ptr + h * L + l, attn_vals, mask=mask)


# Kernel 3: Compute out[h, :] = sum_l attn[h, l] * Kc[l], i.e., out = attn @ Kc
# Inputs:
#   attn_ptr: [H, L] float32
#   Kc_ptr: [L, Dc] float32
#   out_ptr: [H, Dc] float32
# Grid: (ceil(H/BLOCK_H), 1) or (H, 1) with reduction along L. We use (H, 1) and loop.
@triton.jit
def compute_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H: tl.constexpr, L: tl.constexpr, Dc: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_L: tl.constexpr
):
    h = tl.program_id(0)
    # Accumulator for out[h, :]
    acc = tl.zeros((Dc,), dtype=tl.float32)
    # Loop over L in chunks to compute dot with Kc
    for offs in range(0, L, BLOCK_L):
        l = offs + tl.arange(0, BLOCK_L)
        mask_l = l < L
        attn_vec = tl.load(attn_ptr + h * L + l, mask=mask_l, other=0.0)  # [BLOCK_L]
        Kc_vec = tl.load(Kc_ptr + l * Dc, mask=mask_l, other=0.0)        # [BLOCK_L, Dc]
        # Broadcast attn_vec over Dc, then sum over L
        prod = attn_vec[:, None] * Kc_vec                           # [BLOCK_L, Dc]
        acc += tl.sum(prod, axis=0)
    # Store result
    tl.store(out_ptr + h * Dc, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on device (CUDA for Triton), dtype handling
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Triton requires CUDA tensors"

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Prepare Kc_all and Kp_all as float32 slices
        # Note: ckv_cache/kpe_cache have shape [num_pages, 1, Dc/Dp]; squeeze the middle dim
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dp]

        # Compute per-batch token indices
        kv_indptr = kv_indptr.to(device)  # int32
        L_tokens_list = []
        tok_idx_list = []
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                L_tokens_list.append(0)
                tok_idx_list.append(torch.empty(0, dtype=torch.int32, device=device))
                continue
            tok_idx = kv_indices[start:end].to(device)
            tok_idx_list.append(tok_idx)
            L_tokens_list.append(tok_idx.numel())

        # Prepare outputs
        output = torch.empty((batch_size, H, Dc), dtype=torch.float32, device=device)  # compute in float32
        lse = torch.full((batch_size, H), float("-inf"), dtype=torch.float32, device=device)

        # Launch Triton kernels per batch
        for b in range(batch_size):
            L_tokens = L_tokens_list[b]
            tok_idx = tok_idx_list[b]

            if L_tokens == 0:
                # No tokens for this batch; output zeros, lse stays -inf
                # We still need to fill output tensor zeros explicitly
                output[b].zero_()
                lse[b].zero_()
                continue

            # Slice Kc and Kp for this batch
            Kc = Kc_all[tok_idx]  # [L_tokens, Dc], float32
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp], float32

            # Prepare q vectors: q_nope[b] and q_pe[b] as [H, Dc] and [H, Dp]
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, Dc]
            qp = q_pe[b].to(torch.float32).contiguous()    # [H, Dp]

            # Allocate intermediate buffers
            logits_scaled = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # [H, L]
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)           # [H, L]

            # Launch compute_logits_kernel
            BLOCK_H = 16  # H is 16; we process all heads in one block
            BLOCK_L = 64  # tile over L
            grid = (triton.cdiv(H, BLOCK_H), triton.cdiv(L_tokens, BLOCK_L))
            compute_logits_kernel[grid](
                qn, qp, Kc, Kp, logits_scaled,
                H, L_tokens, Dc, Dp,
                BLOCK_H, BLOCK_L,
                num_warps=4, num_stages=2
            )

            # Launch softmax_and_lse_kernel to get attn and lse
            BLOCK_L_SOFT = 128
            grid_softmax = (H, 1)
            softmax_and_lse_kernel[grid_softmax](
                logits_scaled, attn, lse[b],
                H, L_tokens,
                BLOCK_L_SOFT,
                num_warps=4, num_stages=2
            )

            # Launch compute_out_kernel: out = attn @ Kc -> [H, Dc]
            BLOCK_H_OUT = 16
            BLOCK_L_OUT = 64
            grid_out = (H, 1)
            compute_out_kernel[grid_out](
                attn, Kc, output[b],
                H, L_tokens, Dc,
                BLOCK_H_OUT, BLOCK_L_OUT,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 as per original; lse stays float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse