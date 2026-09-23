import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_compute_lse_and_output_per_batch_kernel(
    qn_ptr,         # *fp32, [H, D] contiguous, D=512 (tl.constexpr)
    qp_ptr,         # *fp32, [H, Dp] contiguous, Dp=64 (tl.constexpr)
    Kc_ptr,         # *fp32, [L_tokens, D] contiguous
    Kp_ptr,         # *fp32, [L_tokens, Dp] contiguous
    logits_ptr,     # *fp32, [H, L_tokens] contiguous buffer for logits_scaled
    out_ptr,        # *fp32, [H, D] contiguous buffer for output
    lse_ptr,        # *fp32, [H] buffer for lse (per head)
    H,              # int32
    L_tokens,       # int32
    sm_scale,       # float32
    BLOCK_T: tl.constexpr,  # token block size (e.g., 128)
):
    # One program per head h
    h = tl.program_id(0)

    # Initialize per-head lse
    m = -float("inf")
    s = 0.0

    # Process tokens in blocks
    for t_start in range(0, L_tokens, BLOCK_T):
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        mask = t_offsets < L_tokens

        # Compute logits_scaled for this block: [BLOCK_T]
        logits_block = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Accumulate dot products over D and Dp
        # qn[h, :] and qp[h, :] are vectors of length D and Dp respectively
        for kk in range(0, 512):  # D=512 (tl.constexpr)
            qn_val = tl.load(qn_ptr + h * 512 + kk)  # fp32
            # Load corresponding column from Kc for all t in block
            Kc_col = tl.load(Kc_ptr + t_offsets * 512 + kk, mask=mask, other=0.0)  # fp32
            acc1 = qn_val * Kc_col
            # Accumulate for logits_block
            logits_block += acc1

        for kk in range(0, 64):  # Dp=64 (tl.constexpr)
            qp_val = tl.load(qp_ptr + h * 64 + kk)  # fp32
            Kp_col = tl.load(Kp_ptr + t_offsets * 64 + kk, mask=mask, other=0.0)  # fp32
            acc2 = qp_val * Kp_col
            logits_block += acc2

        # Scale logits
        logits_block = logits_block * sm_scale

        # Store logits_scaled into buffer for this head
        # The buffer is viewed as [H, L_tokens] contiguous, so offset = h * L_tokens + t_offsets
        tl.store(logits_ptr + h * L_tokens + t_offsets, logits_block, mask=mask)

        # Update lse (max and sum) for this head
        # For masked positions, use -inf
        logits_masked = tl.where(mask, logits_block, -float("inf"))
        m_block = tl.max(logits_masked, axis=0)
        s_block = tl.sum(tl.exp(logits_masked - m_block), axis=0)
        # s is scalar; s_block is scalar
        s = s + s_block
        m = tl.maximum(m, m_block)

    # Compute lse for this head: (1/ln(2)) * log(s * 2^m)
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    lse_val = inv_ln2 * (math.log(s) + m)
    tl.store(lse_ptr + h, lse_val)

    # Now compute output[h, :] = softmax(logits_scaled[h, :]) @ Kc[:, :]
    # We need logits_scaled for this head across all tokens. Re-iterate tokens in blocks and accumulate.
    output_row = tl.zeros([512], dtype=tl.float32)
    for t_start_out in range(0, L_tokens, BLOCK_T):
        t_offsets_out = t_start_out + tl.arange(0, BLOCK_T)
        mask_out = t_offsets_out < L_tokens
        logits_row = tl.load(logits_ptr + h * L_tokens + t_offsets_out, mask=mask_out, other=-float("inf"))
        # Compute softmax probabilities for this block
        m_row = tl.max(tl.where(mask_out, logits_row, -float("inf")), axis=0)
        probs = tl.exp(logits_row - m_row)  # masked positions are -inf -> 0
        # Weighted sum with Kc for each column kk
        for kk in range(0, 512):
            Kc_col = tl.load(Kc_ptr + t_offsets_out * 512 + kk, mask=mask_out, other=0.0)
            output_row += probs * Kc_col

    # Store output row
    tl.store(out_ptr + h * 512 + tl.arange(0, 512), output_row)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and constraints
        batch_size, H, D = q_nope.shape  # H=16, D=512
        _, _, Dp = q_pe.shape            # Dp=64
        num_pages = ckv_cache.shape[0]
        _, _, L = ckv_cache.shape        # L=1 (not used directly)
        Lp = kpe_cache.shape[2]          # 64

        assert H == 16 and D == 512 and Dp == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        assert kv_indptr.shape[0] == batch_size + 1
        device = q_nope.device

        # Prepare Kc_all and Kp_all: [num_pages, D] and [num_pages, Dp]
        Kc_all = ckv_cache.to(torch.float32)  # [num_pages, 1, D] -> [num_pages, D]
        Kp_all = kpe_cache.to(torch.float32)  # [num_pages, 1, Dp] -> [num_pages, Dp]
        Kc_all = Kc_all.squeeze(1)  # [num_pages, D]
        Kp_all = Kp_all.squeeze(1)  # [num_pages, Dp]

        # Allocate output and lse
        output = torch.empty((batch_size, H, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # For each batch element, compute token range and process with Triton
        for b in range(batch_size):
            # Compute L_tokens and tok_idx
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No valid tokens for this batch element: output zeros, lse = -inf
                output[b].zero_()
                lse[b].fill_(float("-inf"))
                continue

            tok_idx = kv_indices[start:end]  # [L_tokens]
            # Slice Kc and Kp for this batch element
            Kc = Kc_all[tok_idx]  # [L_tokens, D]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Cast queries to fp32
            qn = q_nope[b].to(torch.float32)  # [H, D]
            qp = q_pe[b].to(torch.float32)    # [H, Dp]

            # Allocate Triton buffers (fp32) for logits and output
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # [H, L_tokens]
            out = torch.empty((H, D), dtype=torch.float32, device=device)            # [H, D]

            # Launch fused Triton kernel: one program per head
            grid = (H,)
            BLOCK_T = 128  # token block size; tuneable
            fused_compute_lse_and_output_per_batch_kernel[grid](
                qn, qp, Kc, Kp, logits, out, lse[b],
                H, L_tokens, float(sm_scale),
                BLOCK_T=BLOCK_T,
                num_warps=4,
            )

            # Store output as bfloat16
            output[b] = out.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
