import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_masked_and_rowmax(
    Q, K, Mask2D, LOGITS, LSE, sm_scale,
    NUM_Q_T, NUM_K_T, NUM_H, HEAD_D,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, BH: tl.constexpr
):
    # Program IDs
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    # Compute h indices this program handles
    h_start = pid_h * BH
    h_offsets = h_start + tl.arange(0, BH)
    h_mask = h_offsets < NUM_H

    # Compute q token offsets this program handles
    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < NUM_Q_T

    # Compute KV offsets
    kv_offsets = tl.arange(0, BLOCK_KV)

    # Initialize row-wise maxima for lse
    row_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)

    # Loop over KV tokens in tiles
    for kv_start in range(0, NUM_K_T, BLOCK_KV):
        kv_idx = kv_start + kv_offsets
        kv_mask = kv_idx < NUM_K_T

        # Accumulator for logits for each (q, h, kv)
        acc = tl.zeros((BLOCK_Q, BH, BLOCK_KV), dtype=tl.float32)

        # Loop over head blocks
        for h in range(0, NUM_H, BH):
            # Get head offsets and mask for this block
            h_block_offsets = h + tl.arange(0, BH)
            h_block_mask = h_block_offsets < NUM_H

            # Build pointers
            # Q: [NUM_Q_T, NUM_H, HEAD_D], K: [NUM_K_T, NUM_H, HEAD_D]
            # We need to load Q[q, h, d] and K[k, h, d] for our q_offsets, h_offsets, and d loop
            d = 0
            # Loop over dimension D=128
            while d < HEAD_D:
                # Load Q tiles: shape [BLOCK_Q, BH] for this h block and d
                q_ptrs = Q + q_offsets[:, None] * (NUM_H * HEAD_D) + h_block_offsets[None, :] * HEAD_D + d
                q_vals = tl.load(q_ptrs, mask=q_mask[:, None] & h_block_mask[None, :], other=0.0)  # [BLOCK_Q, BH]

                # Load K tiles: shape [BLOCK_KV, BH] for this h block and d
                k_ptrs = K + kv_idx[None, :] * (NUM_H * HEAD_D) + h_block_offsets[:, None] * HEAD_D + d
                k_vals = tl.load(k_ptrs, mask=kv_mask[None, :] & h_block_mask[:, None], other=0.0)  # [BLOCK_KV, BH]

                # Compute outer product acc += Q[:, None, :] * K[None, :, :] for each (q,h) row and kv column
                # Broadcast q_vals [BLOCK_Q, BH] and k_vals [BLOCK_KV, BH]:
                # q_vals[:, :, None] -> [BLOCK_Q, BH, 1], k_vals[None, :, :] -> [1, BLOCK_KV, BH]
                # The result is [BLOCK_Q, BH, BLOCK_KV] but we accumulate across BH
                # Since BH is small (8), we can directly multiply:
                # For each h in BH, acc[:, h, :] += sum_over(BH_block) of q_vals[:, h_block] * k_vals[:, h_block]^T
                # We'll do an explicit sum across BH_block:
                # Create acc as [BLOCK_Q, BH, BLOCK_KV] but initialize zeros above.
                # For each h in BH:
                #   qv = q_vals[:, h] -> [BLOCK_Q]
                #   kv = k_vals[:, h] -> [BLOCK_KV]
                #   acc += qv[:, None] * kv[None, :]
                for hi in range(BH):
                    hh = h + hi  # absolute head index
                    # If hh >= NUM_H, skip (h_block_mask ensures we don't load beyond NUM_H)
                    if hh < NUM_H:
                        qv = q_vals[:, hi]  # [BLOCK_Q]
                        kv = k_vals[:, hi]  # [BLOCK_KV]
                        acc += qv[:, None] * kv[None, :]  # [BLOCK_Q, 1, BLOCK_KV] -> broadcast to [BLOCK_Q, BH, BLOCK_KV]
                d += 1

            # End of d loop for this h block
        # Apply scaling
        acc *= sm_scale

        # Apply causal mask: Mask2D has shape [NUM_Q_T, NUM_K_T], we need to use q_offsets and kv_idx
        # Mask is int8: 1 for valid, 0 for invalid. Convert to float and apply.
        mask_ptrs = Mask2D + q_offsets[:, None] * NUM_K_T + kv_idx[None, :]
        mask_vals = tl.load(mask_ptrs, mask=q_mask[:, None] & kv_mask[None, :], other=0).to(tl.float32)
        acc = tl.where(mask_vals > 0, acc, -float("inf"))

        # Store logits
        # LOGITS: [NUM_Q_T, NUM_H, NUM_K_T], float32
        logits_ptrs = LOGITS + q_offsets[:, None] * (NUM_H * NUM_K_T) + h_offsets[None, :] * NUM_K_T + kv_idx[None, :]
        store_mask = q_mask[:, None] & h_mask[None, :] & kv_mask[None, :]
        tl.store(logits_ptrs, acc, mask=store_mask)

        # Update row-wise maxima: max over KV columns and over BH
        # Flatten acc over BH to get [BLOCK_Q, BLOCK_KV], then reduce
        acc_flat = tl.reshape(acc, (BLOCK_Q, BLOCK_KV))
        local_max = tl.max(acc_flat, axis=1)  # [BLOCK_Q]
        row_max = tl.maximum(row_max, local_max)

    # For heads in this block where h_offsets < NUM_H, write lse[q, h] = log(row_max + eps)/log(2)
    # We'll only write where h_mask is true; for other h (>= NUM_H), skip.
    # Note: lse shape is [NUM_Q_T, NUM_H], float32
    eps = 1e-20
    lse_row = tl.log(row_max + eps) / 0.6931471805599453  # 1 / log(2)
    lse_ptrs = LSE + q_offsets * NUM_H + h_offsets
    tl.store(lse_ptrs, lse_row, mask=q_mask & h_mask)


@triton.jit
def compute_output_and_lse_from_logits(
    K_EXP, V_EXP, Mask2D, LOGITS, OUTPUT, LSE, sm_scale,
    NUM_Q_T, NUM_K_T, NUM_H, HEAD_D,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, BH: tl.constexpr
):
    # Program IDs
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    h_start = pid_h * BH
    h_offsets = h_start + tl.arange(0, BH)
    h_mask = h_offsets < NUM_H

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < NUM_Q_T

    kv_offsets = tl.arange(0, BLOCK_KV)

    # Load per-row lse (scaled by 1/log(2)) for each q and head, but we need only once per q across heads
    # LSE is [NUM_Q_T, NUM_H], we'll use it to compute softmax scale = exp(LSE[i,h] - logits[i,h,:]) or better, we use lse directly here.
    # Note: Kernel 1 already computed lse in global LSE buffer; we use it here for numerical stability.
    eps = 1e-20
    lse_row = tl.load(LSE + q_offsets * NUM_H + h_offsets, mask=q_mask & h_mask, other=0.0)  # [BLOCK_Q, BH]
    lse_row = tl.where(h_mask, lse_row, -float("inf"))

    for kv_start in range(0, NUM_K_T, BLOCK_KV):
        kv_idx = kv_start + kv_offsets
        kv_mask = kv_idx < NUM_K_T

        # Compute softmax along KV for each (q, h): softmax(logits[i,h,:] + (lse[i,h] - logsumexp)) equivalent via subtracting lse_row
        # We will load LOGITS[q, h, kv] and subtract lse_row[i,h] before exp for stability.
        # Then compute output via matvec: output[i,h,:] = sum_j softmax[i,h,j] * V_EXP[j,h,:]

        # Prepare accumulators
        # We need to compute the entire softmax vector per (q, h) then multiply by V_EXP. To keep code simple, we will:
        # - Compute a masked logits tile, subtract lse_row, exp, sum, then compute output via matvec. Triton allows loops; we can implement for small sizes.
        # However, Triton does not support returning tensors; we'll implement the reduction and matvec explicitly.

        # Build pointers for LOGITS
        logits_ptrs = LOGITS + q_offsets[:, None] * (NUM_H * NUM_K_T) + h_offsets[None, :] * NUM_K_T + kv_idx[None, :]
        logits = tl.load(logits_ptrs, mask=q_mask[:, None] & h_mask[None, :] & kv_mask[None, :], other=-float("inf"))
        # Subtract lse[i,h] to make the max -inf or small
        logits = logits - lse_row[:, None]  # broadcast over kv dimension

        # Compute softmax along KV per (q, h)
        # First, compute row-wise max for numerical stability
        logits_max = tl.max(logits, axis=1)  # [BLOCK_Q, 1] -> [BLOCK_Q]
        logits = logits - logits_max[:, None]
        exp_logits = tl.exp(logits)  # [BLOCK_Q, BLOCK_KV]
        softmax_sum = tl.sum(exp_logits, axis=1)  # [BLOCK_Q]

        # Apply mask: invalid positions were set to -inf -> exp(-inf)=0, so they are already zero
        # Now compute output: output[i,h,:] = sum_j softmax[i,h,j] * V_EXP[j,h,:]
        # V_EXP: [NUM_K_T, NUM_H, HEAD_D]
        acc_out = tl.zeros((BLOCK_Q, BH, HEAD_D), dtype=tl.float32)

        # Loop over KV tile to compute matvec
        for kv_i in range(BLOCK_KV):
            kv_valid = kv_start + kv_i < NUM_K_T
            if kv_valid:
                # softmax[i,h,kv_i] = exp_logits[i,h,kv_i] / softmax_sum[i,h]
                # Gather exp_logits[i,h,kv_i]
                exp_val = tl.load(
                    LOGITS + q_offsets[:, None] * (NUM_H * NUM_K_T) + h_offsets[None, :] * NUM_K_T + (kv_start + kv_i),
                    mask=q_mask[:, None] & h_mask[None, :], other=0.0
                )  # [BLOCK_Q, BH]
                exp_val = exp_val - logits_max[:, None]  # recompute? Not necessary; we already have exp_logits
                # Better: compute directly from exp_logits tile after loading all, but Triton doesn't allow dynamic selection; we'll reconstruct via load:
                # Instead, compute exp at runtime using the logits tile and softmax_sum:
                # Since we don't have per-element exp, we can compute the full exp_logits and then pick exp_val = exp(logits - max) at position kv_i.
                # We can store exp_logits and softmax_sum in LOGITS output? We need another buffer. For simplicity, we'll recompute.
                # Reconstruct exp value using the exp_logits tile:
                # We already computed exp_logits = exp(logits - logits_max) for the tile, but we didn't store it. So we need to reload logits and compute exp again.
                # However, we only need the single kv_i element, not the full vector. We can reload that single element safely.
                # For performance, we'll approximate by using the softmax_sum and a simplified approach: compute output via direct sum over KV positions in Python.
                # But since Triton kernel needs to be self-contained, we will compute the full softmax vector here to avoid recomputation overhead.
                # Let's recompute exp for this element:
                # We need the per-element logits for each (q,h); Triton doesn't let us store per-element; so we can't. Therefore, this approach is suboptimal.

        # Implementation detail: The above strategy is not ideal in Triton due to lack of dynamic indexing and storing intermediate exp per element.
        # Instead, we will perform the softmax reduction in Triton and compute the output via a second loop across KV tile, where we:
        # - compute exp for each element in the tile, accumulate sum, and store output[i,h,:] += softmax * V_EXP[kv_idx,h,:].
        # We'll implement this loop now.

        softmax_sum = tl.sum(exp_logits, axis=1)  # [BLOCK_Q]
        # Now compute output via loop over kv tile
        for kv_i in range(BLOCK_KV):
            kv_j = kv_start + kv_i
            valid = kv_j < NUM_K_T
            if valid:
                # softmax[i,h,kv_j] = exp(logits[i,h,kv_j] - logits_max[i,h]) / softmax_sum[i,h]
                # Load logits[i,h,kv_j]
                # logits_ptrs_j = LOGITS + q_offsets[:, None] * (NUM_H * NUM_K_T) + h_offsets[None, :] * NUM_K_T + kv_j
                # logit_j = tl.load(logits_ptrs_j, mask=q_mask[:, None] & h_mask[None, :], other=-float("inf"))
                # softmax_j = tl.exp(logit_j - logits_max[:, None]) / softmax_sum[:, None]
                # Load V_EXP[kv_j, h_offsets, :]
                V_ptrs = V_EXP + kv_j * (NUM_H * HEAD_D) + h_offsets * HEAD_D
                V_vals = tl.load(V_ptrs, mask=h_mask, other=0.0)  # [BH]
                # Then accumulate output[i,h,:] += softmax_j * V_vals
                # Since softmax_j is per (i,h), and V_vals is per head, we need to broadcast V_vals to [BLOCK_Q, BH] for accumulation.
                # But V_vals is [BH], and softmax_j is [BLOCK_Q]. We need to assign per (i,h). Triton allows broadcasting across second dimension:
                # We can compute output as:
                # acc_out[:, h, :] += softmax_j * V_vals for each h in BH. Implement via loop:
                for hi in range(BH):
                    hh = h + hi
                    if hh < NUM_H:
                        # softmax_j for this hh:
                        # logit_j = tl.load(LOGITS + q_offsets[:, None] * (NUM_H * NUM_K_T) + hh * NUM_K_T + kv_j, mask=q_mask[:, None], other=-float("inf"))
                        # Compute logit_j per q; but since we need per-(i,h), we can compute logit_j for all q in this tile via pointer:
                        logit_j_ptrs = LOGITS + q_offsets[:, None] * (NUM_H * NUM_K_T) + hh * NUM_K_T + kv_j
                        logit_j = tl.load(logit_j_ptrs, mask=q_mask[:, None], other=-float("inf"))
                        logit_j = logit_j - (logit_j.max(axis=1)[:, None])  # per q max across KV tile
                        exp_j = tl.exp(logit_j - logit_j.max(axis=1)[:, None])
                        sum_j = tl.sum(exp_j, axis=1)
                        softmax_j = exp_j / sum_j[:, None]  # [BLOCK_Q, 1]
                        # V_vals[hi] corresponds to head hh
                        Vh = V_vals[hi]  # scalar
                        acc_out[:, hi, :] += softmax_j * Vh  # [BLOCK_Q, 1, HEAD_D] broadcast along last dim

        # End of KV tile
        # Store output for this tile
        output_ptrs = OUTPUT + q_offsets[:, None] * (NUM_H * HEAD_D) + h_offsets[None, :] * HEAD_D
        store_mask = q_mask[:, None] & h_mask[None, :]
        tl.store(output_ptrs, acc_out, mask=store_mask)

    # No further work needed; second kernel writes output for this (q,h) block.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4

        # Triton kernel meta-parameters
        self.BLOCK_Q = 64
        self.BLOCK_KV = 64
        self.BH = 8  # number of heads processed per program

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Move to CUDA if available
        if not q.is_cuda:
            q = q.cuda()
        if not k.is_cuda:
            k = k.cuda()
        if not v.is_cuda:
            v = v.cuda()

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.numel() - 1

        # Output and lse
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=q.device)

        # Iterate over batches
        for b in range(len_indptr):
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if qo_start >= qo_end or kv_start >= kv_end:
                continue

            # Slice
            q_batch = q[qo_start:qo_end].contiguous()  # [num_qo_tokens, 32, 128]
            k_batch = k[kv_start:kv_end].contiguous()  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end].contiguous()  # [num_kv_tokens, 8, 128]

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_batch.shape[0]

            # Expand k and v along heads by GQA ratio
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Build causal mask on host: [num_q_tokens, num_kv_tokens] int8
            i_range = torch.arange(num_q_tokens, device=q.device)
            j_range = torch.arange(num_kv_tokens, device=q.device)
            delta = num_kv_tokens - num_q_tokens
            mask_2d = (j_range[None, :] < (i_range[:, None] + 1 + delta)).to(torch.int8)

            # Grid: (num_batches=1, ceil_div(32, BH), ceil_div(num_q_tokens, BLOCK_Q))
            grid = (1, triton.cdiv(self.num_qo_heads, self.BH), triton.cdiv(num_q_tokens, self.BLOCK_Q))

            # Kernel 1: compute logits and per-row maxima (for lse)
            compute_logits_masked_and_rowmax[grid](
                q_batch, k_expanded, mask_2d, output, lse, sm_scale,
                num_q_tokens, num_kv_tokens, self.num_qo_heads, self.head_dim,
                self.BLOCK_Q, self.BLOCK_KV, self.BH,
                num_warps=4, num_stages=2
            )

            # Kernel 2: compute output and lse from logits
            compute_output_and_lse_from_logits[grid](
                k_expanded, v_expanded, mask_2d, output, output, lse, sm_scale,
                num_q_tokens, num_kv_tokens, self.num_qo_heads, self.head_dim,
                self.BLOCK_Q, self.BLOCK_KV, self.BH,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 to match original example
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# Helper functions to generate inputs (same as original)
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
