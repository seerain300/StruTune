import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: compute logits[q, h, k] = sum_d q[q,h,d] * k_expanded[k,h,d]
# We reduce over D=128 using broadcasting; no runtime-dependent loops.
@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    SM_SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads, ceil(num_kv_tokens/BLOCK_K))
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    pid_k = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulator for logits[q, h, k] over d in [0, head_dim), BLOCK_Q x BLOCK_K
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Reduce over D=128 via broadcasting: q[:, d] * k[:, d]
    # Create d index and load q[k,h,d] and k[k,h,d] across D
    d_idx = tl.arange(0, 128)  # compile-time constant
    q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d  # [Q, 128]
    k_ptrs = K_EXP + k_offsets[:, None] * K_EXP_stride_k + h * K_EXP_stride_h + d_idx[None, :] * K_EXP_stride_d  # [K, 128]

    # Masked loads for boundaries
    q_vals = tl.load(q_ptrs, mask=(q_mask[:, None] & (d_idx[None, :] < head_dim)), other=0.0)  # [Q, 128]
    k_vals = tl.load(k_ptrs, mask=(k_mask[:, None] & (d_idx[None, :] < head_dim)), other=0.0)  # [K, 128]

    # Multiply and sum across D
    prod = q_vals[:, None, :] * k_vals[None, :, :]  # [Q, K, 128]
    acc += tl.sum(prod, axis=2)  # [Q, K]

    # Scale by sm_scale
    acc *= SM_SCALE

    # Store results to LOGITS[q,h,k]
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q_mask[:, None] & k_mask[None, :]))


# Kernel 2: compute lse[q, h] = logsumexp(LOGITS[q,h,:]) / ln(2)
@triton.jit
def _lse_all_heads_kernel(
    LOGITS, LSE, ln2,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # For numerical stability: max across K
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    for k0 in range(0, 128, BLOCK_K):  # compile-time loop
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_idx < num_kv_tokens
        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [Q, K]
        # Update max
        max_vals = tl.maximum(max_vals, tl.max(vals, axis=1))  # [Q]

    # Sum exp(vals - max) across K
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [Q, K]
        exp_vals = tl.exp(vals - max_vals[:, None])
        sum_exp += tl.sum(exp_vals, axis=1)  # [Q]

    lse_vals = max_vals + tl.log(sum_exp) * ln2  # [Q]
    # Store to LSE[q, h] (assume LSE is [num_q_tokens, num_qo_heads])
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


# Kernel 3: compute output[q, h, d] = sum_k softmax(LOGITS[q,h,k]) * V_EXP[h,d]
@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # Load lse for this (q,h) for all q in tile: assume LSE shape [num_q_tokens, num_qo_heads]
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

    # Compute output across d tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_mask = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Softmax over K tiles
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_mask = k_idx < num_kv_tokens

            LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [Q, K]
            # Subtract lse for numerical stability
            vals = vals - lse_vals[:, None]  # [Q, K]
            exp_vals = tl.exp(vals)          # [Q, K]
            sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
            probs = exp_vals / sum_exp[:, None]  # [Q, K]

            V_EXP_ptrs = V_EXP + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d  # [1, D] broadcast along Q
            v_vals = tl.load(V_EXP_ptrs, mask=d_mask[None, :], other=0.0)              # [1, D]
            # Multiply probs [Q, K] by v_vals [1, D] broadcast to [Q, D] along K (implicitly K=1, but we sum K anyway)
            # To get [Q, D], we need to multiply by each k separately. Here we use v_vals[None, :] to broadcast over K dimension
            v_broadcast = v_vals[None, :]  # [1, D] -> [Q, D] via broadcasting during multiplication
            contrib = probs * v_broadcast  # [Q, D]
            out_row += tl.sum(contrib, axis=1)[:, None]  # sum over K tiles

        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_mask[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale):
        super().__init__()
        self.sm_scale = float(sm_scale)
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        # Fixed tiling parameters; they work for head_dim=128
        self.BLOCK_Q = 1
        self.BLOCK_K = 64
        self.BLOCK_D = 16

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Cast inputs to float32 for Triton compute
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Determine total_q and total_kv
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())

        # Prepare outputs
        # We'll compute per segment and write into output tensor
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=q.device)

        # Iterate segments
        for b in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice tensors
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k_f32[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v_f32[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            # Expand K and V by GQA ratio
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate intermediate logits [num_q_tokens, 32, num_kv_tokens]
            logits = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=q.device)

            # Launch kernel 1: compute logits
            grid1 = (triton.cdiv(num_q_tokens, self.BLOCK_Q), self.num_qo_heads, triton.cdiv(num_kv_tokens, self.BLOCK_K))
            _compute_logits_kernel[grid1](
                q_batch, k_expanded, logits,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                logits.stride(0), logits.stride(1), logits.stride(2),
                SM_SCALE=self.sm_scale,
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K
            )

            # Launch kernel 2: compute lse
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=q.device)
            grid2 = (triton.cdiv(num_q_tokens, self.BLOCK_Q), self.num_qo_heads)
            _lse_all_heads_kernel[grid2](
                logits, lse_seg, 1.0 / math.log(2.0),
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits.stride(0), logits.stride(1), logits.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K
            )

            # Launch kernel 3: compute output
            out_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=q.device)
            grid3 = (triton.cdiv(num_q_tokens, self.BLOCK_Q), self.num_qo_heads)
            _softmax_output_kernel[grid3](
                logits, v_expanded, lse_seg, out_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits.stride(0), logits.stride(1), logits.stride(2),
                v_expanded.stride(0), v_expanded.stride(2),
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K, BLOCK_D=self.BLOCK_D
            )

            # Write segment results into output and lse
            output[q_start:q_end] = out_seg.to(torch.bfloat16)
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
