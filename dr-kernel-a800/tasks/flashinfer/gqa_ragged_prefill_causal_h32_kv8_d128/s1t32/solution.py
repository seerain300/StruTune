import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel 1: compute logits[q, h, k] = sum_d q[q,h,d] * k_expanded[k,h,d]
# We reduce over D=128 using broadcasting; no runtime-dependent loops.
@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    k_offsets = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulator [BLOCK_Q, BLOCK_K]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Reduce over D in tiles of BLOCK_D (here BLOCK_D=128, so single iteration)
    for d0 in range(0, head_dim, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_mask = d_idx < head_dim

        # Load Q[q,h,d] -> [BLOCK_Q, BLOCK_D]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d
        q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)

        # Load K_EXP[k,h,d] -> [BLOCK_K, BLOCK_D]
        K_ptrs = K_EXP + k_offsets[:, None] * K_EXP_stride_k + h * K_EXP_stride_h + d_idx[None, :] * K_EXP_stride_d
        k_vals = tl.load(K_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)

        # Outer product and accumulate: [BLOCK_Q, BLOCK_D] * [BLOCK_K, BLOCK_D]^T -> [BLOCK_Q, BLOCK_K]
        prod = tl.dot(q_vals, tl.trans(k_vals))
        acc += prod

    # Store to LOGITS[q, h, k]
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q_mask[:, None] & k_mask[None, :]))


# Triton kernel 2: compute lse[q,h] = logsumexp(LOGITS[q,h,:]) / ln(2)
# We reduce over K in tiles of BLOCK_K=64; loop bound is constexpr (<=128).
@triton.jit
def _lse_all_heads_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    ln2,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # Initialize max per q
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)

    # Reduce over K in tiles
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_idx < num_kv_tokens

        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [BLOCK_Q, BLOCK_K]
        # Update max
        max_vals = tl.maximum(max_vals, tl.max(vals, axis=1))

    # Compute sum_exp = sum exp(vals - max_vals) over K tiles
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_idx < num_kv_tokens
        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [BLOCK_Q, BLOCK_K]
        exp_vals = tl.exp(vals - max_vals[:, None])  # [BLOCK_Q, BLOCK_K]
        sum_exp += tl.sum(exp_vals, axis=1)

    lse_vals = max_vals + tl.log(sum_exp) * ln2
    # Store LSE[q, h]
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


# Triton kernel 3: compute output[q,h,:] = softmax(LOGITS[q,h,:]) @ V_EXP[h,:]
# We reduce over K in tiles and write output across D (128). No runtime-dependent loops.
@triton.jit
def _softmax_output_all_heads_kernel(
    LOGITS, V_EXP, LSE, OUTPUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUTPUT_stride_q, OUTPUT_stride_h, OUTPUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # Load lse[q, h] and cast to float32 for stability
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [BLOCK_Q]

    # For each D tile
    for d0 in range(0, head_dim, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_mask = d_idx < head_dim

        OUTPUT_ptrs = OUTPUT + q_offsets[:, None] * OUTPUT_stride_q + h * OUTPUT_stride_h + d_idx[None, :] * OUTPUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Softmax over K tiles
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_mask = k_idx < num_kv_tokens

            LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [BLOCK_Q, BLOCK_K]

            # Subtract max for stability
            max_vals = tl.max(vals, axis=1)  # [BLOCK_Q]
            vals = vals - max_vals[:, None]

            # exp and sum
            exp_vals = tl.exp(vals)          # [BLOCK_Q, BLOCK_K]
            sum_exp = tl.sum(exp_vals, axis=1)  # [BLOCK_Q]
            probs = exp_vals / sum_exp[:, None]  # [BLOCK_Q, BLOCK_K]

            # Multiply by V_EXP[k,h,d] and accumulate over K
            V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_vals = tl.load(V_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [BLOCK_K, BLOCK_D]
            # out_row[q,d] += sum_k probs[q,k] * v_vals[k,d]
            # Compute outer product per (q, d): sum_k probs[q,k] * v_vals[k,d]
            out_row += tl.sum(probs[:, :, None] * v_vals[None, :, :], axis=1)

        # Store output[q, h, d]
        tl.store(OUTPUT_ptrs, out_row, mask=(q_mask[:, None] & d_mask[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants per the original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA and dtype; original code uses bfloat16 for inputs
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        device = q.device
        q_f = q.to(torch.float32).contiguous()
        k_f = k.to(torch.float32).contiguous()
        v_f = v.to(torch.float32).contiguous()

        total_q = q_f.shape[0]
        total_kv = k_f.shape[0]
        len_indptr = qo_indptr.shape[0]

        # Allocate outputs
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Expand K and V by GQA ratio
        k_exp = k_f.repeat_interleave(self.gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
        v_exp = v_f.repeat_interleave(self.gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

        # Process each segment [b, b+1)
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Segment views
            q_seg = q_f[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_exp_seg = k_exp[kv_start:kv_end]  # [num_kv_tokens, 32, 128]
            v_exp_seg = v_exp[kv_start:kv_end]  # [num_kv_tokens, 32, 128]

            # Allocate per-segment outputs
            logits_seg = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)
            output_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

            # Launch 1) compute logits
            BLOCK_Q = 1
            BLOCK_K = 64
            BLOCK_D = 128
            grid1 = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads, triton.cdiv(num_kv_tokens, BLOCK_K))
            _compute_logits_kernel[grid1](
                q_seg, k_exp_seg, logits_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_seg.stride(0), q_seg.stride(1), q_seg.stride(2),
                k_exp_seg.stride(0), k_exp_seg.stride(1), k_exp_seg.stride(2),
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # 2) Compute lse per (q,h)
            ln2 = 1.0 / math.log(2.0)
            grid2 = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _lse_all_heads_kernel[grid2](
                logits_seg, lse_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                ln2,
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # 3) Compute output: softmax(LOGITS) @ V_EXP
            grid3 = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _softmax_output_all_heads_kernel[grid3](
                logits_seg, v_exp_seg, lse_seg, output_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                v_exp_seg.stride(0), v_exp_seg.stride(1), v_exp_seg.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # Copy back to global output/lse for this segment
            output[q_start:q_end] = output_seg
            lse[q_start:q_end] = lse_seg

        # Return in the original dtype expectations: output bfloat16, lse float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
