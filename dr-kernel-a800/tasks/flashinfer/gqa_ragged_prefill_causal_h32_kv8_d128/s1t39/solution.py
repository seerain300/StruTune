import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits[q, h, k] = sum_d q[q, h, d] * k_exp[k, h, d]
@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), ceil(num_kv_tokens/BLOCK_K), heads)
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    h = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulator for [BLOCK_Q, BLOCK_K]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Reduction over D with constexpr bound
    for d in range(0, head_dim):
        q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d * Q_stride_d
        k_ptrs = K_EXP + k_offsets[None, :] * K_EXP_stride_k + h * K_EXP_stride_h + d * K_EXP_stride_d
        q_vals = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_vals = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0)
        acc += q_vals * k_vals

    logits_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    tl.store(logits_ptrs, acc, mask=(q_mask[:, None] & k_mask[None, :]))


# Triton kernel: compute lse[q, h] = logsumexp(LOGITS[q,h,:]) / ln(2) with causal mask
@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    ln2, delta,  # delta = num_kv_tokens - num_q_tokens
    BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads). We use BLOCK_Q=1 for per-(q,h) reduction.
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * 1 + tl.arange(0, 1)  # always 1 item per program
    q_mask = q_offsets < num_q_tokens

    # Max for numerical stability
    max_val = -float("inf")
    for k0 in range(0, head_dim, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        allowed = k_idx < (q_offsets[:, None] + 1 + delta)  # causal mask: j < (i + 1 + delta)
        ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
        # vals shape [1, BLOCK_K]; reduce max across K
        cur_max = tl.max(vals, axis=1)
        max_val = tl.maximum(max_val, cur_max)

    sum_exp = 0.0
    for k0 in range(0, head_dim, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        allowed = k_idx < (q_offsets[:, None] + 1 + delta)
        ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
        vals = vals - max_val[:, None]
        exp_vals = tl.exp(vals)
        sum_exp += tl.sum(exp_vals, axis=1)

    lse_vals = max_val + tl.log(sum_exp) * ln2  # [1]
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


# Triton kernel: compute softmax over K and output = softmax @ V_exp
@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads). BLOCK_Q=1 to handle one query per program
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * 1 + tl.arange(0, 1)  # [1]
    q_mask = q_offsets < num_q_tokens

    # Load lse for (q,h)
    LSE_ptrs = LSE + q_offsets * LSE.stride(0) + h * LSE.stride(1)
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [1]

    # Output per head over D tiles
    for d0 in range(0, head_dim, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT.stride(0) + h * OUT.stride(1) + d_idx[None, :] * OUT.stride(2)
        out_row = tl.zeros((1, BLOCK_D), dtype=tl.float32)

        # Compute softmax over K with causal mask, then accumulate output over d
        # Note: We compute softmax per (q,h) and then multiply with V_exp over k tiles
        for k0 in range(0, head_dim, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            q_pos = q_offsets  # [1]
            allowed = k_idx[None, :] < (q_pos[:, None] + 1 + delta)  # causal mask: j < (i + 1 + delta)

            LOGITS_ptrs = LOGITS + q_pos * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [1, K]
            vals = vals - lse_vals[:, None]  # [1, K]
            exp_vals = tl.exp(vals)
            sum_exp = tl.sum(exp_vals, axis=1)  # [1]
            probs = exp_vals / sum_exp[:, None]  # [1, K]

            V_EXP_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_vals = tl.load(V_EXP_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D_tile]
            # out_row += sum_k probs[:,None] * v_vals[None,:]
            out_row += tl.sum(probs[:, None] * v_vals[None, :], axis=0)  # reduce over K to [D_tile]

        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Move to CUDA and ensure contiguous
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton requires tensors on CUDA"
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        total_q = q.shape[0]
        total_kv = k.shape[0]
        num_qo_heads = self.num_qo_heads
        num_kv_heads = self.num_kv_heads

        # Precompute expanded K and V by GQA ratio
        k_expanded = k.repeat_interleave(self.gqa_ratio, dim=1).contiguous()  # [total_kv, num_qo_heads, head_dim]
        v_expanded = v.repeat_interleave(self.gqa_ratio, dim=1).contiguous()  # [total_kv, num_qo_heads, head_dim]

        # Output and lse buffers
        output = torch.empty((total_q, num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Process segments
        for b in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            if num_q_tokens <= 0 or num_kv_tokens <= 0:
                continue

            # Segment tensors
            q_seg = q[q_start:q_end].contiguous()      # [num_q_tokens, 32, 128]
            k_seg = k_expanded[kv_start:kv_end].contiguous()  # [num_kv_tokens, 32, 128]
            v_seg = v_expanded[kv_start:kv_end].contiguous()  # [num_kv_tokens, 32, 128]

            # Compute logits via Triton: [num_q_tokens, 32, num_kv_tokens]
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=q.device)

            BLOCK_Q = 32
            BLOCK_K = 64
            grid = (triton.cdiv(num_q_tokens, BLOCK_Q), triton.cdiv(num_kv_tokens, BLOCK_K), num_qo_heads)
            _compute_logits_kernel[grid](
                q_seg, k_seg, logits,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_seg.stride(0), q_seg.stride(1), q_seg.stride(2),
                k_seg.stride(0), k_seg.stride(1), k_seg.stride(2),
                logits.stride(0), logits.stride(1), logits.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # Compute lse per (q,h) with causal mask via Triton: [num_q_tokens, 32]
            grid_lse = (triton.cdiv(num_q_tokens, 1), num_qo_heads)
            _lse_masked_kernel[grid_lse](
                logits, lse,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits.stride(0), logits.stride(1), logits.stride(2),
                lse.stride(0), lse.stride(1),
                self.ln2, (num_kv_tokens - num_q_tokens),
                BLOCK_K=64
            )

            # Compute output = softmax(logits) @ v_seg via Triton: store into output[q,h,:] directly
            output_seg = torch.empty_like(output[q_start:q_end], dtype=torch.float32, device=q.device)
            grid_out = (triton.cdiv(num_q_tokens, 1), num_qo_heads)
            _softmax_output_kernel[grid_out](
                logits, v_seg, lse, output_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits.stride(0), logits.stride(1), logits.stride(2),
                v_seg.stride(0), v_seg.stride(1), v_seg.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=1, BLOCK_D=16, BLOCK_K=64
            )

            # Copy segment results into global output
            # output[q_start:q_end] = output_seg
            # Equivalent: slice assignment
            output[q_start:q_end] = output_seg.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
