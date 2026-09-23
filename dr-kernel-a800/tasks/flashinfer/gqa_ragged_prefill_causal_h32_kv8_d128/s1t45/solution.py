import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_single_qh(
    Q, KEXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    KEXP_stride_k, KEXP_stride_h, KEXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_D: tl.constexpr  # fixed, e.g., 128
):
    # Grid dims: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Accumulate logits[q, h, k] across d
    acc = tl.zeros((num_kv_tokens,), dtype=tl.float32)

    # Loop over d with compile-time constant bound (head_dim = 128)
    for d in range(0, 128):
        q_val = tl.load(
            Q + q * Q_stride_q + h * Q_stride_h + d * Q_stride_d,
            mask=(q < num_q_tokens),
            other=0.0
        )
        k_vec = tl.load(
            KEXP + tl.arange(0, num_kv_tokens) * KEXP_stride_k + h * KEXP_stride_h + d * KEXP_stride_d,
            mask=(tl.arange(0, num_kv_tokens) < num_kv_tokens),
            other=0.0
        )  # [num_kv_tokens]
        acc += q_val * k_vec

    LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + tl.arange(0, num_kv_tokens) * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q < num_q_tokens))


@triton.jit
def _lse_masked_single_qh(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    ln2: tl.constexpr,
    delta: tl.constexpr,  # num_kv_tokens - num_q_tokens
    BLOCK_K: tl.constexpr
):
    # Grid dims: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Initialize max and sum_exp scalars
    max_vals = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.full((), 0.0, dtype=tl.float32)

    # Reduce over K in constexpr tiles
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens

        LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=k_mask, other=-float("inf"))  # [BLOCK_K]

        # Causal mask: j < (q + 1 + delta)
        allowed_mask = k_idx < (q + 1 + delta)
        vals_masked = tl.where(allowed_mask, vals, -float("inf"))

        seg_max = tl.max(vals_masked, axis=0)
        sum_exp += tl.sum(tl.exp(vals_masked - seg_max), axis=0)
        max_vals = tl.maximum(max_vals, seg_max)

    lse_val = max_vals + tl.log(sum_exp) * ln2
    LSE_ptr = LSE + q * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptr, lse_val, mask=(q < num_q_tokens))


@triton.jit
def _output_single_qh(
    LOGITS, VEXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    VEXP_stride_k, VEXP_stride_h, VEXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_K: tl.constexpr
):
    # Grid dims: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    LSE_ptr = LSE + q * LSE.stride(0) + h * LSE.stride(1)
    lse_val = tl.load(LSE_ptr, mask=(q < num_q_tokens), other=-float("inf"))  # scalar

    for d in range(0, 128):
        out_row = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=k_mask, other=-float("inf"))  # [BLOCK_K]

            # Causal mask: j < (q + 1)
            allowed_mask = k_idx < (q + 1)
            vals_masked = tl.where(allowed_mask, vals, -float("inf"))

            max_v = tl.max(vals_masked, axis=0)
            probs = tl.exp(vals_masked - max_v)  # [BLOCK_K]
            sum_p = tl.sum(probs, axis=0)
            probs = probs / sum_p  # softmax over K

            V_ptrs = VEXP + k_idx * VEXP_stride_k + h * VEXP_stride_h + d * VEXP_stride_d
            v_vec = tl.load(V_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]
            out_row += tl.sum(probs * v_vec, axis=0)

        OUT_ptr = OUT + q * OUT_stride_q + h * OUT_stride_h + d * OUT_stride_d
        tl.store(OUT_ptr, out_row, mask=(q < num_q_tokens))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Convert to float32 for computation
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        gqa_ratio = num_qo_heads // num_kv_heads  # 4
        k_expanded = k.repeat_interleave(gqa_ratio, dim=1).to(torch.float32)  # [total_kv, 32, 128]
        v_expanded = v.repeat_interleave(gqa_ratio, dim=1).to(torch.float32)  # [total_kv, 32, 128]

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Process each segment
        for b in range(1, qo_indptr.numel()):
            q_start = int(qo_indptr[b - 1].item())
            q_end = int(qo_indptr[b].item())
            kv_start = int(kv_indptr[b - 1].item())
            kv_end = int(kv_indptr[b].item())

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            if num_q_tokens <= 0 or num_kv_tokens <= 0:
                continue

            q_batch = q[q_start:q_end]             # [num_q_tokens, 32, 128]
            k_expanded_batch = k_expanded[kv_start:kv_end]  # [num_kv_tokens, 32, 128]
            v_expanded_batch = v_expanded[kv_start:kv_end]  # [num_kv_tokens, 32, 128]

            # Allocate intermediates
            logits_seg = torch.empty(
                (num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=q.device
            )
            output_seg = torch.empty(
                (num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=q.device
            )
            lse_seg = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=q.device)

            # 1) Compute logits for all (q,h)
            grid = (num_q_tokens, num_qo_heads)
            _compute_logits_single_qh[grid](
                q_batch, k_expanded_batch, logits_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_expanded_batch.stride(0), k_expanded_batch.stride(1), k_expanded_batch.stride(2),
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                BLOCK_D=128
            )

            # 2) Compute lse per (q,h) with causal mask
            _lse_masked_single_qh[grid](
                logits_seg, lse_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                self.ln2, (num_kv_tokens - num_q_tokens),
                BLOCK_K=64
            )

            # 3) Compute output per (q,h)
            _output_single_qh[grid](
                logits_seg, v_expanded_batch, lse_seg, output_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                v_expanded_batch.stride(0), v_expanded_batch.stride(1), v_expanded_batch.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_K=64
            )

            # Store segment results
            output[q_start:q_end] = output_seg.to(torch.bfloat16)
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
