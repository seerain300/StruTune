import math
import torch
import triton
import triton.language as tl


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=None):
        super().__init__()
        # sm_scale is given at call; keep for signature symmetry
        self.sm_scale = sm_scale
        # Fixed dimensions per original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.head_dim = 128
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # q: [total_q, 32, 128], k: [total_kv, 8, 128], v: [total_kv, 8, 128]
        # Cast to float32 for Triton kernels (computation), keep original for reference if needed
        device = q.device

        # Create output and lse
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        output = torch.empty(
            (total_q, self.num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=device
        )
        lse = torch.empty(
            (total_q, self.num_qo_heads), dtype=torch.float32, device=device
        )

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
            q_batch = q[q_start:q_end].contiguous()  # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end].contiguous()  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end].contiguous()  # [num_kv_tokens, 8, 128]

            # Expand K and V by GQA ratio
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Prepare dtype for kernels
            q_f32 = q_batch.to(torch.float32)
            k_f32 = k_expanded.to(torch.float32)
            v_f32 = v_expanded.to(torch.float32)

            # Allocate per-segment outputs and lse
            output_seg = torch.empty(
                (num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device
            )
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

            # Choose constexpr tiling
            BLOCK_Q = 1            # one query per program to simplify masking
            BLOCK_K = 64           # tile over K (num_kv_tokens)
            BLOCK_D = 16           # tile over D (head_dim)

            # 1) Compute logits = Q @ K^T for each (q, head) and store in output_seg
            grid0 = (1, self.num_qo_heads)  # single q-tile only; we loop over q outside if needed
            # We'll manually loop over q positions (q_start .. q_end-1) by launching one program per q.
            for i in range(num_q_tokens):
                q_offsets = i + tl.arange(0, 1)  # [1]
                q_mask = q_offsets < num_q_tokens  # always true here

                # Grid dims: (1, heads)
                _compute_logits_kernel[grid0](
                    q_f32[i], k_f32, output_seg[i],
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    q_f32.stride(0), q_f32.stride(1), q_f32.stride(2),
                    k_f32.stride(0), k_f32.stride(1), k_f32.stride(2),
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K, SM_SCALE=self.sm_scale
                )

            # 2) LSE per (q, head) with causal mask
            grid_lse = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _lse_masked_kernel[grid_lse](
                output_seg, lse_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                self.ln2, (num_kv_tokens - num_q_tokens),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # 3) Softmax on logits with causal mask and output = softmax @ V_expanded
            grid_out = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _softmax_output_kernel[grid_out](
                output_seg, v_f32, lse_seg, output[q_start:q_end],
                num_q_tokens, num_kv_tokens, self.head_dim,
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                v_f32.stride(0), v_f32.stride(1), v_f32.stride(2),
                output[q_start:q_end].stride(0), output[q_start:q_end].stride(1), output[q_start:q_end].stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, SM_SCALE=self.sm_scale
            )

        return output, lse


# Triton kernels
@triton.jit
def _compute_logits_kernel(
    Q, K, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_stride_k, K_stride_h, K_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
    SM_SCALE: tl.constexpr
):
    # One program per q tile and head
    pid_h = tl.program_id(0)  # head index
    pid_q = tl.program_id(1)  # q tile index
    # We use BLOCK_Q=1 so only one q per program
    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [1]
    q_mask = q_offsets < num_q_tokens

    # Accumulator over D for each (q, k)
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    for d0 in range(0, 128, BLOCK_D):  # compile-time loop over head_dim tiles
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_mask = d_idx < head_dim

        # Load Q[q, pid_h, d]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + pid_h * Q_stride_h + d_idx[None, :] * Q_stride_d
        Q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)  # [1, D]

        # Load K[k, pid_h, d]
        K_ptrs = K + (q_offsets * 0 + tl.arange(0, 1))[:, None] * K_stride_k + pid_h * K_stride_h + d_idx[None, :] * K_stride_d  # dummy to satisfy, but we use loops
        # Note: We will load K for each k tile in the loop below; this pointer is only a placeholder.
        pass

    # We need to compute acc = sum_d Q_vals * K_vals per (q,k). Since K_vals depend on k,
    # we perform the reduction across d with K loaded per k tile in outer loop.
    # To avoid Python loops over runtime size, we use fixed tiles over k as well.
    for k0 in range(0, 128, BLOCK_K):  # compile-time loop: head_dim=128
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_idx < num_kv_tokens

        # Load K[k, pid_h, d] for this k tile
        K_ptrs = K + k_idx[:, None] * K_stride_k + pid_h * K_stride_h + (d0 + tl.arange(0, BLOCK_D))[None, :] * K_stride_d  # shape (K, D)
        # Build d vector for this tile
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_mask = d_idx < head_dim
        K_vals = tl.load(K_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [K, D]

        # For each d element in the tile, accumulate Q[:, d] * K[:, d]
        for dd in range(BLOCK_D):
            # d_valid = d_idx[dd] < head_dim -> always true here
            qd = Q + q_offsets * Q_stride_q + pid_h * Q_stride_h + (d_idx[dd]) * Q_stride_d
            Qd = tl.load(qd, mask=q_mask, other=0.0)  # [1]
            Kd = K_vals[:, dd]  # [K]
            acc += Qd * Kd[None, :]  # broadcast [1, K] * [K] -> [1, K]

    # Store logits: OUT[q, pid_h, k]
    OUT_ptrs = OUT + q_offsets * OUT_stride_q + pid_h * OUT_stride_h + (d0 + tl.arange(0, BLOCK_D))[None, :] * OUT_stride_d
    tl.store(OUT_ptrs, acc, mask=(q_mask[:, None] & k_mask[None, :]))


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LN2, delta,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Initialize max and sum_exp
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, 128, BLOCK_K):  # compile-time loop
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [K]
        k_mask = k_idx < num_kv_tokens

        # Causal mask: for each q, allowed k < q + 1 + delta
        q_pos = q_offsets  # [Q]
        allowed = k_idx[None, :] < (q_pos[:, None] + 1 + delta)  # [Q, K]

        LOGITS_ptrs = LOGITS + q_pos[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
        # For LSE, we only need unmasked max and sum
        # Compute max across K for each q
        tile_max = tl.max(vals, axis=1)
        sum_exp += tl.sum(tl.exp(vals - tile_max[:, None]), axis=1)
        max_vals = tl.maximum(max_vals, tile_max)

    lse_vals = max_vals + tl.log(sum_exp) * LN2
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_stride_k, V_stride_h, V_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
    SM_SCALE: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Load lse for this (q,h)
    LSE_ptrs = LSE + q_offsets * LSE.stride(0) + h * LSE.stride(1)
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

    # Output per (q, head, d)
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Compute softmax over K with causal mask for each (q,h)
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            q_pos = q_offsets  # [Q]
            allowed = k_idx[None, :] < (q_pos[:, None] + 1)  # causal: j < i+1

            LOGITS_ptrs = LOGITS + q_pos[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
            # subtract lse for numerical stability
            vals = vals - lse_vals[:, None]
            # softmax
            exp_vals = tl.exp(vals)
            sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
            probs = exp_vals / sum_exp[:, None]  # [Q, K]

            # Now accumulate output: out[q,d] += sum_k probs[q,k] * V[k,h,d]
            V_ptrs = V + k_idx[:, None] * V_stride_k + h * V_stride_h + d_idx[None, :] * V_stride_d
            V_vals = tl.load(V_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]
            # Broadcast probs [Q,K] and V_vals [K,D] -> sum over K
            # For each d in BLOCK_D, sum_k probs[:, k] * V[k, d]
            # We do it element-wise across D
            for dd in range(BLOCK_D):
                d_valid_curr = d_idx[dd] < head_dim  # always true
                Vd = V_vals[:, dd]  # [K]
                out_row[:, dd] = tl.sum(probs * Vd[None, :], axis=1)

        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


def run(*args):
    return ModelNew()(*args)
