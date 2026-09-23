import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _compute_logits_matmul_kernel(
        Q, KEXP, LOGITS,
        NUM_Q_TOK, NUM_K_TOK, HEAD_DIM,
        Q_stride_q, Q_stride_h, Q_stride_d,
        K_stride_k, K_stride_h, K_stride_d,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        # Grid dims: (ceil(NUM_Q_TOK/BLOCK_Q), ceil(NUM_K_TOK/BLOCK_K), HEADS)
        pid_q = tl.program_id(0)
        pid_k = tl.program_id(1)
        h = tl.program_id(2)

        q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)    # [Q]
        k_idx = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)    # [K]
        d_idx = tl.arange(0, BLOCK_D)                      # [D]

        q_mask = q_idx < NUM_Q_TOK
        k_mask = k_idx < NUM_K_TOK
        d_mask = d_idx < HEAD_DIM  # always true for HEAD_DIM=128, but keep mask for generality

        # Load Q tile: (Q, D)
        Q_ptrs = Q + q_idx[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d
        q_vals = tl.load(Q_ptrs, mask=q_mask[:, None], other=0.0)  # (Q, D)

        # Load K^T tile: (K, D)
        K_ptrs = KEXP + k_idx[:, None] * K_stride_k + h * K_stride_h + d_idx[None, :] * K_stride_d
        k_vals = tl.load(K_ptrs, mask=k_mask[:, None], other=0.0)  # (K, D)

        # Compute logits (Q, K) = (Q, D) @ (K, D).T
        logits_tile = tl.sum(q_vals[:, None, :] * k_vals[None, :, :], axis=2)  # (Q, K)

        # Store to LOGITS at [q, h, k]
        LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        tl.store(LOGITS_ptrs, logits_tile, mask=(q_mask[:, None] & k_mask[None, :]))


    @triton.jit
    def _lse_all_heads_kernel(
        LOGITS, LSE, LN2,
        NUM_Q_TOK, NUM_K_TOK, HEAD_DIM,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        LSE_stride_q, LSE_stride_h,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        # Grid: (ceil(NUM_Q_TOK/BLOCK_Q), HEADS)
        pid_q = tl.program_id(0)
        h = tl.program_id(1)

        # We assume BLOCK_Q=1; grid covers all Q tokens. For general cases, pid_q indexes tiles.
        q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [Q]
        q_mask = q_idx < NUM_Q_TOK

        # Compute max over K for numerical stability
        max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
        for k0 in range(0, 128, BLOCK_K):  # compile-time loop, 128 is constexpr in kernel context
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [K]
            k_mask = k_idx < NUM_K_TOK

            LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))
            # Reduce max over K
            max_vals = tl.maximum(max_vals, tl.max(vals, axis=1))

        # Compute sum of exp(vals - max_vals) over K
        sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [K]
            k_mask = k_idx < NUM_K_TOK

            LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))
            exp_vals = tl.exp(vals - max_vals[:, None])
            sum_exp += tl.sum(exp_vals, axis=1)

        lse_vals = max_vals + tl.log(sum_exp) * LN2

        # Store lse[q, h]
        LSE_ptrs = LSE + q_idx * LSE_stride_q + h * LSE_stride_h
        tl.store(LSE_ptrs, lse_vals, mask=q_mask)


    @triton.jit
    def _softmax_output_all_heads_kernel(
        LOGITS, VEXP, LSE, OUT,
        NUM_Q_TOK, NUM_K_TOK, HEAD_DIM,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        V_stride_k, V_stride_h, V_stride_d,
        OUT_stride_q, OUT_stride_h, OUT_stride_d,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        # Grid: (ceil(NUM_Q_TOK/BLOCK_Q), HEADS)
        pid_q = tl.program_id(0)
        h = tl.program_id(1)

        q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [Q]
        q_mask = q_idx < NUM_Q_TOK

        # Load lse[q, h]
        LSE_ptrs = LSE + q_idx * LSE_stride_q + h * LSE_stride_h
        lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

        # Compute output[q, h, d] across D tiles
        for d0 in range(0, 128, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)  # [D]
            d_mask = d_idx < HEAD_DIM

            # Initialize output for this (Q, D) tile
            OUT_ptrs = OUT + q_idx[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
            out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

            # Iterate over K tiles to compute softmax and accumulate
            for k0 in range(0, 128, BLOCK_K):
                k_idx = k0 + tl.arange(0, BLOCK_K)  # [K]
                k_mask = k_idx < NUM_K_TOK

                # Load LOGITS[q, h, k] for this K chunk and subtract lse
                LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
                vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # (Q, K)
                vals = vals - lse_vals[:, None]  # (Q, K)

                probs = tl.exp(vals)  # (Q, K)
                sum_probs = tl.sum(probs, axis=1)  # (Q,)
                probs = probs / sum_probs[:, None]  # (Q, K)

                # Load VEXP[k, h, d] chunk and accumulate: sum_k probs[q,k] * V[k,h,d]
                V_ptrs = VEXP + k_idx[:, None] * V_stride_k + h * V_stride_h + d_idx[None, :] * V_stride_d
                v_chunk = tl.load(V_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # (K, D)
                # Broadcast probs to (Q, K, 1) and v_chunk to (1, K, D), then reduce over K
                out_row += tl.sum(probs[:, :, None] * v_chunk[None, :, :], axis=1)  # (Q, D)

            # Store output tile
            tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_mask[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Inputs must be CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        device = q.device

        # Constants
        num_qo_heads = 32
        num_kv_heads = 8
        assert num_qo_heads == 32 and num_kv_heads == 8
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        total_q = q.shape[0]
        total_kv = k.shape[0]
        len_indptr = qo_indptr.shape[0]
        assert kv_indptr.shape[0] == len_indptr

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, 128), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(1, len_indptr):
            q_start = int(qo_indptr[b - 1].item())
            q_end = int(qo_indptr[b].item())
            kv_start = int(kv_indptr[b - 1].item())
            kv_end = int(kv_indptr[b].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice batch
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            # Expand K/V by GQA ratio
            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Cast to float32 for Triton
            q_f = q_batch.to(torch.float32)
            kexp_f = k_expanded.to(torch.float32)
            vexp_f = v_expanded.to(torch.float32)

            # Allocate segment outputs
            output_seg = torch.empty((num_q_tokens, num_qo_heads, 128), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)

            # Tiling constants
            BLOCK_Q = 1
            BLOCK_K = 64
            BLOCK_D = 16

            # 1) Compute logits: Q @ K^T for all q,h,k
            grid1 = (triton.cdiv(num_q_tokens, BLOCK_Q), triton.cdiv(num_kv_tokens, BLOCK_K), num_qo_heads)
            _compute_logits_matmul_kernel[grid1](
                q_f, kexp_f, output_seg,
                num_q_tokens, num_kv_tokens, 128,
                q_f.stride(0), q_f.stride(1), q_f.stride(2),
                kexp_f.stride(0), kexp_f.stride(1), kexp_f.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # 2) Compute lse per (q,h)
            ln2 = 1.0 / math.log(2.0)
            grid2 = (triton.cdiv(num_q_tokens, BLOCK_Q), num_qo_heads)
            _lse_all_heads_kernel[grid2](
                output_seg, lse_seg, ln2,
                num_q_tokens, num_kv_tokens, 128,
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # 3) Compute output: softmax(LOGITS) @ V_expanded
            grid3 = (triton.cdiv(num_q_tokens, BLOCK_Q), num_qo_heads)
            _softmax_output_all_heads_kernel[grid3](
                output_seg, vexp_f, lse_seg, output[q_start:q_end],
                num_q_tokens, num_kv_tokens, 128,
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                vexp_f.stride(0), vexp_f.stride(1), vexp_f.stride(2),
                output[q_start:q_end].stride(0), output[q_start:q_end].stride(1), output[q_start:q_end].stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

        # Cast output to bfloat16 to match original behavior
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
