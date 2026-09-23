import torch
import math
import triton
import triton.language as tl

# Triton kernels: no Python loops with runtime-dependent bounds; fixed tiling (tl.constexpr).
# We use compile-time constants for loops over d, k, and q tiles.

@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    num_q_tokens, head_dim, num_kv_tokens,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Accumulator for logits[q, h, k] over d in fixed tiles
    acc = tl.zeros((BLOCK_Q, num_kv_tokens), dtype=tl.float32)

    # Iterate over d in fixed tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        # Load Q[q, h, d] -> [Q, D]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d
        q_mat = tl.load(
            Q_ptrs,
            mask=q_mask[:, None] & d_valid[None, :],
            other=0.0
        )  # [BLOCK_Q, BLOCK_D], float32

        # Load K[k, h, d] -> [K, D] across all k
        K_ptrs = K_EXP + tl.arange(0, num_kv_tokens)[:, None] * K_EXP_stride_k + h * K_EXP_stride_h + d_idx[None, :] * K_EXP_stride_d
        k_mat = tl.load(
            K_ptrs,
            mask=d_valid[None, :],
            other=0.0
        )  # [num_kv_tokens, BLOCK_D], float32

        # Outer product and accumulate: [Q, D] * [K, D]^T -> [Q, K]
        contrib = tl.dot(q_mat, tl.trans(k_mat))  # [BLOCK_Q, num_kv_tokens]
        acc += contrib

    # Store logits[q, h, k]
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + tl.arange(0, num_kv_tokens)[None, :] * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=q_mask[:, None])


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    num_q_tokens, head_dim,
    ln2: tl.constexpr,
    delta: tl.constexpr,  # num_kv_tokens - num_q_tokens
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Initialize max and sum for logsumexp
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Reduce over K tiles
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_valid = k_idx < head_dim

        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        # Apply causal mask: k < (q + 1 + delta)
        q_pos = q_offsets[:, None]  # [Q, 1] but used as [Q]
        allowed = k_idx[None, :] < (q_pos + 1 + delta)  # [1, K] will broadcast to [Q, K]
        vals = tl.load(LOGITS_ptrs, mask=q_mask[:, None] & k_valid[None, :], other=-float("inf"))  # [Q, K]

        # Mask out invalid allowed positions by setting to -inf
        vals = tl.where(allowed, vals, -float("inf"))

        # For masked positions, max cannot increase; ensure they don't affect max/sum
        # Compute row-wise max and sum(exp(vals))
        row_max = tl.max(vals, axis=1)                   # [Q]
        sum_exp += tl.sum(tl.exp(vals - row_max[:, None]), axis=1)
        max_vals = tl.maximum(max_vals, row_max)

    lse_vals = max_vals + tl.log(sum_exp) * ln2
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    num_q_tokens, head_dim, num_kv_tokens,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Load lse[q, h] for numerical stability
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [BLOCK_Q]

    # Compute output[q, h, d] across d tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Accumulate over K tiles
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_valid = k_idx < num_kv_tokens

            # Softmax over K with causal mask
            LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            q_pos = q_offsets[:, None]
            allowed = k_idx[None, :] < (q_pos + 1)  # [1, K]
            vals = tl.load(LOGITS_ptrs, mask=q_mask[:, None] & k_valid[None, :] & allowed, other=-float("inf"))  # [Q, K]
            vals = vals - lse_vals[:, None]  # [Q, K]

            exp_vals = tl.exp(vals)
            sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
            probs = exp_vals / sum_exp[:, None]  # [Q, K]

            V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_mat = tl.load(V_ptrs, mask=k_valid[:, None] & d_valid[None, :], other=0.0)  # [K, D]
            contrib = tl.dot(probs, v_mat)  # [Q, D]
            out_row += contrib

        tl.store(OUT_ptrs, out_row, mask=q_mask[:, None] & d_valid[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants for tiling; head_dim is 128
        self.BLOCK_Q = 1
        self.BLOCK_D = 16
        self.BLOCK_K = 64

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA and contiguous
        assert q.is_cuda and k.is_cuda and v.is_cuda
        device = q.device

        total_q = q.shape[0]
        total_kv = k.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[2]

        # Constants from original check
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Check constraints
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        # Prepare output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice tensors
            q_batch = q[q_start:q_end]                         # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]                      # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]                      # [num_kv_tokens, 8, 128]

            # GQA expansion: repeat along head_dim
            k_expanded = k_batch.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]

            # Make contiguous
            q_batch = q_batch.contiguous()
            k_expanded = k_expanded.contiguous()
            v_expanded = v_expanded.contiguous()

            # Allocate logits and segment outputs
            logits = torch.empty((num_q_tokens, num_kv_tokens), dtype=torch.float32, device=device)
            output_seg = torch.empty((num_q_tokens, 32, 128), dtype=torch.float32, device=device)

            # 1) Compute logits: q @ k_expanded^T per (q,h)
            grid_q = (triton.cdiv(num_q_tokens, self.BLOCK_Q), num_qo_heads)
            _compute_logits_kernel[grid_q](
                q_batch, k_expanded, logits,
                logits.stride(0), logits.stride(1), logits.stride(2),
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                num_q_tokens, head_dim, num_kv_tokens,
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D
            )

            # 2) Compute lse per (q,h) with causal mask
            lse_seg = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)
            grid_lse = (triton.cdiv(num_q_tokens, self.BLOCK_Q), num_qo_heads)
            _lse_masked_kernel[grid_lse](
                logits, lse_seg,
                logits.stride(0), logits.stride(1), logits.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                num_q_tokens, head_dim,
                ln2=1.0 / math.log(2.0),
                delta=(num_kv_tokens - num_q_tokens),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K
            )

            # 3) Compute output = softmax(logits) @ v_expanded
            grid_out = (triton.cdiv(num_q_tokens, self.BLOCK_Q), num_qo_heads)
            _softmax_output_kernel[grid_out](
                logits, v_expanded, lse_seg, output_seg,
                logits.stride(0), logits.stride(1), logits.stride(2),
                v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                num_q_tokens, head_dim, num_kv_tokens,
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D, BLOCK_K=self.BLOCK_K
            )

            # Store segment to global output
            output[q_start:q_end] = output_seg.to(torch.bfloat16)
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
