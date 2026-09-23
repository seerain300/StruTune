import torch
import math
import triton
import triton.language as tl

# Constants from the original code
HEAD_DIM = 128
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS  # 4
LN2 = math.log(2.0)

@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), ceil(num_kv_tokens/BLOCK_K), heads)
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    h = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)   # [BLOCK_Q]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)   # [BLOCK_K]

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulate logits[q,h,k] over d=0..127 in chunks of BLOCK_D
    for d0 in range(0, 128, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)            # [BLOCK_D]
        d_mask = d_offsets < head_dim

        # Load Q[q, h, d] -> [BLOCK_Q, BLOCK_D]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_offsets[None, :] * Q_stride_d
        q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)  # [Q, D], fp32

        # Load K[k, h, d] -> [BLOCK_K, BLOCK_D]
        K_ptrs = K_EXP + k_offsets[:, None] * K_EXP_stride_k + h * K_EXP_stride_h + d_offsets[None, :] * K_EXP_stride_d
        k_vals = tl.load(K_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [K, D], fp32

        # Compute dot(q_vals, k_vals) over D: result shape [Q, K]
        dot = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        for dd in range(0, BLOCK_D):
            dot += q_vals[:, dd] * k_vals[:, dd]

        # Store to LOGITS[q,h,k] for this tile
        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
        tl.store(LOGITS_ptrs, dot, mask=(q_mask[:, None] & k_mask[None, :]))


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    LN2: tl.constexpr,  # division by ln(2)
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)   # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # Accumulate max and sum-exp over K tiles
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    for k0 in range(0, 128, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)            # [BLOCK_K]
        k_mask = k_offsets < num_kv_tokens

        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [Q, K]
        # max reduction across K for each q
        max_vals = tl.maximum(max_vals, tl.max(vals, axis=1))
        # sum exp for logsumexp
        exp_vals = tl.exp(vals - max_vals[:, None])
        sum_exp += tl.sum(exp_vals, axis=1)

    lse_vals = max_vals + tl.log(sum_exp) * LN2  # [Q]

    # Store to LSE[q,h]
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads). We set BLOCK_Q=1 to process one query per program.
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)   # [1]
    q_mask = q_offsets < num_q_tokens  # always true with BLOCK_Q=1

    # Load lse for this (q,h): shape [1]
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [1]

    # Compute output[q,h,d] across d tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)                   # [BLOCK_D]
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Softmax over K tiles with causal mask
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)               # [BLOCK_K]
            k_mask = k_idx < num_kv_tokens

            q_pos = q_offsets                                # [1]
            allowed = k_idx[None, :] < (q_pos[:, None] + 1) # [1, K]

            LOGITS_ptrs = LOGITS + q_pos * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [1, K]
            vals = vals - lse_vals[:, None]                   # [1, K]
            exp_vals = tl.exp(vals)                          # [1, K]
            sum_exp = tl.sum(exp_vals, axis=1)               # [1]
            probs = exp_vals / sum_exp[:, None]              # [1, K]

            V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_vals = tl.load(V_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]
            out_row += tl.sum(probs[:, :, None] * v_vals[None, :, :], axis=1)               # [1, D]

        # Store output for this d tile
        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = NUM_QO_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        device = q.device

        # Compute in fp32
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Expand K and V by GQA ratio (host-side)
        k_exp = k_f32.repeat_interleave(GQA_RATIO, dim=1)
        v_exp = v_f32.repeat_interleave(GQA_RATIO, dim=1)

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert total_q == qo_indptr[-1].item()
        assert total_kv == kv_indptr[-1].item()

        # Allocate outputs
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Iterate segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            q_batch = q_f32[q_start:q_end]  # [Q, 32, 128]
            k_batch = k_exp[kv_start:kv_end]  # [K, 8, 128]
            v_batch = v_exp[kv_start:kv_end]  # [K, 8, 128]

            # Allocate per-segment tensors
            LOGITS_seg = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)
            OUT_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)

            # Tiling parameters
            BLOCK_Q = 4   # tile queries per program
            BLOCK_K = 64  # tile keys per program
            BLOCK_D = 16  # chunk of head_dim for reduction

            # 1) Compute logits for this segment
            grid_log = (triton.cdiv(num_q_tokens, BLOCK_Q), triton.cdiv(num_kv_tokens, BLOCK_K), self.num_qo_heads)
            _compute_logits_kernel[grid_log](
                q_batch, k_batch, LOGITS_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_batch.stride(0), k_batch.stride(1), k_batch.stride(2),
                LOGITS_seg.stride(0), LOGITS_seg.stride(1), LOGITS_seg.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # 2) Compute lse per (q,h) with causal mask
            grid_lse = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _lse_masked_kernel[grid_lse](
                LOGITS_seg, lse[q_start:q_end],
                num_q_tokens, num_kv_tokens, self.head_dim,
                LOGITS_seg.stride(0), LOGITS_seg.stride(1), LOGITS_seg.stride(2),
                lse.stride(0), lse.stride(1),
                LN2=LN2,
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # 3) Compute softmax and output
            grid_out = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _softmax_output_kernel[grid_out](
                LOGITS_seg, v_batch, lse[q_start:q_end], OUT_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                LOGITS_seg.stride(0), LOGITS_seg.stride(1), LOGITS_seg.stride(2),
                v_batch.stride(0), v_batch.stride(1), v_batch.stride(2),
                OUT_seg.stride(0), OUT_seg.stride(1), OUT_seg.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K
            )

            # Store segment outputs
            output[q_start:q_end] = OUT_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
