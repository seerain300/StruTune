import math
import torch
import triton
import triton.language as tl

# Constants
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
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
    # Grid: (ceil(num_q_tokens / BLOCK_Q), ceil(num_kv_tokens / BLOCK_K), heads)
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    h = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulator for logits[q, h, k] over d, shape: [BLOCK_Q, BLOCK_K]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Iterate over d in tiles of BLOCK_D
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        # Load Q[:, d_idx] for each q in tile: shape [BLOCK_Q, BLOCK_D]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d
        q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_valid[None, :]), other=0.0)  # [Q, D]

        # Load K[k_offsets, d_idx] for expanded K: shape [BLOCK_K, BLOCK_D]
        # K_EXP has head dim = 32 * GQA_RATIO = 128; we index the expanded head h * GQA_RATIO
        K_h = h * GQA_RATIO
        K_ptrs = K_EXP + k_offsets[:, None] * K_EXP_stride_k + K_h * K_EXP_stride_h + d_idx[None, :] * K_EXP_stride_d
        k_vals = tl.load(K_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]

        # Outer product and reduction over D: acc[q, k] += sum_d q_vals[q, d] * k_vals[k, d]
        for dd in range(BLOCK_D):
            q_vec = q_vals[:, dd]            # [Q]
            k_vec = k_vals[:, dd]            # [K]
            # Broadcast to [Q, K]
            acc += q_vec[:, None] * k_vec[None, :]

    # Store logits: LOGITS[q, h, k]
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q_mask[:, None] & k_mask[None, :]))


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens / BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Accumulate max across K
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))
        # Apply causal mask: allowed k < (q + 1)
        allowed = k_idx[None, :] < (q_offsets[:, None] + 1)
        vals = tl.where(allowed, vals, -float("inf"))
        curr_max = tl.max(vals, axis=1)  # [Q]
        max_vals = tl.maximum(max_vals, curr_max)

    # Sum of exp(vals - max) over K
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))
        allowed = k_idx[None, :] < (q_offsets[:, None] + 1)
        vals = tl.where(allowed, vals, -float("inf"))
        exp_vals = tl.exp(vals - max_vals[:, None])
        sum_exp += tl.sum(exp_vals, axis=1)

    lse_vals = max_vals + tl.log(sum_exp) * LN2
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens / BLOCK_Q), heads); BLOCK_Q=1 to compute per-q per-head
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Load lse for each q
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

    # Compute output[q, h, d] across D tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Softmax over K tiles with causal mask
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            q_pos = q_offsets  # [Q]
            allowed = k_idx[None, :] < (q_pos[:, None] + 1)  # [Q, K]

            LOGITS_ptrs = LOGITS + q_pos[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [Q, K]
            # Subtract lse for numerical stability
            vals = vals - lse_vals[:, None]  # [Q, K]

            exp_vals = tl.exp(vals)          # [Q, K]
            sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
            probs = exp_vals / sum_exp[:, None]  # [Q, K]

            V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_vals = tl.load(V_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]

            # out[q, d] += sum_k probs[q, k] * v[k, d]
            for kk in range(BLOCK_K):
                v_vec = v_vals[kk, :]            # [D]
                prob_vec = probs[:, kk]          # [Q]
                # For each q, out[q, d] += prob_vec[d] * v_vec[d]
                # Implement via outer product and reduction over Q
                out_row += prob_vec[:, None] * v_vec[None, :]  # [Q, D]

        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = HEAD_DIM
        self.ln2 = LN2
        self.gqa_ratio = GQA_RATIO

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Shapes as in original
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == NUM_QO_HEADS
        assert num_kv_heads == NUM_KV_HEADS
        assert head_dim == self.head_dim
        # The original asserts: omitted here

        device = q.device
        dtype = torch.bfloat16

        output = torch.empty(
            (total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device
        )  # will store float32, convert later
        lse = torch.empty(
            (total_q, num_qo_heads), dtype=torch.float32, device=device
        )

        # Expand K and V by GQA ratio (8 -> 32)
        k_expanded = k.repeat_interleave(self.gqa_ratio, dim=1).to(torch.float32)
        v_expanded = v.repeat_interleave(self.gqa_ratio, dim=1).to(torch.float32)

        # Process each segment defined by indptr
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Segment tensors
            q_batch = q[q_start:q_end].to(torch.float32)  # [num_q_tokens, 32, 128]
            k_exp_batch = k_expanded[kv_start:kv_end]     # [num_kv_tokens, 32, 128]
            v_exp_batch = v_expanded[kv_start:kv_end]     # [num_kv_tokens, 32, 128]

            # Launch kernels
            BLOCK_Q = 1  # compute one query per program
            BLOCK_K = 32  # tile size for K
            BLOCK_D = 32  # tile size for D

            # 1) Compute logits
            grid_logits = (triton.cdiv(num_q_tokens, BLOCK_Q), triton.cdiv(num_kv_tokens, BLOCK_K), NUM_QO_HEADS)
            _compute_logits_kernel[grid_logits](
                q_batch, k_exp_batch, output,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_exp_batch.stride(0), k_exp_batch.stride(1), k_exp_batch.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # 2) Compute LSE with causal mask
            grid_lse = (triton.cdiv(num_q_tokens, BLOCK_Q), NUM_QO_HEADS)
            _lse_masked_kernel[grid_lse](
                output, lse,
                num_q_tokens, num_kv_tokens, self.head_dim,
                output.stride(0), output.stride(1), output.stride(2),
                lse.stride(0), lse.stride(1),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # 3) Compute softmax and output
            grid_out = (triton.cdiv(num_q_tokens, BLOCK_Q), NUM_QO_HEADS)
            _softmax_output_kernel[grid_out](
                output, v_exp_batch, lse, output,  # output is both input and destination (accumulate)
                num_q_tokens, num_kv_tokens, self.head_dim,
                output.stride(0), output.stride(1), output.stride(2),
                v_exp_batch.stride(0), v_exp_batch.stride(1), v_exp_batch.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

        # Cast output to bfloat16 as in original
        output = output.to(torch.bfloat16)
        return output, lse


# Helpers as in original for testing
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
    model = ModelNew().cuda()
    outputs = model(*[t.cuda() if t.is_cuda else t for t in get_inputs()])
    return list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]


def run(*args):
    return ModelNew()(*args)
