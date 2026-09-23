import math
import triton
import triton.language as tl


# Kernel 1: Compute logits[q, h, k] = sum_d q[q, h, d] * k_exp[k, h, d]
# We process one (q,h) pair per program_id(0) and tile over k with BLOCK_K
@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    NUM_Q, NUM_K, HEAD_DIM,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # One program computes a single q (pid_q) and a single head h (program_id(1))
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # single element since BLOCK_Q=1
    q_mask = q_idx < NUM_Q

    # Accumulator for logits[q, h, k] across d
    # Shape: [BLOCK_Q, BLOCK_K]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Iterate over d in tiles of BLOCK_D
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_idx < HEAD_DIM

        # Load Q[:, d_idx] -> shape [BLOCK_Q, BLOCK_D]
        Q_ptrs = Q + q_idx[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d
        q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)

        # Load K_exp[:, h, d_idx] -> shape [BLOCK_K, BLOCK_D]
        K_ptrs = K_EXP + (tl.arange(0, BLOCK_K)[:, None]) * K_EXP_stride_k + h * K_EXP_stride_h + d_idx[None, :] * K_EXP_stride_d
        k_vals = tl.load(K_ptrs, mask=(tl.arange(0, BLOCK_K)[:, None] < NUM_K), other=0.0)

        # Compute outer product and reduce over D
        # acc += sum_d q[:, d] * k[d, :]
        for dd in range(BLOCK_D):
            # Cast to float32 for numeric stability
            q_col = q_vals[:, dd].to(tl.float32)  # [BLOCK_Q]
            k_row = k_vals[:, dd].to(tl.float32)  # [BLOCK_K]
            acc += q_col[:, None] * k_row[None, :]  # [BLOCK_Q, BLOCK_K]

    # Store acc into LOGITS[q, h, k]
    LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + (tl.arange(0, BLOCK_K)[:, None]) * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q_mask[:, None] & (tl.arange(0, BLOCK_K)[:, None] < NUM_K)))


# Kernel 2: Compute lse[q, h] = logsumexp(LOGITS[q, h, :]) / ln(2) with causal mask
@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    NUM_Q, NUM_K, HEAD_DIM,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    LN2, DELTA,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # One program computes a single q and a single head h
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # BLOCK_Q=1
    q_mask = q_idx < NUM_Q

    # Initialize max and sum
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Reduce over K in tiles
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < NUM_K

        # Causal mask: allowed keys j < (q_pos + 1 + delta)
        q_pos = q_idx  # [1]
        allowed = k_idx[None, :] < (q_pos[:, None] + 1 + DELTA)

        LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
        # Update max and sum
        max_vals = tl.maximum(max_vals, vals)
        exp_vals = tl.exp(vals - max_vals[:, None])
        sum_exp += tl.sum(exp_vals, axis=1)

    lse_vals = max_vals + tl.log(sum_exp) * LN2  # [1]
    LSE_ptrs = LSE + q_idx * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


# Kernel 3: Compute softmax over K for each (q,h) and output[q,h,d] = sum_k softmax[q,h,k] * V_exp[k,h,d]
@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    NUM_Q, NUM_K, HEAD_DIM,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # One program computes a single q and a single head h
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # BLOCK_Q=1

    # Load lse[q,h]
    LSE_ptrs = LSE + q_idx * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=(q_idx < NUM_Q), other=-float("inf"))  # [1]

    # Output across D tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_idx < HEAD_DIM

        OUT_ptrs = OUT + q_idx[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Softmax over K tiles with causal mask
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < NUM_K

            q_pos = q_idx  # [1]
            allowed = k_idx[None, :] < (q_pos[:, None] + 1)  # causal: j < (i + 1)

            LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_idx[:, None] < NUM_Q & k_mask[None, :] & allowed), other=-float("inf"))  # [1, K]
            # Subtract lse for numerical stability
            vals = vals - lse_vals[:, None]  # [1, K]

            exp_vals = tl.exp(vals)          # [1, K]
            sum_exp = tl.sum(exp_vals, axis=1)  # [1]
            probs = exp_vals / sum_exp[:, None]  # [1, K]

            V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_vals = tl.load(V_ptrs, mask=(k_idx[:, None] < NUM_K & d_mask[None, :]), other=0.0)  # [K, D]
            # out_row += probs * v_vals
            # Handle broadcasting: probs[0, :] * v_vals[:, :]
            for kk in range(BLOCK_K):
                p = probs[0, kk]  # scalar
                V_sub = v_vals[kk, :]  # [D]
                out_row[0, :] += p * V_sub

        # Store output row
        tl.store(OUT_ptrs, out_row, mask=(q_idx < NUM_Q & d_mask[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure tensors are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        total_q = q.shape[0]
        total_kv = k.shape[0]
        len_indptr = qo_indptr.shape[0]

        # Constants assertions similar to original
        assert q.shape[1] == self.num_qo_heads
        assert k.shape[1] == self.num_kv_heads
        assert v.shape[1] == self.num_kv_heads
        assert q.shape[2] == self.head_dim
        assert k.shape[2] == self.head_dim
        assert v.shape[2] == self.head_dim
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        # Expand K and V by GQA ratio along head dimension (data movement, not computation)
        k_expanded = k.repeat_interleave(self.gqa_ratio, dim=1).contiguous()
        v_expanded = v.repeat_interleave(self.gqa_ratio, dim=1).contiguous()

        # Prepare output and lse buffers
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=q.device)

        # Process each segment defined by qo_indptr and kv_indptr
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            # Early exit if no tokens
            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slices
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_exp_batch = k_expanded[kv_start:kv_end]  # [num_kv_tokens, 32, 128]
            v_exp_batch = v_expanded[kv_start:kv_end]  # [num_kv_tokens, 32, 128]

            # Kernel launch parameters (fixed tiling to avoid runtime loops)
            BLOCK_Q = 1
            BLOCK_K = 32
            BLOCK_D = 32

            # 1) Compute logits: LOGITS shape [num_q_tokens, 32, num_kv_tokens]
            LOGITS = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=q.device)

            grid = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _compute_logits_kernel[grid](
                q_batch, k_exp_batch, LOGITS,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_exp_batch.stride(0), k_exp_batch.stride(1), k_exp_batch.stride(2),
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            )

            # 2) Compute LSE with causal mask
            grid_lse = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _lse_masked_kernel[grid_lse](
                LOGITS, lse,
                num_q_tokens, num_kv_tokens, self.head_dim,
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                lse.stride(0), lse.stride(1),
                self.ln2, (num_kv_tokens - num_q_tokens),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # 3) Compute softmax + output
            grid_out = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _softmax_output_kernel[grid_out](
                LOGITS, v_exp_batch, lse, output,
                num_q_tokens, num_kv_tokens, self.head_dim,
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                v_exp_batch.stride(0), v_exp_batch.stride(1), v_exp_batch.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K
            )

        return output, lse


# For completeness: same get_inputs as original
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]


# The following helper is not used by the evaluator, but provided for parity
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
