import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_kernel(
    Q, KEXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    KEXP_stride_k, KEXP_stride_h, KEXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens, BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # single element if BLOCK_Q=1
    q_mask = q_offsets < num_q_tokens  # always true for BLOCK_Q=1

    # Accumulator for logits: shape [BLOCK_Q, BLOCK_K], dtype float32
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Loop over d in chunks of BLOCK_D with constexpr bounds
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        # Load Q[q,h,d] for current tile -> shape [BLOCK_Q, BLOCK_D]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d
        Q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_valid[None, :]), other=0.0)

        # For each k in chunk
        for k0 in range(0, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            # Load KEXP[k,h,d] and reduce over d: shape [BLOCK_K, BLOCK_D]
            KEXP_ptrs = KEXP + k_idx[:, None] * KEXP_stride_k + h * KEXP_stride_h + d_idx[None, :] * KEXP_stride_d
            K_vals = tl.load(KEXP_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)

            # Accumulate: acc += sum_d Q_vals * K_vals along d
            # Broadcasting: [1,1] + [BLOCK_Q,1] * [1,BLOCK_K] -> [BLOCK_Q, BLOCK_K]
            acc += tl.sum(Q_vals * K_vals, axis=1)[None, :]

    # Store acc to LOGITS[q,h,k]
    LOGITS_ptrs = LOGITS + q_offsets * LOGITS_stride_q + h * LOGITS_stride_h + tl.arange(0, BLOCK_K) * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=q_mask)


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens, BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # single element if BLOCK_Q=1
    q_mask = q_offsets < num_q_tokens  # always true for BLOCK_Q=1

    # Compute max across K
    max_val = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    for k0 in range(0, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        LOGITS_ptrs = LOGITS + q_offsets * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))
        max_val = tl.maximum(max_val, tl.max(vals, axis=1))

    # Compute sum_exp with causal mask over K
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    for k0 in range(0, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens

        # Causal mask: allowed = k < (q_pos + 1)
        q_pos = q_offsets  # [1]
        allowed = k_idx[None, :] < (q_pos[:, None] + 1)

        LOGITS_ptrs = LOGITS + q_offsets * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
        # Subtract max for numerical stability
        vals = vals - max_val[:, None]
        exp_vals = tl.exp(vals)
        sum_exp += tl.sum(exp_vals, axis=1)

    lse_vals = max_val + tl.log(sum_exp) * 1.4426950408889634  # log(2) reciprocal = 1 / ln(2)
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, VEXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    VEXP_stride_k, VEXP_stride_h, VEXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens, BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # single element if BLOCK_Q=1
    q_mask = q_offsets < num_q_tokens  # always true for BLOCK_Q=1

    # Load lse[q,h]
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [1]

    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets * OUT_stride_q + h * OUT_stride_h + d_idx * OUT_stride_d  # [1, D]
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Compute softmax over K with causal mask and accumulate into out_row
        for k0 in range(0, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            q_pos = q_offsets  # [1]
            allowed = k_idx[None, :] < (q_pos[:, None] + 1)  # [1, K]

            LOGITS_ptrs = LOGITS + q_offsets * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [1, K]
            vals = vals - lse_vals[:, None]  # [1, K]
            exp_vals = tl.exp(vals)
            sum_exp = tl.sum(exp_vals, axis=1)  # [1]
            probs = exp_vals / sum_exp[:, None]  # [1, K]

            VEXP_ptrs = VEXP + k_idx[:, None] * VEXP_stride_k + h * VEXP_stride_h + d_idx[None, :] * VEXP_stride_d
            V_vals = tl.load(VEXP_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]
            # out_row += probs * V_vals across K: [Q, D]
            out_row += tl.sum(probs[:, None, :] * V_vals[None, :, :], axis=1)

        tl.store(OUT_ptrs, out_row, mask=q_mask & d_valid)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants for Triton kernels
        self.BLOCK_Q = 1
        self.BLOCK_K = 64
        self.BLOCK_D = 128

    @torch.no_grad()
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]

        # Constraints/assertions (kept for correctness)
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels require CUDA tensors"

        # GQA ratio
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Output and lse buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Helper: make sure tensors are contiguous and float32 for kernels
        def to_contig3(x):
            return x.contiguous().to(torch.float32)

        q_batch = to_contig3(q)
        # Expand K and V by GQA ratio along head dimension
        k_batch = to_contig3(k).repeat_interleave(gqa_ratio, dim=1)
        v_batch = to_contig3(v).repeat_interleave(gqa_ratio, dim=1)

        # Segment processing
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Prepare expanded K and V for this segment
            k_exp_batch = k_batch[kv_start:kv_end]  # [num_kv_tokens, 32, 128]
            v_exp_batch = v_batch[kv_start:kv_end]  # [num_kv_tokens, 32, 128]

            # Allocate intermediate
            logits_seg = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Strides (elements, not bytes)
            Q_strides = q_batch.stride()  # [32, 128, 1]
            KEXP_strides = k_exp_batch.stride()  # [32, 128, 1]
            LOGITS_strides = logits_seg.stride()  # [32, num_kv_tokens, 128]
            VEXP_strides = v_exp_batch.stride()  # [32, 128, 1]
            OUT_strides = output[q_start:].stride()  # but we write block via torch segment; better allocate per-block
            # We will write directly to output[q_start:q_end] via torch copy; here we use output as float32

            # Launch Triton kernels
            # 1) Compute logits
            _compute_logits_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), num_qo_heads)](
                q_batch, k_exp_batch, logits_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                Q_strides[0], Q_strides[1], Q_strides[2],
                KEXP_strides[0], KEXP_strides[1], KEXP_strides[2],
                LOGITS_strides[0], LOGITS_strides[1], LOGITS_strides[2],
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D, BLOCK_K=self.BLOCK_K
            )

            # 2) LSE with causal mask
            _lse_masked_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), num_qo_heads)](
                logits_seg, lse[q_start:q_end],
                num_q_tokens, num_kv_tokens, head_dim,
                LOGITS_strides[0], LOGITS_strides[1], LOGITS_strides[2],
                lse.stride(0), lse.stride(1),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K
            )

            # 3) Softmax + output
            out_seg = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)
            _softmax_output_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), num_qo_heads)](
                logits_seg, v_exp_batch, lse[q_start:q_end], out_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                LOGITS_strides[0], LOGITS_strides[1], LOGITS_strides[2],
                VEXP_strides[0], VEXP_strides[1], VEXP_strides[2],
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D, BLOCK_K=self.BLOCK_K
            )

            # Copy segment to global output
            output[q_start:q_end] = out_seg

        # Cast output to bfloat16 as required by original run signature
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
