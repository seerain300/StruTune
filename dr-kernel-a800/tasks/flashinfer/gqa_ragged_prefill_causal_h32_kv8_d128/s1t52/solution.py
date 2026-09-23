import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits = Q @ K^T for each (q, h), tile over K and D
# q: [Q, H, D], k: [K, H, D], logits: [Q, H, K]
if TRITON_AVAILABLE:
    @triton.jit
    def _compute_logits_kernel(
        Q, KEXP, LOGITS,
        num_q_tokens, num_kv_tokens, head_dim,
        Q_stride_q, Q_stride_h, Q_stride_d,
        KEXP_stride_k, KEXP_stride_h, KEXP_stride_d,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
        pid_q = tl.program_id(0)
        h = tl.program_id(1)

        q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # since BLOCK_Q=1, q is scalar-like
        q_mask = q < num_q_tokens  # [1] -> always true

        # Accumulator for logits[q, h, k]: shape [BLOCK_K]
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        # Loop over d in constexpr chunks
        for d0 in range(0, 128, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D], constexpr
            d_valid = d_idx < head_dim  # [BLOCK_D], constexpr

            # Load Q[q, h, d] -> shape [BLOCK_K, BLOCK_D]
            # Note: Q is 3D [Q, H, D]; we index q and h scalar, d vector.
            Q_ptrs = Q + q * Q_stride_q + h * Q_stride_h + d_idx * Q_stride_d  # [D]
            # Broadcast Q_vals over K dimension:
            # Build K index vector for this program: k_idx = tl.arange(0, BLOCK_K)
            k_idx = tl.arange(0, BLOCK_K)  # [BLOCK_K]
            # For each kk in k_idx, we need to load Q[q, h, d] -> but we want [BLOCK_K, BLOCK_D]
            # Create 2D pointer by expanding d_idx
            # However, Triton expects 2D loads; we do it by looping over kk and loading scalar per kk, then form 2D manually.
            # Simpler: compute Q_vals = tl.load(Q_ptrs, mask=d_valid, other=0.0) which yields [BLOCK_D]
            # Then multiply by K_vals along d.
            # Implement reduction explicitly:
            for kk in range(BLOCK_K):
                Q_vals = tl.load(Q + q * Q_stride_q + h * Q_stride_h + d_idx * Q_stride_d, mask=d_valid, other=0.0)  # [BLOCK_D]
                # Load KEXP[kk, h, d] -> [BLOCK_D]
                KEXP_ptrs = KEXP + kk * KEXP_stride_k + h * KEXP_stride_h + d_idx * KEXP_stride_d
                K_vals = tl.load(KEXP_ptrs, mask=d_valid, other=0.0)  # [BLOCK_D]
                acc[kk] += tl.sum(Q_vals * K_vals, axis=0)

        # Store acc into LOGITS[q, h, :]
        LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + tl.arange(0, BLOCK_K) * LOGITS_stride_k
        tl.store(LOGITS_ptrs, acc, mask=(q_mask & (tl.arange(0, BLOCK_K) < num_kv_tokens)))


    # Triton kernel: compute lse[q, h] = logsumexp(logits) / ln(2) with causal mask
    @triton.jit
    def _lse_masked_kernel(
        LOGITS, LSE,
        num_q_tokens, num_kv_tokens, head_dim,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        LSE_stride_q, LSE_stride_h,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
        pid_q = tl.program_id(0)
        h = tl.program_id(1)

        q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # BLOCK_Q=1
        q_mask = q < num_q_tokens

        # Initialize max and sum_exp
        max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
        sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

        # Loop over K in constexpr chunks
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_mask = k_idx < num_kv_tokens

            LOGITS_ptrs = LOGITS + q[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            # Causal mask: k < (q + 1 + delta), delta = num_kv_tokens - num_q_tokens
            delta = num_kv_tokens - num_q_tokens
            allowed = k_idx[None, :] < (q[:, None] + 1 + delta)  # [1, K]
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [1, K]

            # Compute row-wise max and sum of exp(vals - max) with mask
            # max_vals = max(max_vals, vals)
            max_vals = tl.maximum(max_vals, tl.max(vals, axis=1))  # [1]
            # sum_exp += sum(exp(vals - max_vals))
            exp_vals = tl.exp(vals - max_vals[:, None])  # [1, K]
            # Zero out invalid k positions
            exp_vals = tl.where(k_mask[None, :], exp_vals, 0.0)
            sum_exp += tl.sum(exp_vals, axis=1)  # [1]

        # lse = max + log(sum_exp) / ln(2)
        ln2 = 1.0 / 1.4426950408889634  # 1 / log(2)
        lse_vals = max_vals + tl.log(sum_exp) * ln2  # [1]
        # Store to LSE[q, h]
        LSE_ptrs = LSE + q * LSE_stride_q + h * LSE_stride_h
        tl.store(LSE_ptrs, lse_vals, mask=q_mask)


    # Triton kernel: softmax over K with causal mask and compute output[q, h, d] = sum_k softmax * V_exp[k, h, d]
    @triton.jit
    def _softmax_output_kernel(
        LOGITS, VEXP, LSE, OUT,
        num_q_tokens, num_kv_tokens, head_dim,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        VEXP_stride_k, VEXP_stride_h, VEXP_stride_d,
        OUT_stride_q, OUT_stride_h, OUT_stride_d,
        BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        # Grid: (ceil(num_q_tokens/BLOCK_Q), heads), BLOCK_Q=1
        pid_q = tl.program_id(0)
        h = tl.program_id(1)

        q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [1]
        q_mask = q < num_q_tokens

        # Load lse[q, h]
        LSE_ptrs = LSE + q * LSE_stride_q + h * LSE_stride_h
        lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [1]

        # Compute output[q, h, d] for d tiles
        for d0 in range(0, 128, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            d_valid = d_idx < head_dim

            OUT_ptrs = OUT + q[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d  # [1, D]
            out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

            # Softmax over K tiles with causal mask
            for k0 in range(0, 128, BLOCK_K):
                k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                k_mask = k_idx < num_kv_tokens

                LOGITS_ptrs = LOGITS + q[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k  # [1, K]
                delta = num_kv_tokens - num_q_tokens
                allowed = k_idx[None, :] < (q[:, None] + 1 + delta)  # [1, K]
                vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [1, K]

                # Subtract lse for numerical stability
                vals = vals - lse_vals[:, None]  # [1, K]
                exp_vals = tl.exp(vals)          # [1, K]
                sum_exp = tl.sum(exp_vals, axis=1)  # [1]
                probs = exp_vals / sum_exp[:, None]  # [1, K]

                VEXP_ptrs = VEXP + k_idx[:, None] * VEXP_stride_k + h * VEXP_stride_h + d_idx[None, :] * VEXP_stride_d  # [K, D]
                V_vals = tl.load(VEXP_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]
                # sum over K: out_row += probs * V_vals, broadcast probs over D
                out_row += tl.sum(probs[:, None] * V_vals, axis=0)  # [1, D]

            # Store output
            # Note: OUT is float32; we'll cast to bfloat16 in host
            tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        # Triton tiling constants (constexpr)
        self.BLOCK_Q = 1
        self.BLOCK_K = 64
        self.BLOCK_D = 128

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure on CUDA and dtype
        device = q.device
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            # Move to CUDA if available
            if not TRITON_AVAILABLE:
                # Fallback to original PyTorch logic if Triton is unavailable
                # This ensures correctness if no Triton
                # We mimic the original run() semantics here, but since we need Triton,
                # we'll raise NotImplementedError to avoid silent fallback.
                raise RuntimeError("Triton is not available; cannot run Triton-only model.")
            q = q.contiguous().to(torch.float32).to(device)
            k = k.contiguous().to(torch.float32).to(device)
            v = v.contiguous().to(torch.float32).to(device)

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())

        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
        assert q.shape[1] == self.num_qo_heads and k.shape[1] == self.num_kv_heads and v.shape[1] == self.num_kv_heads
        assert q.shape[2] == self.head_dim and k.shape[2] == self.head_dim and v.shape[2] == self.head_dim

        # Output and lse
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Process segments
        for b in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice and expand k, v by GQA ratio
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            k_exp = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_exp = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate logits [Q, H, K]
            logits = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Strides
            Q_strides = q_batch.stride()  # (D, H, Q) -> (128, 1, num_q_tokens)
            KEXP_strides = k_exp.stride()  # (D, H, K) -> (128, 1, num_kv_tokens)
            LOGITS_strides = logits.stride()  # (K, H, Q) -> (num_kv_tokens, 32, num_q_tokens)

            # Launch compute_logits kernel: grid over (Q, H) -> (num_q_tokens, 32)
            grid = (num_q_tokens, self.num_qo_heads)
            _compute_logits_kernel[grid](
                q_batch, k_exp, logits,
                num_q_tokens, num_kv_tokens, self.head_dim,
                Q_strides[2], Q_strides[1], Q_strides[0],  # (Q_stride_q, Q_stride_h, Q_stride_d)
                KEXP_strides[2], KEXP_strides[1], KEXP_strides[0],  # (KEXP_stride_k, KEXP_stride_h, KEXP_stride_d)
                LOGITS_strides[2], LOGITS_strides[1], LOGITS_strides[0],
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K, BLOCK_D=self.BLOCK_D
            )

            # Launch lse_masked kernel: grid over (Q, H)
            grid2 = (num_q_tokens, self.num_qo_heads)
            _lse_masked_kernel[grid2](
                logits, lse[q_start:q_end],  # write into lse at q_start
                num_q_tokens, num_kv_tokens, self.head_dim,
                LOGITS_strides[2], LOGITS_strides[1], LOGITS_strides[0],
                lse.stride(0), lse.stride(1),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K
            )

            # Launch softmax_output kernel to produce output for this segment
            OUT_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
            OUT_strides = OUT_seg.stride()  # (D, H, Q) -> (128, 32, num_q_tokens)
            VEXP_strides = v_exp.stride()  # (D, H, K) -> (128, 1, num_kv_tokens)
            _softmax_output_kernel[(num_q_tokens, self.num_qo_heads)](
                logits, v_exp, lse[q_start:q_end], OUT_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                LOGITS_strides[2], LOGITS_strides[1], LOGITS_strides[0],
                VEXP_strides[2], VEXP_strides[1], VEXP_strides[0],
                OUT_strides[2], OUT_strides[1], OUT_strides[0],
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D, BLOCK_K=self.BLOCK_K
            )

            # Accumulate into output
            output[q_start:q_end] = OUT_seg

        # Return output and lse; lse is float32 as in original
        # Cast output to bfloat16 to match original run output dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
