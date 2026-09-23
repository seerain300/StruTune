import math
import torch
import triton
import triton.language as tl

# Triton kernels: all math inside Triton, no runtime-dependent loops.

@triton.jit
def compute_logits_kernel(
    Q, K, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_stride_k, K_stride_h, K_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (heads, q_tiles, k_tiles)
    h = tl.program_id(0)
    q_tile = tl.program_id(1)  # since BLOCK_Q=128, q_tiles = 1
    k_tile = tl.program_id(2)  # since BLOCK_K=128, k_tiles = 1

    # We process all q and k in one program since tiles are 128 and head_dim=128.
    # For q, we iterate constexpr d and for each q value, accumulate into LOGITS.
    for d in range(0, 128):
        # q indices vector (BLOCK_Q=128), mask for q range
        q_idx = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_idx < num_q_tokens

        # Load q[q, h, d]
        Q_ptrs = Q + q_idx * Q_stride_q + h * Q_stride_h + d * Q_stride_d
        q_vals = tl.load(Q_ptrs, mask=q_mask, other=0.0)  # [BLOCK_Q]

        # k indices vector (BLOCK_K=128), mask for k range
        k_idx = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens

        # Load k[k, h, d]
        K_ptrs = K + k_idx * K_stride_k + h * K_stride_h + d * K_stride_d
        k_vals = tl.load(K_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Outer product and accumulate: LOGITS[q,h,k] += q_vals[:, None] * k_vals[None, :]
        LOGITS_ptrs = LOGITS + q_idx[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        # Broadcast: q_vals[:, None] -> [BLOCK_Q, 1], k_vals[None, :] -> [1, BLOCK_K]
        tl.store(LOGITS_ptrs, q_vals[:, None] * k_vals[None, :], mask=(q_mask[:, None] & k_mask[None, :]))


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    ln2, delta,  # delta = num_kv_tokens - num_q_tokens (runtime scalar)
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (num_q_tokens, heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Accumulate max and sum-exp over K for numerical stability
    max_val = -float("inf")
    sum_exp = 0.0

    # Tile over K with constexpr BLOCK_K
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens

        # Causal mask: allowed if k < q + 1 + delta
        q_pos = q
        allowed = k_idx < (q_pos + 1 + delta)

        LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(k_mask & allowed), other=-float("inf"))  # [BLOCK_K]
        # Max across this tile
        tile_max = tl.max(vals, axis=0)
        # Sum exp(vals - max) across this tile
        sum_exp += tl.sum(tl.exp(vals - tile_max), axis=0)
        max_val = tl.maximum(max_val, tile_max)

    lse_val = max_val + tl.log(sum_exp) * ln2
    LSE_ptrs = LSE + q * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_val)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_stride_k, V_stride_h, V_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (num_q_tokens, heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Load lse for this (q,h)
    LSE_ptrs = LSE + q * LSE_stride_q + h * LSE_stride_h
    lse_val = tl.load(LSE_ptrs)  # scalar

    # Compute output[q,h,d] across D
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q * OUT_stride_q + h * OUT_stride_h + d_idx * OUT_stride_d
        out_row = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Softmax over K with causal mask
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            q_pos = q
            allowed = k_idx < (q_pos + 1)  # [BLOCK_K]

            LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(k_mask & allowed), other=-float("inf"))  # [BLOCK_K]
            # Subtract lse for stability
            vals = vals - lse_val

            probs = tl.exp(vals)
            sum_probs = tl.sum(probs, axis=0)  # scalar
            probs = probs / sum_probs  # [BLOCK_K]

            # Load V[k,h,d] for this d tile and accumulate
            V_ptrs = V + k_idx * V_stride_k + h * V_stride_h + d0 * V_stride_d
            V_vals = tl.load(V_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

            # Multiply probs and V_vals elementwise and reduce
            for j in range(0, 128):
                out_row[j] += tl.sum(probs * V[j + d0], axis=0)  # reduce across K

        # Store output
        tl.store(OUT_ptrs, out_row, mask=d_valid)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0 / math.sqrt(128.0)):
        super().__init__()
        self.sm_scale = sm_scale  # not used in Triton path; kept for API consistency
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Cast inputs to float32 for computation
        device = q.device
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Expand K and V by GQA ratio (num_qo_heads // num_kv_heads) to 32 heads
        gqa_ratio = self.num_qo_heads // self.num_kv_heads
        k_expanded = k_f32.repeat_interleave(gqa_ratio, dim=1)
        v_expanded = v_f32.repeat_interleave(gqa_ratio, dim=1)

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        assert q_f32.shape[0] == total_q, "total_q mismatch"
        assert k_f32.shape[0] == total_kv, "total_kv mismatch"

        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.numel()
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Extract batched tensors for this segment
            q_batch = q_f32[q_start:q_end]                       # [num_q_tokens, 32, 128]
            k_batch = k_expanded[kv_start:kv_end]               # [num_kv_tokens, 32, 128]
            v_batch = v_expanded[kv_start:kv_end]               # [num_kv_tokens, 32, 128]

            # Allocate segment output and lse
            output_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

            # 1) Compute LOGITS[q,h,k] = sum_d q[q,h,d] * k[k,h,d]
            # Grid over (heads, q_tiles=1, k_tiles=1)
            compute_logits_kernel[(self.num_qo_heads, 1, 1)](
                q_batch, k_batch, output_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_batch.stride(0), k_batch.stride(1), k_batch.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=128, BLOCK_K=128, BLOCK_D=128
            )

            # 2) Compute lse[q,h] = logsumexp with causal mask
            _lse_masked_kernel[(num_q_tokens, self.num_qo_heads)](
                output_seg, lse_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                self.ln2, (num_kv_tokens - num_q_tokens),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                BLOCK_Q=1, BLOCK_K=128
            )

            # 3) Compute output = softmax(logits) @ V_expanded
            _softmax_output_kernel[(num_q_tokens, self.num_qo_heads)](
                output_seg, v_batch, lse_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                v_batch.stride(0), v_batch.stride(1), v_batch.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=1, BLOCK_K=128, BLOCK_D=128
            )

            # Copy segment results to global output/lse
            output[q_start:q_end] = output_seg
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
