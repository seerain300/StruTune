import torch
import math
import triton
import triton.language as tl


@triton.jit
def _segment_attention_kernel(
    Q, K_EXP, V_EXP, OUT, LSE,
    num_q_tokens, num_kv_tokens,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    SM_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,  # e.g., 16
    BLOCK_K: tl.constexpr,  # e.g., 64
):
    # One program per (q index, head) in the segment
    pid_q = tl.program_id(0)  # q index in [0, num_q_tokens)
    h = tl.program_id(1)      # head index in [0, 32)

    q = pid_q  # since grid uses num_q_tokens along dim 0
    q_valid = q < num_q_tokens

    # 1) Compute logits[Q, K] for this q and all heads h over K dimension
    #    logits[q, k] = sum_d Q[q, h, d] * K_EXP[k, h, d] * sm_scale
    K_total = num_kv_tokens

    # We need to accumulate logits for all k. We'll build a matrix [Q, K_total].
    # Note: Triton allows loops with range(128) because 128 is a compile-time constant here.
    logits_qk = tl.zeros((1, K_total), dtype=tl.float32)  # dummy init; we'll fill via tiled loads

    # Precompute d offsets for BLOCK_D tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [D]
        d_mask = d_idx < 128

        # Load Q[q, h, d] as vector
        Q_ptrs = Q + q * Q_stride_q + h * Q_stride_h + d_idx * Q_stride_d
        q_vec = tl.load(Q_ptrs, mask=d_mask, other=0.0)  # [D]

        # For each k in K tiles, compute dot with K_EXP[k, h, d]
        for k0 in range(0, K_total, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [K]
            k_mask = k_idx < K_total

            K_ptrs = K_EXP + k_idx * K_EXP_stride_k + h * K_EXP_stride_h + d_idx[None, :] * K_EXP_stride_d  # [K, D]
            k_mat = tl.load(K_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [K, D]

            # Accumulate dot: (q_vec * k_mat) reduced over D -> [K]
            # q_vec is [D]; k_mat is [K, D] => q_vec[:, None] * k_mat => [D, K]; reduce over axis=0
            # But we need a vector of size [K], so use matmul-like reduction:
            # Compute sum_d q_vec[d] * k_mat[k, d] -> [K]
            # Triton supports elementwise multiply and sum reduction via tl.sum
            dot = tl.sum(q_vec[None, :] * k_mat, axis=1)  # [K]
            logits_qk += dot  # accumulate across d tiles

    # Apply SM_SCALE
    logits_qk = logits_qk * SM_SCALE

    # 2) Apply causal mask: only k < q are valid; k >= q -> -inf
    #    Since q is a scalar program id, mask per k: k < q
    for k0 in range(0, K_total, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [K]
        k_mask = k_idx < K_total

        mask_ptrs = tl.zeros((1,), dtype=tl.int1)  # dummy
        causal_mask = k_idx < q  # [K]
        # Update logits_qk with masked values
        # We need to apply mask to existing logits_qk; recompute per tile if needed.
        # Simpler: keep logits_qk unchanged and apply mask when computing softmax; we'll incorporate mask into softmax step.
        # Instead, set invalid entries to -inf before softmax. We'll do it once after we have all logits.
        # To do that, we need to store logits to a buffer; however, Triton kernel currently only computes them in registers.
        # Triton does not allow direct modification of outputs here; so we'll compute softmax with mask by treating invalid entries as -inf during softmax reduction.

    # For now, keep logits_qk as is; we will apply mask implicitly in softmax by setting logits_qk[k] = -inf where k >= q.

    # Apply mask: set k >= q to -inf
    # We need a vector logits_qk; but Triton doesn't support modifying existing tensors; handle via masked softmax using original logits and masking at reduction time. So we recompute the masked logits per softmax stage.
    # Instead, set invalid entries to -inf directly:
    # Note: Triton kernel currently cannot index into a matrix with a scalar condition across the entire vector; so we skip this here and mask during softmax by feeding original logits and using masked vals in softmax computation.

    # We need to compute lse = logsumexp(logits_qk, axis=1) / ln(2). Since we cannot easily store logits_qk here,
    # we instead compute logits per k and update lse incrementally. But Triton doesn't support dynamic number of k in loops,
    # so we recompute logits per k in softmax stage. To avoid recomputation, we store logits_qk as a tensor.
    # However, Triton JIT kernels don't support returning multiple outputs directly; we'll handle masking in softmax by reconstructing.

    # Therefore, we recompute logits in a way that allows masked softmax. We'll keep logits_qk in a temporary vector updated in two steps:
    # Step A: compute sum exp; Step B: compute max; Step C: compute softmax with mask; Step D: accumulate output.

    # Compute max over K for numerical stability
    max_val = tl.full((), -float("inf"), dtype=tl.float32)
    for k0 in range(0, K_total, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K_total
        # We don't have logits_qk per k here; so we recompute dot products for this k-tile against q_vec and apply SM_SCALE.
        # But we need per-k logits. Triton requires compile-time loops; we can compute per k:
        # For each k in the tile:
        for j in range(0, BLOCK_K):
            # Check if k_idx[j] < K_total
            valid_k = k_idx[j] < K_total
            # Compute q_vec.dot(K_EXP[k_idx[j], h, :]) * SM_SCALE
            # Build d vector
            d_idx = tl.arange(0, 128)
            d_mask = d_idx < 128
            Q_ptrs = Q + q * Q_stride_q + h * Q_stride_h + d_idx * Q_stride_d
            q_vec = tl.load(Q_ptrs, mask=d_mask, other=0.0)  # [128]
            K_ptrs = K_EXP + k_idx[j] * K_EXP_stride_k + h * K_EXP_stride_h + d_idx * K_EXP_stride_d
            k_vec = tl.load(K_ptrs, mask=d_mask, other=0.0)  # [128]
            val = tl.sum(q_vec * k_vec, axis=0) * SM_SCALE  # scalar
            # Update max
            if valid_k:
                max_val = tl.maximum(max_val, val)

    # Compute sum_exp with mask: for k >= q, set val = -inf before exp
    sum_exp = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K_total, BLOCK_K):
        for j in range(0, BLOCK_K):
            valid_k = k_idx[j] < K_total
            d_idx = tl.arange(0, 128)
            d_mask = d_idx < 128
            Q_ptrs = Q + q * Q_stride_q + h * Q_stride_h + d_idx * Q_stride_d
            q_vec = tl.load(Q_ptrs, mask=d_mask, other=0.0)  # [128]
            K_ptrs = K_EXP + k_idx[j] * K_EXP_stride_k + h * K_EXP_stride_h + d_idx * K_EXP_stride_d
            k_vec = tl.load(K_ptrs, mask=d_mask, other=0.0)  # [128]
            val = tl.sum(q_vec * k_vec, axis=0) * SM_SCALE  # scalar
            if valid_k:
                # causal mask
                causal = (k_idx[j] < q)
                val = tl.where(causal, val, -float("inf"))
                sum_exp += tl.exp(val - max_val)

    lse_scalar = max_val + tl.log(sum_exp)  # logsumexp across K for this q,h
    # Store lse to LSE[q, h]
    LSE_ptr = LSE + q * LSE.stride(0) + h * LSE.stride(1)
    tl.store(LSE_ptr, lse_scalar, mask=q_valid)

    # 3) Compute output[q, h, d] = sum_k softmax(logits[q, k]) * V_EXP[k, h, d] for all d in 128
    #    softmax(logits) = exp(logits - lse_scalar) / sum_exp
    #    We will compute output in d tiles.
    ln2 = 0.6931471805599453
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [D]
        d_mask = d_idx < 128

        out_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # For each k in K tiles, compute probs and accumulate
        for k0 in range(0, K_total, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [K]
            k_mask = k_idx < K_total

            # Compute per-k logits and masked softmax
            for j in range(0, BLOCK_K):
                valid_k = k_idx[j] < K_total
                d_idx = tl.arange(0, 128)
                d_mask = d_idx < 128
                Q_ptrs = Q + q * Q_stride_q + h * Q_stride_h + d_idx * Q_stride_d
                q_vec = tl.load(Q_ptrs, mask=d_mask, other=0.0)  # [128]
                K_ptrs = K_EXP + k_idx[j] * K_EXP_stride_k + h * K_EXP_stride_h + d_idx * K_EXP_stride_d
                k_vec = tl.load(K_ptrs, mask=d_mask, other=0.0)  # [128]
                val = tl.sum(q_vec * k_vec, axis=0) * SM_SCALE  # scalar
                if valid_k:
                    causal = (k_idx[j] < q)
                    # set -inf for causal false
                    val = tl.where(causal, val, -float("inf"))
                    prob = tl.exp(val - lse_scalar) / (sum_exp / ln2)  # softmax probability for this k
                else:
                    prob = 0.0

                # V_EXP[k, h, d]
                V_ptrs = V_EXP + k_idx[j] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx * V_EXP_stride_d
                v_vec = tl.load(V_ptrs, mask=d_mask, other=0.0)  # [128]
                out_vec += prob * v_vec

        # Store output[q, h, d] for this d tile
        OUT_ptrs = OUT + q * OUT_stride_q + h * OUT_stride_h + (d0 + tl.arange(0, BLOCK_D)) * OUT_stride_d
        tl.store(OUT_ptrs, out_vec, mask=d_mask)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure device and shapes
        device = q.device
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        # Pre-expand K and V by GQA ratio


def run(*args):
    return ModelNew()(*args)
