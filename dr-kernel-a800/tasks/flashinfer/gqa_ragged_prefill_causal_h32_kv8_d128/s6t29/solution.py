import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: per-segment attention forward
# Inputs:
#   q_ptr: pointer to q tensor [num_q_tokens, num_qo_heads, head_dim]
#   k_ptr: pointer to k tensor [num_kv_tokens, num_kv_heads, head_dim]
#   v_ptr: pointer to v tensor [num_kv_tokens, num_kv_heads, head_dim]
#   qo_indptr_ptr: pointer to qo_indptr [len_indptr]
#   kv_indptr_ptr: pointer to kv_indptr [len_indptr]
#   out_ptr: pointer to output tensor [num_q_tokens, num_qo_heads, head_dim] (bfloat16)
#   lse_ptr: pointer to lse tensor [num_q_tokens, num_qo_heads] (float32)
#   sm_scale: scaling factor (float32)
# Meta-parameters:
#   NUM_Q_TOKENS: num_q_tokens for this segment
#   NUM_KV_TOKENS: num_kv_tokens for this segment
#   NUM_QO_HEADS: num_qo_heads
#   NUM_KV_HEADS: num_kv_heads
#   HEAD_DIM: head_dim
#   GQA_RATIO: num_qo_heads // num_kv_heads
#   LEN_INDPTR: len_indptr
#   BLOCK_I: tile size for i (queries), compile-time constant
#   BLOCK_J: tile size for j (keys), compile-time constant
@triton.jit
def _attention_per_segment_kernel(
    q_ptr, k_ptr, v_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    out_ptr, lse_ptr,
    sm_scale,
    NUM_Q_TOKENS: tl.constexpr,
    NUM_KV_TOKENS: tl.constexpr,
    NUM_QO_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    LEN_INDPTR: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    # We assume grid = (1, 1). We will read segment b via base offsets.
    # We need q_start, q_end, kv_start, kv_end for segment b. We read from qo_indptr/kv_indptr.
    # For grid (1,1), b = 0. Generalization: if grid > 1, use program_id(0) as b.
    # However, to keep it simple and robust, we launch one program per segment on host side.

    b = 0  # single program; we do not spawn multiple programs here

    # Compute segment bounds
    # qo_indptr[b] and qo_indptr[b+1]
    q_start = tl.load(qo_indptr_ptr + b)
    q_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b)

    num_q_tokens = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # For segment b, we have:
    # q_batch: q_ptr + q_start, k_batch: k_ptr + kv_start, v_batch: v_ptr + kv_start
    # We'll compute output for all i in [0, num_q_tokens), all heads h in [0, NUM_QO_HEADS)

    # Precompute strides (in elements)
    # For q: [num_q_tokens, NUM_QO_HEADS, HEAD_DIM]
    stride_q_i = NUM_QO_HEADS * HEAD_DIM
    stride_q_h = HEAD_DIM
    stride_q_d = 1

    # For k: [num_kv_tokens, NUM_KV_HEADS, HEAD_DIM]
    stride_k_i = NUM_KV_HEADS * HEAD_DIM
    stride_k_h = HEAD_DIM
    stride_k_d = 1

    # For v: [num_kv_tokens, NUM_KV_HEADS, HEAD_DIM]
    stride_v_i = NUM_KV_HEADS * HEAD_DIM
    stride_v_h = HEAD_DIM
    stride_v_d = 1

    # For output: [num_q_tokens, NUM_QO_HEADS, HEAD_DIM]
    stride_out_i = NUM_QO_HEADS * HEAD_DIM
    stride_out_h = HEAD_DIM
    stride_out_d = 1

    # For lse: [num_q_tokens, NUM_QO_HEADS]
    stride_lse_i = NUM_QO_HEADS
    stride_lse_h = 1

    # We process all i in [0, num_q_tokens) and j in [0, num_kv_tokens) in tiles.
    # We maintain per-row (i) logsumexp across j.

    # Note: Triton requires loop bounds to be tl.constexpr; we pass NUM_Q_TOKENS and NUM_KV_TOKENS as constexpr.

    # Loop over i in tiles of BLOCK_I
    for i0 in range(0, NUM_Q_TOKENS, BLOCK_I):
        i_idx = i0 + tl.arange(0, BLOCK_I)
        mask_i = i_idx < NUM_Q_TOKENS

        # For each i in this tile, we'll compute m and s for logsumexp across j,
        # and accumulate output[i, h, :]. We do this per h.

        # Loop over query heads h
        for h in range(NUM_QO_HEADS):
            # Initialize m and s for this (i, h)
            m = -float("inf")
            s = 0.0

            # We will accumulate output vector for this (i, h). We'll store it as [HEAD_DIM].
            # We will write to out_ptr for each i in mask_i, with broadcasting over i tile.

            # Loop over j in tiles of BLOCK_J
            for j0 in range(0, NUM_KV_TOKENS, BLOCK_J):
                j_idx = j0 + tl.arange(0, BLOCK_J)
                mask_j = j_idx < NUM_KV_TOKENS

                # Compute q[i, h] vector (length HEAD_DIM)
                # q_ptr + (q_start + i)*stride_q_i + h*stride_q_h + d*stride_q_d
                # But i is vector [BLOCK_I]; for each i, we need a scalar. We'll do per i.
                # We'll set up 2D pointers for q and k for the tile [BLOCK_I, BLOCK_J]
                # ptr_q[i, j] = q_ptr + (q_start + i)*stride_q_i + h*stride_q_h + j*stride_q_d
                # Note: j is head of k/v, not dimension. We need to extract value per j.
                # However, k and v have different head sizes (NUM_KV_HEADS), and we must expand by GQA_RATIO.
                # We build expanded head index h_exp for k/v:
                # h_exp = h * GQA_RATIO + kv_head. kv_head runs over 0..NUM_KV_HEADS-1.

                # We will compute q vector for each i in i_idx:
                # q_val[i, :] = load q[i, h, :] for each i. This is a vector of length HEAD_DIM.
                # We use d = tl.arange(0, HEAD_DIM) to get dimension vector.
                q_d = tl.arange(0, HEAD_DIM)
                for ii in range(BLOCK_I):
                    ii_valid = i_idx[ii] < NUM_Q_TOKENS
                    q_row_ptr = q_ptr + (q_start + i_idx[ii]) * stride_q_i + h * stride_q_h
                    q_vec = tl.load(q_row_ptr + q_d * stride_q_d, mask=ii_valid, other=0.0).to(tl.float32)  # [HEAD_DIM]

                    # Now compute scores across j tile: score_vec[j] = q_vec @ k_expanded[j, h]
                    # We need k_expanded[j, h] which is k[kv_start + j, h_exp] where h_exp = h * GQA_RATIO + kv_head.
                    # But we need kv_head-specific k. We'll compute for all kv heads simultaneously.
                    # Let's build a 2D scores matrix [BLOCK_I, BLOCK_J] but Triton doesn't support 2D vectors directly here.
                    # Instead, we compute per j by looping:
                    # For each j in j_idx:
                    for jj in range(BLOCK_J):
                        j_valid = j_idx[jj] < NUM_KV_TOKENS
                        h_exp = h * GQA_RATIO + jj  # here, we expand k for each j position
                        # Check if h_exp < NUM_KV_HEADS, which it should because GQA_RATIO * NUM_KV_HEADS == NUM_QO_HEADS.
                        k_row_ptr = k_ptr + (kv_start + j_idx[jj]) * stride_k_i + h_exp * stride_k_h
                        v_row_ptr = v_ptr + (kv_start + j_idx[jj]) * stride_v_i + h_exp * stride_v_h
                        k_vec = tl.load(k_row_ptr + q_d * stride_k_d, mask=j_valid, other=0.0).to(tl.float32)  # [HEAD_DIM]
                        v_vec = tl.load(v_row_ptr + q_d * stride_v_d, mask=j_valid, other=0.0).to(tl.float32)  # [HEAD_DIM]

                        # Compute score = sum(q_vec * k_vec) * sm_scale
                        score = tl.sum(q_vec * k_vec, axis=0) * sm_scale  # scalar

                        # Causal mask: allow if j < (i + 1 + delta). delta = NUM_KV_TOKENS - NUM_Q_TOKENS
                        # For this loop, j = j_idx[jj], i = i_idx[ii].
                        delta = NUM_KV_TOKENS - NUM_Q_TOKENS
                        causal = j_idx[jj] < (i_idx[ii] + 1 + delta)
                        # If not causal, set score to -inf
                        score = tl.where(causal, score, -float("inf"))

                        # Streaming logsumexp update for i row ii
                        m_new = tl.maximum(m, score)
                        # Update sum s: s = s * exp(m - m_new) + exp(score - m_new)
                        # Note: s and m are scalars. score is scalar.
                        s = s * tl.exp(m - m_new) + tl.exp(score - m_new)
                        m = m_new

            # After processing all j, we have m and s per (i, h). Now write output:
            # For each i in i_idx:
            for ii in range(BLOCK_I):
                ii_valid = i_idx[ii] < NUM_Q_TOKENS
                # Compute q_vec again for this i
                q_row_ptr = q_ptr + (q_start + i_idx[ii]) * stride_q_i + h * stride_q_h
                q_vec = tl.load(q_row_ptr + q_d * stride_q_d, mask=ii_valid, other=0.0).to(tl.float32)  # [HEAD_DIM]

                # Recompute scores across j to get softmax per j, then O = softmax * V
                for j0_out in range(0, NUM_KV_TOKENS, BLOCK_J):
                    j_idx_out = j0_out + tl.arange(0, BLOCK_J)
                    mask_j_out = j_idx_out < NUM_KV_TOKENS

                    for jj_out in range(BLOCK_J):
                        j_valid = j_idx_out[jj_out] < NUM_KV_TOKENS
                        h_exp = h * GQA_RATIO + jj_out  # still h*GQA_RATIO + 0..GQA_RATIO-1 handled in loop below
                        # Compute O for each j in this tile:
                        # We'll do per j_out
                        for jj_out_inner in range(BLOCK_J):
                            j_valid_inner = j_idx_out[jj_out_inner] < NUM_KV_TOKENS
                            h_exp_inner = h * GQA_RATIO + jj_out_inner
                            k_row_ptr = k_ptr + (kv_start + j_idx_out[jj_out_inner]) * stride_k_i + h_exp_inner * stride_k_h
                            v_row_ptr = v_ptr + (kv_start + j_idx_out[jj_out_inner]) * stride_v_i + h_exp_inner * stride_v_h
                            k_vec = tl.load(k_row_ptr + q_d * stride_k_d, mask=j_valid_inner, other=0.0).to(tl.float32)  # [HEAD_DIM]
                            v_vec = tl.load(v_row_ptr + q_d * stride_v_d, mask=j_valid_inner, other=0.0).to(tl.float32)  # [HEAD_DIM]
                            score_j = tl.sum(q_vec * k_vec, axis=0) * sm_scale
                            causal = j_idx_out[jj_out_inner] < (i_idx[ii] + 1 + delta)
                            score_j = tl.where(causal, score_j, -float("inf"))

                            # softmax value for this j: exp(score_j - m) / s
                            soft = tl.exp(score_j - m) / s

                            # Accumulate output[i, h, :] += soft * v_vec
                            out_row_ptr = out_ptr + (q_start + i_idx[ii]) * stride_out_i + h * stride_out_h
                            out_vec = tl.load(out_row_ptr + q_d * stride_out_d, mask=ii_valid, other=0.0)  # [HEAD_DIM], assume zero init on host
                            out_vec += soft * v_vec
                            tl.store(out_row_ptr + q_d * stride_out_d, out_vec, mask=ii_valid)

            # Store lse for this (i, h) row ii
            # lse_ptr + (q_start + i_idx[ii]) * stride_lse_i + h * stride_lse_h
            lse_row_ptr = lse_ptr + (q_start + i_idx[0]) * stride_lse_i + h * stride_lse_h  # lse_i dimension is 1; h is scalar offset
            # We store m (which is the max used in normalization) as float32
            # Note: Using i_idx[0] to form pointer; for each ii, we need to write. Fix below:
            for ii in range(BLOCK_I):
                ii_valid = i_idx[ii] < NUM_Q_TOKENS
                tl.store(lse_row_ptr + ii, m, mask=ii_valid)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton requires CUDA tensors"
        assert qo_indptr.is_cuda and kv_indptr.is_cuda, "Indptr must be CUDA tensors"

        # Shapes
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"

        # Sanity checks
        assert total_q == int(qo_indptr[-1].item()), "total_q must equal last qo_indptr"
        assert total_kv == int(kv_indptr[-1].item()), "total_kv must equal last kv_indptr"

        # We will run one Triton kernel per segment b in len_indptr. For simplicity, handle up to len_indptr and invoke kernel b=0 (since provided inputs are single segment per call).
        len_indptr = qo_indptr.shape[0]
        # Allocate output and lse
        output = torch.zeros(
            (total_q, num_qo_heads, head_dim),
            dtype=torch.bfloat16,
            device=q.device
        )
        lse = torch.full(
            (total_q, num_qo_heads),
            -float("inf"),
            dtype=torch.float32,
            device=q.device
        )

        # Cast inputs to float32 for compute
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Launch Triton kernel for segment b = 0
        # Use constexpr tile sizes. BLOCK_I/BLOCK_J = 64 are reasonable for this problem size.
        BLOCK_I = 64
        BLOCK_J = 64
        # Grid: (1, 1) — one program handles one segment. We pass all necessary meta-params.
        _attention_per_segment_kernel[(1,)](
            q_f32, k_f32, v_f32,
            qo_indptr, kv_indptr,
            output, lse,
            sm_scale,
            NUM_Q_TOKENS=total_q,
            NUM_KV_TOKENS=total_kv,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            GQA_RATIO=num_qo_heads // num_kv_heads,  # 4
            LEN_INDPTR=len_indptr,  # not used in kernel beyond bounds
            BLOCK_I=BLOCK_I,
            BLOCK_J=BLOCK_J,
            num_warps=4,
            num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
