import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _qkvmatmul_forward_kernel(
    q_ptr,         # *float32, shape [M, D] (we process one q per launch; M can be dynamic but kernel handles general M)
    k_ptr,         # *float32, shape [N, D]
    v_ptr,         # *float32, shape [N, D]
    out_ptr,       # *bfloat16, shape [M, 32, D] (output buffer for this block)
    # block offsets: host provides qo_indptr[b:b+2] and kv_indptr[b:b+2]
    qo_indptr_ptr,  # *int32, [2] = [q_start, q_end]
    kv_indptr_ptr,  # *int32, [2] = [kv_start, kv_end]
    # sizes (constexpr)
    M: tl.constexpr,     # number of queries in this block
    N: tl.constexpr,     # number of key/value tokens in this block
    D: tl.constexpr,     # head_dim (e.g., 128)
    SM_SCALE: tl.constexpr,  # scaling factor (e.g., 1.0 / sqrt(128))
    BLOCK_D: tl.constexpr,    # tile for head dim (e.g., 128)
    BLOCK_N: tl.constexpr,    # tile for N (e.g., 128)
):
    # Load block offsets
    q_start = tl.load(qo_indptr_ptr + 0)
    q_end = tl.load(qo_indptr_ptr + 1)
    kv_start = tl.load(kv_indptr_ptr + 0)
    kv_end = tl.load(kv_indptr_ptr + 1)

    # Process one query index at a time
    for i in range(0, M):
        q_idx = q_start + i

        # Compute q vector for this query: load q[i, :]
        q_vec = tl.zeros((D,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            q_row_base = q_ptr + i * D
            q_sub = tl.load(q_row_base + d_offsets, mask=mask_d, other=0.0)  # [BLOCK_D]
            q_vec[d0:d0+BLOCK_D] = q_sub

        # Compute logits = q @ K^T (scalar for each j)
        logits = tl.full((N,), -float('inf'), tl.float32)
        for j in range(0, N):
            # Load k[j, :] and v[j, :]
            k_row_base = k_ptr + j * D
            v_row_base = v_ptr + j * D
            k_vec = tl.zeros((D,), dtype=tl.float32)
            v_vec = tl.zeros((D,), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                d_offsets = d0 + tl.arange(0, BLOCK_D)
                mask_d = d_offsets < D
                k_sub = tl.load(k_row_base + d_offsets, mask=mask_d, other=0.0)  # [BLOCK_D]
                v_sub = tl.load(v_row_base + d_offsets, mask=mask_d, other=0.0)  # [BLOCK_D]
                k_vec[d0:d0+BLOCK_D] = k_sub
                v_vec[d0:d0+BLOCK_D] = v_sub

            # Dot product q_vec · k_vec
            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_vec[d]
            logits[j] = dot * SM_SCALE

        # Causal mask: j < q_idx + 1 (GQA style, expanded ratio handled by host)
        q_add = q_idx + 1
        for j in range(0, N):
            if j >= q_add:
                logits[j] = -float('inf')

        # Compute attention output: out = softmax(logits) @ V
        max_score = tl.max(logits)
        soft = tl.exp(logits - max_score)  # [N]
        sum_soft = tl.sum(soft)
        soft = soft / sum_soft  # [N]

        # Accumulate output vector
        out_vec = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, N):
            v_row_base = v_ptr + j * D
            v_vec = tl.zeros((D,), dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                d_offsets = d0 + tl.arange(0, BLOCK_D)
                mask_d = d_offsets < D
                v_sub = tl.load(v_row_base + d_offsets, mask=mask_d, other=0.0)  # [BLOCK_D]
                v_vec[d0:d0+BLOCK_D] = v_sub
            out_vec += soft[j] * v_vec

        # Store output for this q_idx: out_ptr[q_idx, 0, :]
        for d0 in range(0, D, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            out_sub = out_vec[d0:d0+BLOCK_D].to(tl.bfloat16)
            tl.store(out_ptr + q_idx * (32 * D) + 0 * D + d_offsets, out_sub, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA and contiguous; use float32 for compute
        device = q.device
        assert TRITON_AVAILABLE, "Triton is not available."
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton requires CUDA tensors."

        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous


def run(*args):
    return ModelNew()(*args)
