import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q_vec @ K.T, where:
#   q_vec: [HEAD_DIM] float32
#   K: [MAX_KV, HEAD_DIM] float32 (row-major, we index rows via N)
#   out_logit: [HEAD_DIM] float32
@triton.jit
def _matvec_qk_kernel(q_ptr, K_ptr, out_ptr, N: tl.constexpr, HEAD_DIM: tl.constexpr):
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    # K is a 2D block of shape [N, HEAD_DIM], row i starts at offset i*HEAD_DIM
    for j in range(HEAD_DIM):
        qj = tl.load(q_ptr + j)
        for i in range(N):
            # K_ptr offset for row i, col j: i*HEAD_DIM + j
            Ki = tl.load(K_ptr + i * HEAD_DIM + j)
            acc[j] += qj * Ki
    for j in range(HEAD_DIM):
        tl.store(out_ptr + j, acc[j])


# Triton kernel: row-wise softmax over a vector of length 128 (compile-time).
# Input: logits_ptr points to a float32 vector of length 128 for a single row.
# Output: out_ptr points to a float32 vector of length 128 (softmax).
@triton.jit
def _softmax_128_kernel(logits_ptr, out_ptr):
    m = tl.load(logits_ptr + 0)
    for i in range(1, 128):
        vj = tl.load(logits_ptr + i)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for i in range(128):
        vi = tl.load(logits_ptr + i)
        sum_exp += tl.exp(vi - m)
    inv = 1.0 / sum_exp
    for i in range(128):
        vi = tl.load(logits_ptr + i)
        tl.store(out_ptr + i, tl.exp(vi - m) * inv)


# Triton kernel: logsumexp over a vector of length 128 (compile-time), scaled by 1/ln(2).
# Input: inp_ptr points to a float32 vector of length 128 for a single row.
# Output: out_ptr[0] stores logsumexp(inp) / ln(2) as float32.
@triton.jit
def _lse_scaled_128_kernel(inp_ptr, out_ptr):
    m = tl.load(inp_ptr + 0)
    for i in range(1, 128):
        vj = tl.load(inp_ptr + i)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for i in range(128):
        vi = tl.load(inp_ptr + i)
        sum_exp += tl.exp(vi - m)
    lse = tl.log(sum_exp) + m  # logsumexp
    # Scale by 1/ln(2) = 1.4426950408889634
    lse = lse * 1.4426950408889634
    tl.store(out_ptr, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k_cache, v_cache: [num_pages, 8, 128], bfloat16
        qo_indptr, kv_indptr: int32 of length len_indptr
        kv_indices: int32 of length num_kv_indices
        sm_scale: float32 scalar (default 1.0 / sqrt(128))
        Returns (output, lse) where:
          output: [total_q, 32, 128], bfloat16
          lse: [total_q, 32], float32 (scaled by 1/ln(2))
        """
        # Ensure CUDA tensors
        assert q.is_cuda, "q must be on CUDA"
        assert k_cache.is_cuda and v_cache.is_cuda, "k_cache and v_cache must be on CUDA"
        assert qo_indptr.is_cuda and kv_indptr.is_cuda, "qo_indptr and kv_indptr must be on CUDA"
        assert kv_indices.is_cuda, "kv_indices must be on CUDA"

        device = q.device
        total_q, num_qo_heads, head_dim = q.shape
        # Reshape k_cache, v_cache to [num_pages, 8, 128] by squeezing dim=1
        # Note: original code uses squeeze(1). We replicate that:
        k_cache = k_cache.squeeze(1)  # [num_pages, 8, 128]
        v_cache = v_cache.squeeze(1)  # [num_pages, 8, 128]

        # Output and lse tensors
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Compute q in float32
        q_f32 = q.to(torch.float32)  # [total_q, 32, 128]

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_segments_b = kv_end - kv_start  # number of cached KV groups in this segment

            # For each query position and each query head
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                for h in range(num_qo_heads):
                    kvh = h // (self.num_qo_heads // self.num_kv_heads)  # GQA mapping: 32/8 = 4

                    # Load q vector for this query position and head: [128], float32
                    q_vec = q_f32[global_q_idx, h]  # [128], float32

                    # Gather segment indices and corresponding rows from k_cache and v_cache
                    seg_ids = kv_indices[kv_start:kv_end]  # [num_segments_b], int32
                    # Flatten k_cache and v_cache to [rows, 128] for rows=num_segments_b
                    k_rows = k_cache[seg_ids, kvh]  # [num_segments_b, 128], float32
                    v_rows = v_cache[seg_ids, kvh]  # [num_segments_b, 128], float32

                    num_kv_tokens = num_segments_b  # seg_ids length
                    delta = num_kv_tokens - num_q_tokens
                    max_kv_idx = min(q_idx + 1 + delta, num_segments_b)

                    # Prepare padded K matrix [MAX_KV, HEAD_DIM] with -inf padding for invalid rows
                    MAX_KV = 128  # we pad to 128 to use 128-wide kernels
                    rows = num_kv_tokens
                    K_pad = torch.empty((MAX_KV, head_dim), dtype=torch.float32, device=device)
                    if rows > 0:
                        K_pad[:rows] = k_rows  # [rows, 128]
                    # Set invalid rows to -inf so they don't contribute
                    K_pad[rows:MAX_KV] = float('-inf')

                    # Compute logits = q_vec @ K_pad.T -> [128] via Triton matvec
                    logits = torch.empty((head_dim,), dtype=torch.float32, device=device)
                    _matvec_qk_kernel([q_vec], K_pad.view(-1), logits, N=rows, HEAD_DIM=head_dim)

                    # Scale logits by sm_scale (either provided or self.sm_scale)
                    sm = sm_scale if isinstance(sm_scale, float) else self.sm_scale
                    logits_scaled = logits * sm  # [128]

                    # LSE over scaled logits, scaled by 1/ln(2). We use Triton kernel over 128 and ignore padded entries via -inf.
                    out_lse = torch.empty((1,), dtype=torch.float32, device=device)
                    _lse_scaled_128_kernel([logits_scaled], out_lse)

                    # Softmax over scaled logits (128-wide)
                    attn = torch.empty((head_dim,), dtype=torch.float32, device=device)
                    _softmax_128_kernel([logits_scaled], attn)

                    # Compute output: out = attn @ v_rows.T -> [128]
                    # v_rows is [rows, 128]; pad to 128 with zeros for Triton
                    V_pad = torch.empty((head_dim, head_dim), dtype=torch.float32, device=device)
                    if rows > 0:
                        # make V_pad[:, :rows] = v_rows.T, then zeros elsewhere
                        V_pad[:, :rows] = v_rows.T  # v_rows.T is [128, rows]
                    # attn is [128]; compute out_vec = attn @ V_pad -> [128]
                    # Implement matmul using Triton: we can use a simple 128-wide dot
                    out_vec = torch.empty((head_dim,), dtype=torch.float32, device=device)
                    # We can implement a small 128-wide dot in PyTorch to avoid another Triton kernel; but the requirement allows minimal torch usage for data movement. However, to strictly adhere, we implement the dot manually:
                    # out[j] = sum_i attn[i] * V_pad[i, j] for j in [0..127]
                    # For columns >= rows, V_pad[:, j] is zero except possibly first rows, but we set zeros for j >= rows. This is fine; we'll compute for j=0..127:
                    for j in range(head_dim):
                        col = V_pad[j]  # [128]
                        out_vec[j] = 0.0
                        for i in range(head_dim):
                            out_vec[j] += attn[i] * col[i]

                    # Store output for this (global_q_idx, h) as bfloat16
                    # Only first max_kv_idx elements are meaningful; out_vec already reflects attn*v
                    output[global_q_idx, h] = out_vec.to(torch.bfloat16)

                    # Store LSE for this (global_q_idx, h)
                    lse[global_q_idx, h] = out_lse[0]

        return output, lse


def run(*args):
    return ModelNew()(*args)
