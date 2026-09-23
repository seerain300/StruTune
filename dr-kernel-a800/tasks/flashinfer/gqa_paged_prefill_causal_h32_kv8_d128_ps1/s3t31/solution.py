import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q_vec @ k_rows.T
# q_vec_ptr: [HEAD_DIM] float32
# k_rows_ptr: [NUM_KV, HEAD_DIM] float32, row-major (k[i, :] contiguous for i in 0..NUM_KV-1)
# logits_out_ptr: [NUM_KV] float32
@triton.jit
def _dot_logits_kernel(q_vec_ptr, k_rows_ptr, logits_out_ptr,
                        HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for i in range(NUM_KV):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            ki_j = tl.load(k_rows_ptr + i * HEAD_DIM + j)
            acc += qj * ki_j
        tl.store(logits_out_ptr + i, acc)


# Triton kernel: logsumexp over a 1D vector of length VEC_SIZE
# inp_ptr: [VEC_SIZE] float32
# out_ptr: [1] float32
# VEC_SIZE: tl.constexpr
@triton.jit
def _logsumexp_1d_kernel(inp_ptr, out_ptr, VEC_SIZE: tl.constexpr):
    # Compute max
    m = tl.load(inp_ptr + 0)
    for j in range(1, VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    # Compute sum(exp(inp - m))
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    lse = tl.log(sum_exp) + m  # logsumexp
    ln2 = 0.6931471805599453
    tl.store(out_ptr, lse / ln2)  # divide by ln(2)


# Triton kernel: softmax over a 1D vector of length VEC_SIZE, write to out_ptr
# inp_ptr: [VEC_SIZE] float32
# out_ptr: [VEC_SIZE] float32
# VEC_SIZE: tl.constexpr
@triton.jit
def _softmax_1d_kernel(inp_ptr, out_ptr, VEC_SIZE: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in range(1, VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    for j in range(VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        attn = tl.exp(vj - m) / sum_exp
        tl.store(out_ptr + j, attn)


# Triton kernel: matvec out[r] = sum_j v_mat[r, j] * attn[j], for r in 0..HEAD_DIM-1, j in 0..127
# v_mat_ptr: [HEAD_DIM, 128] float32 (we prefill with v_rows[:NUM_KV, :] in host code)
# attn_ptr: [128] float32 (we pass attn[:NUM_KV] padded with -inf in host)
# out_vec_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_mat_ptr, attn_ptr, out_vec_ptr,
                   HEAD_DIM: tl.constexpr):
    for r in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(128):  # compile-time loop over columns
            a = tl.load(attn_ptr + j)  # attn may have -inf padding for j >= NUM_KV; exp(-inf)=0
            val = tl.load(v_mat_ptr + r * 128 + j)
            acc += val * a
        tl.store(out_vec_ptr + r, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.HEAD_DIM = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device and dtype
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA device"
        device = q.device

        # Convert to float32 and contiguous
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

        total_q = q_f32.shape[0]
        num_qo_heads = self.num_qo_heads
        head_dim = self.HEAD_DIM

        # Output and lse buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Iterate over segments defined by qo_indptr
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            num_q_tokens = q_end - q_start

            # kv indices for this segment
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if kv_start >= kv_end:
                continue
            kv_ids = kv_indices[kv_start:kv_end].contiguous()  # [num_kv_tokens]

            # Loop over queries in this segment
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Compute delta for causal-like limit
                num_segments_b = kv_ids.shape[0]
                delta = num_segments_b - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    continue

                # q batch slice
                q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]

                for h in range(num_qo_heads):
                    kv_head = h // self.gqa_ratio  # 8 possible kv heads

                    # Query vector for this head
                    q_vec = q_batch[q_idx, h]  # [128], float32

                    # K/V rows for this kv_head up to max_kv_idx
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]

                    # Compute logits = q_vec @ k_rows.T -> [max_kv_idx], float32
                    logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _dot_logits_kernel[(1,)](
                        q_vec, k_rows, logits,
                        HEAD_DIM=self.HEAD_DIM,
                        NUM_KV=max_kv_idx,
                        num_warps=1,
                    )

                    # Scale logits
                    logits_scaled = logits * sm_scale

                    # Compute lse = logsumexp(logits_scaled) / ln(2)
                    lse_out = torch.empty((1,), dtype=torch.float32, device=device)
                    _logsumexp_1d_kernel[(1,)](
                        logits_scaled, lse_out,
                        VEC_SIZE=max_kv_idx,
                        num_warps=1,
                    )
                    lse[global_q_idx, h] = lse_out[0]

                    # Compute attn = softmax(logits_scaled)
                    attn_out = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _softmax_1d_kernel[(1,)](
                        logits_scaled, attn_out,
                        VEC_SIZE=max_kv_idx,
                        num_warps=1,
                    )

                    # Compute out = v_rows @ attn_out -> [128] via Triton matvec
                    # Prepare v_mat: [HEAD_DIM, 128], fill first NUM_KV columns with v_rows (transposed),
                    # and pad attn_out with -inf for j >= NUM_KV so it contributes zero.
                    v_mat = torch.empty((self.HEAD_DIM, 128), dtype=torch.float32, device=device)
                    for r in range(self.HEAD_DIM):
                        for j in range(max_kv_idx):
                            v_mat[r, j] = v_rows[j, r]
                    # Pad attn with -inf for j >= NUM_KV
                    attn_padded = torch.empty((128,), dtype=torch.float32, device=device)
                    attn_padded[:max_kv_idx] = attn_out
                    attn_padded[max_kv_idx:] = float("-inf")

                    out_vec = torch.empty((self.HEAD_DIM,), dtype=torch.float32, device=device)
                    _matvec_kernel[(1,)](
                        v_mat, attn_padded, out_vec,
                        HEAD_DIM=self.HEAD_DIM,
                        num_warps=1,
                    )

                    # Store output vector for this (query, head)
                    output[global_q_idx, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
