import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q_vec @ k_rows.T
# q_vec_ptr: [HEAD_DIM] float32
# k_rows_ptr: [NUM_KV, HEAD_DIM] float32, row-major
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
# inp_ptr: [VEC_SIZE] float32 (we pre-fill with actual values and sentinel -1e20 for padded)
# out_ptr: [1] float32
# VEC_SIZE: tl.constexpr (fixed 128)
@triton.jit
def _logsumexp_1d_kernel(inp_ptr, out_ptr, VEC_SIZE: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in range(1, VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    lse = tl.log(sum_exp) + m
    # divide by ln(2)
    tl.store(out_ptr, lse / 0.6931471805599453)


# Triton kernel: softmax over a 1D vector of length VEC_SIZE, write only first NUM_REAL
# inp_ptr: [VEC_SIZE] float32 (we pre-fill with actual values and sentinel -1e20 for padded)
# out_ptr: [VEC_SIZE] float32 (host will read only first NUM_REAL)
# VEC_SIZE: tl.constexpr (fixed 128), NUM_REAL: tl.constexpr actual number of real elements
@triton.jit
def _softmax_1d_kernel(inp_ptr, out_ptr, VEC_SIZE: tl.constexpr, NUM_REAL: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in range(1, VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    for j in range(NUM_REAL):
        vj = tl.load(inp_ptr + j)
        out_val = tl.exp(vj - m) / sum_exp
        tl.store(out_ptr + j, out_val)


# Triton kernel: matvec out = v_rows @ attn, where
# v_rows_ptr: [HEAD_DIM, NUM_KV] float32, row-major
# attn_ptr: [NUM_KV] float32
# out_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_rows_ptr, attn_ptr, out_ptr,
                   HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for r in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(NUM_KV):
            vr_j = tl.load(v_rows_ptr + r * NUM_KV + j)
            aj = tl.load(attn_ptr + j)
            acc += vr_j * aj
        tl.store(out_ptr + r, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, head_dim=128, gqa_ratio=4):
        super().__init__()
        self.HEAD_DIM = head_dim
        self.gqa_ratio = gqa_ratio

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q.device

        # Ensure contiguous and dtype for compute
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # k_cache and v_cache: [num_pages, 8, 128] -> squeeze(1)
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        assert head_dim == self.HEAD_DIM

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Fixed size for Triton reductions
        VEC_SIZE = self.HEAD_DIM  # 128

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_segments_b = int(kv_end - kv_start)

            # Gather kv_ids for this segment
            kv_ids = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [num_segments_b]

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Causal-like bound: delta can be negative or positive
                delta = num_segments_b - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    continue

                for h in range(num_qo_heads):
                    kv_head = h // self.gqa_ratio  # map query head to KV head

                    # Load q_vec as float32 [128]
                    q_vec = q_f32[global_q_idx, h]  # [128] float32

                    # Load K/V rows for this kv_head
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128] float32
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128] float32

                    # Compute logits = q_vec @ k_rows.T -> [max_kv_idx]
                    logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _dot_logits_kernel[(1,)](
                        q_vec, k_rows, logits,
                        HEAD_DIM=self.HEAD_DIM,
                        NUM_KV=max_kv_idx
                    )

                    # Scale logits
                    logits_scaled = logits * sm_scale  # [max_kv_idx]

                    # Prepare padded input for Triton logsumexp: inp_pad [VEC_SIZE=128], fill padded entries with -1e20
                    inp_pad = torch.empty((VEC_SIZE,), dtype=torch.float32, device=device)
                    inp_pad.fill_(-1e20)
                    # Copy actual logits_scaled into first max_kv_idx positions
                    inp_pad[:max_kv_idx] = logits_scaled

                    # Compute lse = logsumexp(logits_scaled) / ln(2)
                    lse_val = torch.empty((1,), dtype=torch.float32, device=device)
                    _logsumexp_1d_kernel[(1,)](
                        inp_pad, lse_val,
                        VEC_SIZE=VEC_SIZE
                    )
                    lse[global_q_idx, h] = lse_val[0]

                    # Prepare padded input for Triton softmax: inp_pad [VEC_SIZE], fill padded entries with -1e20
                    inp_pad_softmax = torch.empty((VEC_SIZE,), dtype=torch.float32, device=device)
                    inp_pad_softmax.fill_(-1e20)
                    inp_pad_softmax[:max_kv_idx] = logits_scaled

                    # Compute attn = softmax(logits_scaled) using Triton; write only first max_kv_idx
                    attn_pad = torch.empty((VEC_SIZE,), dtype=torch.float32, device=device)
                    _softmax_1d_kernel[(1,)](
                        inp_pad_softmax, attn_pad,
                        VEC_SIZE=VEC_SIZE, NUM_REAL=max_kv_idx
                    )
                    attn = attn_pad[:max_kv_idx]  # [max_kv_idx] float32

                    # Compute output = attn @ v_rows -> [128]
                    out_vec = torch.empty((self.HEAD_DIM,), dtype=torch.float32, device=device)
                    _matvec_kernel[(1,)](
                        v_rows, attn, out_vec,
                        HEAD_DIM=self.HEAD_DIM, NUM_KV=max_kv_idx
                    )

                    # Store output as bfloat16
                    output[global_q_idx, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
