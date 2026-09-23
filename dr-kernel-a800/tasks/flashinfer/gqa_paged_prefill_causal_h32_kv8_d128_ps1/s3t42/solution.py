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


# Triton kernel: softmax over a 1D vector of length L
# inp_ptr: [L] float32
# out_ptr: [L] float32
# L: tl.constexpr (compile-time)
@triton.jit
def _softmax_1d_kernel(inp_ptr, out_ptr, L: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in range(1, L):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(L):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    inv_sum = 1.0 / sum_exp
    for j in range(L):
        vj = tl.load(inp_ptr + j)
        out_val = tl.exp(vj - m) * inv_sum
        tl.store(out_ptr + j, out_val)


# Triton kernel: logsumexp over a 1D vector of length L -> write logsumexp to out_ptr[0]
# inp_ptr: [L] float32
# out_ptr: [1] float32
# L: tl.constexpr (compile-time)
@triton.jit
def _logsumexp_1d_kernel(inp_ptr, out_ptr, L: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in range(1, L):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(L):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    lse = tl.log(sum_exp) + m
    tl.store(out_ptr, lse)


# Triton kernel: matvec out = v_rows @ attn, where v_rows is [HEAD_DIM, NUM_KV], attn is [NUM_KV]
# v_rows_ptr: [HEAD_DIM * NUM_KV] float32, row-major: v_rows[i] row is offset i*NUM_KV : (i+1)*NUM_KV
# attn_ptr: [NUM_KV] float32
# out_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_rows_ptr, attn_ptr, out_ptr,
                   HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for i in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        base = i * NUM_KV
        for j in range(NUM_KV):
            vij = tl.load(v_rows_ptr + base + j)
            aj = tl.load(attn_ptr + j)
            acc += vij * aj
        tl.store(out_ptr + i, acc)


# Triton kernel: cast 1D float32 vector to bfloat16 and store to out_ptr
# inp_ptr: [N] float32
# out_ptr: [N] bfloat16
# N: tl.constexpr (compile-time)
@triton.jit
def _cast_to_bf16_1d_kernel(inp_ptr, out_ptr, N: tl.constexpr):
    for i in range(N):
        vi = tl.load(inp_ptr + i)
        tl.store(out_ptr + i, tl.cast(vi, tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gqa_ratio = 32 // 8  # 4

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes and dtypes
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]
        device = q.device

        # Constants from original assertions
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Output tensors
        output = torch.empty(
            (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
        )

        # Ensure inputs are float32 for compute
        q_f32 = q.to(torch.float32)
        # k_cache and v_cache are [num_pages, 8, 128]; squeeze singleton dim is already handled
        k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

        # Iterate over segments defined by qo_indptr
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            num_q_tokens = q_end - q_start

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_segments_b = int(kv_end - kv_start)

            if num_q_tokens <= 0 or num_segments_b <= 0:
                continue

            # Gather kv_ids for this segment
            kv_ids = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [num_segments_b]

            # Loop over each query position in this segment
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Causal-like bound: max_kv_idx = min(q_idx + 1 + (num_segments_b - num_q_tokens), num_segments_b)
                delta = num_segments_b - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    continue

                # For each query head
                for h in range(num_qo_heads):
                    kv_head = h // self.gqa_ratio  # map to 8 KV heads

                    # Load q_vec [128] as float32
                    q_vec = q_f32[global_q_idx, h]  # [128] float32

                    # Load K/V rows [max_kv_idx, 128] as float32
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]

                    # Compute logits = q_vec @ k_rows.T using Triton dot kernel
                    logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _dot_logits_kernel[(1,)](q_vec, k_rows, logits, HEAD_DIM=128, NUM_KV=max_kv_idx)

                    # Scale logits
                    logits_scaled = logits * sm_scale  # [max_kv_idx] float32

                    # lse = logsumexp(logits_scaled) / ln(2)
                    lse_val_buf = torch.empty((1,), dtype=torch.float32, device=device)
                    _logsumexp_1d_kernel[(1,)](logits_scaled, lse_val_buf, L=max_kv_idx)
                    lse_val = (lse_val_buf[0] / math.log(2.0)).item()  # get scalar
                    lse[global_q_idx, h] = lse_val

                    # attn = softmax(logits_scaled)
                    attn = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _softmax_1d_kernel[(1,)](logits_scaled, attn, L=max_kv_idx)

                    # Compute output = attn @ v_rows -> [128] using Triton matvec kernel
                    out_vec = torch.empty((128,), dtype=torch.float32, device=device)
                    v_rows_flat = v_rows.reshape(-1)  # [max_kv_idx * 128]
                    _matvec_kernel[(1,)](v_rows_flat, attn, out_vec, HEAD_DIM=128, NUM_KV=max_kv_idx)

                    # Cast to bfloat16 and store
                    out_vec_bf16 = torch.empty((128,), dtype=torch.bfloat16, device=device)
                    _cast_to_bf16_1d_kernel[(1,)](out_vec, out_vec_bf16, N=128)
                    output[global_q_idx, h] = out_vec_bf16

        return output, lse


def run(*args):
    return ModelNew()(*args)
