import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q_vec @ k_rows.T
# q_vec_ptr: [HEAD_DIM] float32
# k_rows_ptr: [NUM_REAL, HEAD_DIM] float32, row-major (stride along rows is HEAD_DIM)
# logits_out_ptr: [NUM_REAL] float32
@triton.jit
def _dot_kernel(q_vec_ptr, k_rows_ptr, logits_out_ptr,
                HEAD_DIM: tl.constexpr, NUM_REAL: tl.constexpr):
    for i in range(NUM_REAL):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            ki_j = tl.load(k_rows_ptr + i * HEAD_DIM + j)
            acc += qj * ki_j
        tl.store(logits_out_ptr + i, acc)


# Triton kernel: compute logsumexp over a 1D vector of length NUM_REAL
# inp_ptr: [NUM_REAL] float32
# out_ptr: [1] float32
@triton.jit
def _lse_kernel(inp_ptr, out_ptr, NUM_REAL: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in range(1, NUM_REAL):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(NUM_REAL):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)

    lse_val = tl.log(sum_exp) + m  # logsumexp
    tl.store(out_ptr, lse_val)


# Triton kernel: compute softmax over a 1D vector of length NUM_REAL
# inp_ptr: [NUM_REAL] float32
# out_ptr: [NUM_REAL] float32
@triton.jit
def _softmax_kernel(inp_ptr, out_ptr, NUM_REAL: tl.constexpr):
    # compute max
    m = tl.load(inp_ptr + 0)
    for j in range(1, NUM_REAL):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)

    # compute sum of exp
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(NUM_REAL):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)

    # write normalized softmax
    for j in range(NUM_REAL):
        vj = tl.load(inp_ptr + j)
        out_j = tl.exp(vj - m) / sum_exp
        tl.store(out_ptr + j, out_j)


# Triton kernel: matvec out = v_rows @ attn, where v_rows is [HEAD_DIM, NUM_REAL] row-major and attn is [NUM_REAL]
# v_rows_ptr: [HEAD_DIM, NUM_REAL] float32
# attn_ptr: [NUM_REAL] float32
# out_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_rows_ptr, attn_ptr, out_ptr,
                   HEAD_DIM: tl.constexpr, NUM_REAL: tl.constexpr):
    for i in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(NUM_REAL):
            vij = tl.load(v_rows_ptr + i * NUM_REAL + j)
            aj = tl.load(attn_ptr + j)
            acc += vij * aj
        tl.store(out_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gqa_ratio = 4  # 32 // 8

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA and contiguous
        device = q.device
        assert device.type == 'cuda', "ModelNew requires CUDA tensors"
        total_q, num_qo_heads, head_dim = q.shape
        num_kv_heads = 8
        assert num_qo_heads == 32
        assert head_dim == 128

        # Convert q to float32 for compute; keep k_cache/v_cache float32 (bfloat16->float32)
        q_f32 = q.to(torch.float32)
        k_cache_f32 = k_cache.to(torch.float32)
        v_cache_f32 = v_cache.to(torch.float32)

        # Flatten k_cache/v_cache to [num_pages, num_kv_heads, head_dim] -> [num_pages, 8, 128]
        # (already squeezed in original code, but ensure contiguous)
        k_cache_f32 = k_cache_f32.reshape(k_cache_f32.shape[0], num_kv_heads, head_dim).contiguous()
        v_cache_f32 = v_cache_f32.reshape(v_cache_f32.shape[0], num_kv_heads, head_dim).contiguous()

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            # Number of queries and cached segments in this segment
            num_q_tokens = q_end - q_start
            num_segments_b = kv_end - kv_start

            if num_q_tokens <= 0 or num_segments_b <= 0:
                continue

            # Gather kv_ids for this segment
            kv_ids = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [num_segments_b]

            # For each query position q_idx
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Causal-like bound (often 1 for the provided workloads)
                delta = num_segments_b - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    continue

                # Map query head to KV head via GQA
                for h in range(num_qo_heads):
                    kv_head = h // self.gqa_ratio  # 8

                    # Load q_vec [128]
                    q_vec = q_f32[global_q_idx, h]  # [128] float32

                    # Load K/V rows [max_kv_idx, 128]
                    # Note: kv_ids[:max_kv_idx] is safe because kv_ids is [num_segments_b]
                    k_rows = k_cache_f32[kv_ids[:max_kv_idx], kv_head].contiguous()  # [max_kv_idx, 128]
                    v_rows = v_cache_f32[kv_ids[:max_kv_idx], kv_head].contiguous()  # [max_kv_idx, 128]

                    # Compute logits = q_vec @ k_rows.T using Triton _dot_kernel
                    logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    grid = (1,)  # single program handles all accumulation
                    _dot_kernel[grid](
                        q_vec, k_rows, logits,
                        HEAD_DIM=128, NUM_REAL=max_kv_idx
                    )

                    # Scale logits
                    logits_scaled = logits * sm_scale  # [max_kv_idx] float32

                    # Compute lse = logsumexp(logits_scaled)
                    # Use Triton _lse_kernel with NUM_REAL=max_kv_idx
                    lse_vec = torch.empty((1,), dtype=torch.float32, device=device)
                    _lse_kernel[(1,)](
                        logits_scaled, lse_vec,
                        NUM_REAL=max_kv_idx
                    )
                    lse_val = lse_vec[0]  # scalar float32
                    lse[global_q_idx, h] = lse_val

                    # Compute attn = softmax(logits_scaled) using Triton _softmax_kernel
                    attn = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _softmax_kernel[(1,)](
                        logits_scaled, attn,
                        NUM_REAL=max_kv_idx
                    )

                    # Compute output = attn @ v_rows -> [128] using Triton _matvec_kernel
                    out_vec = torch.empty((head_dim,), dtype=torch.float32, device=device)
                    _matvec_kernel[(1,)](
                        v_rows, attn, out_vec,
                        HEAD_DIM=128, NUM_REAL=max_kv_idx
                    )

                    # Store output[q_idx, h, :] as bfloat16
                    output[global_q_idx, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
