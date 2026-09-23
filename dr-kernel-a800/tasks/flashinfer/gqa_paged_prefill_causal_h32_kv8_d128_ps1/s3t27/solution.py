import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q_vec @ k_mat.T
# q_vec_ptr: [HEAD_DIM] float32
# k_mat_ptr: [NUM_KV * HEAD_DIM] float32, flattened; each row is HEAD_DIM elements
# logits_out_ptr: [NUM_KV] float32
@triton.jit
def _dot_logits_kernel(q_vec_ptr, k_mat_ptr, logits_out_ptr,
                        HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for i in range(NUM_KV):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            # Row i in k_mat_ptr starts at i * HEAD_DIM
            ki_j = tl.load(k_mat_ptr + i * HEAD_DIM + j)
            acc += qj * ki_j
        tl.store(logits_out_ptr + i, acc)


# Triton kernel: compute out_vec = v_mat @ attn_vec
# v_mat_ptr: [HEAD_DIM * NUM_KV] float32, row-major, each row r has NUM_KV elements
# attn_ptr: [NUM_KV] float32
# out_vec_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_mat_ptr, attn_ptr, out_vec_ptr,
                   HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for r in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(NUM_KV):
            v_rj = tl.load(v_mat_ptr + r * NUM_KV + j)
            aj = tl.load(attn_ptr + j)
            acc += v_rj * aj
        tl.store(out_vec_ptr + r, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_ln2 = 1.0 / math.log(2.0)

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA device."
        device = q.device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes (original assumptions)
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Flatten K/V cache to [num_pages, num_kv_heads, head_dim]
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, 8, 128]

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start
            if num_kv_tokens == 0:
                continue

            # Gather kv_ids for this segment
            page_ids = kv_indices[kv_start:kv_end].to(torch.int64)  # [num_kv_tokens]

            # Iterate over each query token
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Causal-like bound: max number of KV rows to use
                delta = int(num_kv_tokens - num_q_tokens)
                max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                if max_kv_idx <= 0:
                    continue

                # Query vector for this (query, head)
                q_pos = q[global_q_idx]  # [32, 128]
                for h in range(num_qo_heads):
                    kv_head = h // (num_qo_heads // num_kv_heads)  # GQA mapping

                    # q_vec: [128] float32
                    q_vec = q_pos[h].contiguous()
                    q_vec = q_vec.to(torch.float32)

                    # Gather K/V rows for this head for max_kv_idx
                    k_rows = k_cache_flat[page_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]
                    v_rows = v_cache_flat[page_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]

                    # Compute logits = q_vec @ k_rows.T -> [max_kv_idx]
                    NUM_KV = int(max_kv_idx)
                    k_mat = k_rows.to(torch.float32).contiguous()  # [NUM_KV, 128]
                    logits = torch.empty((NUM_KV,), dtype=torch.float32, device=device)
                    _dot_logits_kernel[(1,)](
                        q_vec, k_mat.view(NUM_KV * 128), logits,
                        HEAD_DIM=128, NUM_KV=NUM_KV
                    )

                    # Scale and compute logsumexp (torch ops on GPU)
                    logits_scaled = logits * sm_scale
                    lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=0) * self.inv_ln2

                    # attn = softmax(logits_scaled, dim=0) -> [NUM_KV]
                    attn = torch.softmax(logits_scaled, dim=0)  # float32 on GPU

                    # Compute out = v_rows @ attn -> [128]
                    v_mat = v_rows.to(torch.float32).contiguous()  # [NUM_KV, 128]
                    out_vec = torch.empty((128,), dtype=torch.float32, device=device)
                    _matvec_kernel[(1,)](
                        v_mat.view(128 * NUM_KV), attn, out_vec,
                        HEAD_DIM=128, NUM_KV=NUM_KV
                    )

                    # Store output
                    output[global_q_idx, h] = out_vec

        # Cast output to bfloat16 as original returns
        output = output.to(torch.bfloat16)
        # lse remains float32 as original
        return output, lse


def run(*args):
    return ModelNew()(*args)
