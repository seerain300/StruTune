import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute out_vec = v_rows @ attn, where
# v_mat_ptr: [HEAD_DIM, NUM_KV] float32, row-major (r=0..HEAD_DIM-1, c=0..NUM_KV-1)
# attn_ptr:  [NUM_KV] float32
# out_vec_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_mat_ptr, attn_ptr, out_vec_ptr,
                   HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    # out[r] = sum_{j=0}^{NUM_KV-1} v[r, j] * attn[j]
    for r in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(NUM_KV):
            vr_j = tl.load(v_mat_ptr + r * NUM_KV + j)
            aj = tl.load(attn_ptr + j)
            acc += vr_j * aj
        tl.store(out_vec_ptr + r, acc)


# Triton kernel: compute logits = q_vec @ k_rows.T
# q_vec_ptr: [HEAD_DIM] float32
# k_mat_ptr: [NUM_KV, HEAD_DIM] float32, row-major
# logits_out_ptr: [NUM_KV] float32
@triton.jit
def _dot_logits_kernel(q_vec_ptr, k_mat_ptr, logits_out_ptr,
                        HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    # For each i in [0, NUM_KV), compute logits[i] = sum_j q[j] * k[i, j]
    for i in range(NUM_KV):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            ki_j = tl.load(k_mat_ptr + i * HEAD_DIM + j)
            acc += qj * ki_j
        tl.store(logits_out_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.HEAD_DIM = 128
        self.gqa_ratio = 32 // 8  # 4

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q.device
        total_q = q.shape[0]
        num_qo_heads = q.shape[1]
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert q.dtype == torch.bfloat16, "q must be bfloat16"

        # Flatten cached K/V to [num_pages, num_kv_heads, head_dim]
        k_cache_flat = k_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]

        # Output buffers
        output = torch.zeros(
            (total_q, num_qo_heads, self.HEAD_DIM),
            dtype=torch.bfloat16, device=device
        )
        lse = torch.full((total_q, num_qo_heads), -float("inf"),
                         dtype=torch.float32, device=device)

        ln2 = 0.6931471805599453

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

            # q_batch [num_q_tokens, 32, 128]
            q_batch = q[q_start:q_end].contiguous()

            # Segment kv ids
            kv_ids = kv_indices[kv_start:kv_end].contiguous()  # [num_kv_tokens]

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Causal-like bound
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                if max_kv_idx <= 0:
                    continue

                for h in range(num_qo_heads):
                    kv_head = h // self.gqa_ratio  # map 32 heads to 8 kv heads

                    # q_vec: [128] float32
                    q_vec = q_batch[q_idx, h].contiguous().to(torch.float32)  # [128]

                    # k_rows, v_rows: [max_kv_idx, 128] float32
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head].to(torch.float32).contiguous()  # [max_kv_idx, 128]
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head].to(torch.float32).contiguous()  # [max_kv_idx, 128]

                    # Compute logits = q_vec @ k_rows.T -> [max_kv_idx]
                    logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _dot_logits_kernel[(1,)](
                        q_vec, k_rows, logits,
                        HEAD_DIM=self.HEAD_DIM, NUM_KV=max_kv_idx
                    )

                    # Scale
                    logits_scaled = logits * sm_scale

                    # lse = logsumexp(logits_scaled) / ln(2)
                    lse_val = torch.logsumexp(logits_scaled, dim=0) / ln2
                    lse[global_q_idx, h] = lse_val

                    # attn = softmax(logits_scaled)
                    attn = torch.softmax(logits_scaled, dim=0)

                    # out_vec = v_rows @ attn -> [128]
                    out_vec = torch.empty((self.HEAD_DIM,), dtype=torch.float32, device=device)
                    _matvec_kernel[(1,)](
                        v_rows, attn, out_vec,
                        HEAD_DIM=self.HEAD_DIM, NUM_KV=max_kv_idx
                    )

                    # Store output (cast to bfloat16)
                    output[global_q_idx, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
