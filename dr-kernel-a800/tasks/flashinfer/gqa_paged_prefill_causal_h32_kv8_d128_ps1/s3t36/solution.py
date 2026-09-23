import math
import torch
import triton
import triton.language as tl


# Triton kernel 1: compute logits_scaled = (q_vec @ k_rows.T) * sm_scale
# Writes out a 1D float32 vector of length NUM_KV.
@triton.jit
def _dot_logits_scaled_kernel(q_vec_ptr, k_rows_ptr, out_ptr, sm_scale,
                               HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for i in range(NUM_KV):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            ki_j = tl.load(k_rows_ptr + i * HEAD_DIM + j)
            acc += qj * ki_j
        tl.store(out_ptr + i, acc * sm_scale)


# Triton kernel 2: logsumexp over a 1D vector of length NUM_KV
# inp_ptr: [NUM_KV] float32 containing logits_scaled
# out_ptr: [1] float32, stores lse = logsumexp(inp) / ln(2)
@triton.jit
def _lse_kernel(inp_ptr, out_ptr, inv_ln2,
                NUM_KV: tl.constexpr):
    # Compute max
    m = tl.load(inp_ptr + 0)
    for j in range(1, NUM_KV):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    # Compute sum(exp(inp - m))
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(NUM_KV):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    lse_val = tl.log(sum_exp) + m
    lse_scaled = lse_val * inv_ln2
    tl.store(out_ptr, lse_scaled)


# Triton kernel 3: softmax over a 1D vector of length NUM_KV
# inp_ptr: [NUM_KV] float32 containing logits_scaled
# out_ptr: [NUM_KV] float32, stores attn
@triton.jit
def _softmax_kernel(inp_ptr, out_ptr, inv_ln2,
                     NUM_KV: tl.constexpr):
    # We need m and sum_exp; recompute them here
    m = tl.load(inp_ptr + 0)
    for j in range(1, NUM_KV):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(NUM_KV):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    # Write normalized attn
    for j in range(NUM_KV):
        vj = tl.load(inp_ptr + j)
        attn_j = tl.exp(vj - m) / sum_exp
        tl.store(out_ptr + j, attn_j)


# Triton kernel 4: matvec out = attn @ v_rows
# attn_ptr: [NUM_KV] float32
# v_rows_ptr: [NUM_KV, HEAD_DIM], row-major
# out_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(attn_ptr, v_rows_ptr, out_ptr,
                   HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for i in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(NUM_KV):
            acc += tl.load(attn_ptr + j) * tl.load(v_rows_ptr + j * HEAD_DIM + i)
        tl.store(out_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gqa_ratio = 4  # 32 / 8

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and dtype
        total_q, num_qo_heads, head_dim = q.shape
        q_f32 = q.to(torch.float32).contiguous()
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()

        device = q.device

        # Output buffers: float32 for compute, final cast done at end if needed
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_buf = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]

        # Constants
        inv_ln2 = 1.0 / math.log(2.0)

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_segments_b = int(kv_end - kv_start)
            if num_q_tokens <= 0 or num_segments_b <= 0:
                continue

            kv_ids = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [num_segments_b]

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                delta = num_segments_b - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    # No valid KV for this query position
                    continue

                for h in range(num_qo_heads):
                    kv_head = h // self.gqa_ratio  # GQA mapping: 32 -> 8

                    # Load q_vec [128] float32
                    q_vec = q_f32[global_q_idx, h]  # [128]
                    # Load K/V rows [max_kv_idx, 128] float32
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]

                    # Buffer for logits_scaled
                    logits_scaled = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    # Kernel 1: compute q @ k_rows.T * sm_scale
                    _dot_logits_scaled_kernel[(1,)](
                        q_vec, k_rows, logits_scaled,
                        sm_scale,
                        HEAD_DIM=128, NUM_KV=max_kv_idx
                    )

                    # Buffer for attn and output vector
                    attn = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    out_vec = torch.empty((128,), dtype=torch.float32, device=device)

                    # Kernel 2: lse = logsumexp(logits_scaled) / ln(2)
                    lse_buf[global_q_idx, h] = torch.empty((), dtype=torch.float32, device=device)  # placeholder for clarity
                    _lse_kernel[(1,)](
                        logits_scaled, lse_buf[global_q_idx, h],
                        inv_ln2,
                        NUM_KV=max_kv_idx
                    )
                    # Note: Triton launch stores into lse_buf[global_q_idx, h] directly; no need to read back.

                    # Kernel 3: softmax over logits_scaled -> attn
                    _softmax_kernel[(1,)](
                        logits_scaled, attn,
                        inv_ln2,
                        NUM_KV=max_kv_idx
                    )

                    # Kernel 4: out_vec = attn @ v_rows
                    _matvec_kernel[(1,)](
                        attn, v_rows, out_vec,
                        HEAD_DIM=128, NUM_KV=max_kv_idx
                    )

                    # Store output: cast to bfloat16 for final
                    output[global_q_idx, h] = out_vec  # keep float32; final cast happens below

        # Cast output to bfloat16 as required; lse already float32; return both
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse_buf


def run(*args):
    return ModelNew()(*args)
