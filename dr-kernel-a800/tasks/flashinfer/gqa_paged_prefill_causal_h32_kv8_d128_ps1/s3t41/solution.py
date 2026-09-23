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


# Triton kernel: matvec out = v_rows @ attn, where
# v_rows_ptr: [HEAD_DIM, NUM_KV] float32, row-major (stride_v = NUM_KV)
# attn_ptr: [NUM_KV] float32
# out_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_rows_ptr, attn_ptr, out_ptr,
                   HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr, stride_v: tl.constexpr):
    for i in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(NUM_KV):
            vi_j = tl.load(v_rows_ptr + i * stride_v + j)
            aj = tl.load(attn_ptr + j)
            acc += vi_j * aj
        tl.store(out_ptr + i, acc)


def _run_segment_triton(q_f32, k_cache_flat, v_cache_flat, qo_indptr, kv_indptr, kv_indices, sm_scale,
                        output, lse, b, num_qo_heads, gqa_ratio):
    # For segment b: b in [0, len_indptr-2]
    q_start = int(qo_indptr[b].item())
    q_end = int(qo_indptr[b + 1].item())
    num_q_tokens = q_end - q_start

    kv_start = int(kv_indptr[b].item())
    kv_end = int(kv_indptr[b + 1].item())
    num_segments_b = kv_end - kv_start  # number of cached groups in this segment

    if num_q_tokens <= 0 or num_segments_b <= 0:
        return

    # Gather kv_ids for this segment
    kv_ids = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [num_segments_b]

    # Iterate queries in segment
    for q_idx in range(num_q_tokens):
        global_q_idx = q_start + q_idx

        # Causal-like bound
        delta = num_segments_b - num_q_tokens
        max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
        if max_kv_idx <= 0:
            continue

        # For each query head h, map to kv head kv_head = h // gqa_ratio
        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio  # map 32 query heads to 8 kv heads (GQA)

            # Load q_vec [128] float32
            q_vec = q_f32[global_q_idx, h]  # [128] float32

            # Load K/V rows [max_kv_idx, 128]
            k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]
            v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128]

            # Triton: compute logits = q_vec @ k_rows.T
            logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=q_vec.device)
            _dot_logits_kernel[(1,)](
                q_vec, k_rows, logits,
                HEAD_DIM=128, NUM_KV=max_kv_idx
            )

            # Scale logits
            logits_scaled = logits * sm_scale  # [max_kv_idx]

            # Compute lse = logsumexp(logits_scaled) / ln(2) using torch (reduce over NUM_KV)
            lse_val = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)
            lse[global_q_idx, h] = lse_val

            # Compute attn = softmax(logits_scaled) using torch (reduce over NUM_KV)
            attn = torch.softmax(logits_scaled, dim=0)  # [max_kv_idx]

            # Triton matvec: out_vec = v_rows @ attn
            out_vec = torch.empty((128,), dtype=torch.float32, device=v_rows.device)
            _matvec_kernel[(1,)](
                v_rows, attn, out_vec,
                HEAD_DIM=128, NUM_KV=max_kv_idx, stride_v=128
            )

            # Store output (float32 for compute, cast outside if needed)
            output[global_q_idx, h] = out_vec


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gqa_ratio = 4  # 32 / 8

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes and device
        device = q.device
        total_q, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"

        num_pages, num_pages_dim2, num_kv_heads, _ = k_cache.shape
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert num_pages_dim2 == 1, "k_cache second dim must be 1 (squeezed later)"
        # squeeze singleton dimension
        k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

        # Output tensors (float32 for compute)
        output = torch.empty(
            (total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device
        )
        lse = torch.full(
            (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
        )

        # Cast q to float32 for compute
        q_f32 = q.to(torch.float32)

        len_indptr = qo_indptr.shape[0]
        # Iterate segments (b = 0..len_indptr-2)
        for b in range(len_indptr - 1):
            _run_segment_triton(
                q_f32, k_cache_flat, v_cache_flat, qo_indptr, kv_indptr, kv_indices, sm_scale,
                output, lse, b, num_qo_heads, self.gqa_ratio
            )

        # Return output as bfloat16 (original dtype), lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# Helper functions for evaluator
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16)
    k_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16)
    v_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 51, [34], dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

# Entry point expected by evaluator
class Model(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def run(*args):
    return ModelNew()(*args)
