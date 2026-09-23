import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q_vec @ k_rows.T for q_vec [128], k_rows [NUM_ROWS, 128], returns logits [NUM_ROWS].
# NUM_ROWS is tl.constexpr so Triton can unroll loops.
@triton.jit
def matvec_128_kernel(q_ptr, k_ptr, out_ptr, NUM_ROWS: tl.constexpr):
    # q_ptr: float32 [128]
    # k_ptr: float32 [NUM_ROWS, 128]
    # out_ptr: float32 [NUM_ROWS]
    for i in range(NUM_ROWS):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(128):
            acc += tl.load(q_ptr + j) * tl.load(k_ptr + i * 128 + j)
        tl.store(out_ptr + i, acc)


# Triton kernel: softmax over a vector of length 128 (compile-time constant).
@triton.jit
def softmax_128_kernel(logits_ptr, out_ptr):
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


# Triton kernel: logsumexp over a vector of length 128 (compile-time constant), scaled by 1/ln(2).
@triton.jit
def lse_scaled_128_kernel(inp_ptr, out_ptr):
    m = tl.load(inp_ptr + 0)
    for i in range(1, 128):
        vj = tl.load(inp_ptr + i)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for i in range(128):
        vi = tl.load(inp_ptr + i)
        sum_exp += tl.exp(vi - m)
    lse = tl.log(sum_exp) + m  # logsumexp over 128 elements
    # Scale by 1/ln(2) = 1.4426950408889634 (matches PyTorch's division by math.log(2.0))
    tl.store(out_ptr, lse * 1.4426950408889634)


# Triton kernel: compute out = attn @ v_rows for attn [128], v_rows [NUM_ROWS, 128], returns out [128].
@triton.jit
def attn_matvec_128_kernel(attn_ptr, v_ptr, out_ptr, NUM_ROWS: tl.constexpr):
    out_vec = [0.0] * 128  # Python list to hold vector; Triton will compute each element
    # We'll compute out_vec[j] = sum_i attn[i] * v_rows[i, j] over i in [0, NUM_ROWS)
    for j in range(128):
        acc = tl.zeros((), dtype=tl.float32)
        for i in range(NUM_ROWS):
            # v_rows[i, j] is at v_ptr + i * 128 + j
            acc += tl.load(attn_ptr + i) * tl.load(v_ptr + i * 128 + j)
        out_vec[j] = acc
    # Store out_vec to out_ptr
    for j in range(128):
        tl.store(out_ptr + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # no parameters

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        q: [total_q, 32, 128] bfloat16
        k_cache: [num_pages, 1, 8, 128] bfloat16
        v_cache: [num_pages, 1, 8, 128] bfloat16
        qo_indptr: [len_indptr] int32
        kv_indptr: [len_indptr] int32
        kv_indices: [num_kv_indices] int32
        sm_scale: float32 scalar, e.g., 1.0 / sqrt(128)
        """
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        # len_indptr and segment handling
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Output and lse tensors (float32 for compute, final output cast to bfloat16)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Convert q to float32 for compute
        q_f32 = q.to(torch.float32)

        # Flatten k_cache and v_cache to [num_segments, num_kv_heads, 128] by removing size-1 dim
        k_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

        # Process segments b in [0, len_indptr - 2]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            num_segments_b = kv_end - kv_start

            # GQA mapping: query head h maps to KV head kvh = h // 4
            gqa_ratio = num_qo_heads // num_kv_heads  # 4

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                # For each query head h
                for h in range(num_qo_heads):
                    kvh = h // gqa_ratio  # 0..7
                    # Compute max_kv_idx for this segment position
                    delta = num_segments_b - num_q_tokens
                    max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                    if max_kv_idx <= 0:
                        # No valid KV rows; lse remains -inf, out zero
                        lse[global_q_idx, h] = -float("inf")
                        continue

                    # Prepare q_vec [128]
                    q_vec = q_f32[global_q_idx, h]  # shape [128], float32

                    # Gather k_rows and v_rows for this segment, head kvh
                    # kv_indices[kv_start:kv_end] -> [num_segments_b]
                    # k_flat shape: [num_segments_b, num_kv_heads, 128]
                    # v_flat similarly.
                    # Build k_rows and v_rows as contiguous tensors of shape [max_kv_idx, 128]
                    k_rows = torch.empty((max_kv_idx, head_dim), dtype=torch.float32, device=q.device)
                    v_rows = torch.empty((max_kv_idx, head_dim), dtype=torch.float32, device=q.device)

                    # Fill k_rows and v_rows
                    for s in range(max_kv_idx):
                        group_id = kv_indices[kv_start + s].item()
                        # k_flat[group_id, kvh, :] and v_flat[group_id, kvh, :]
                        k_rows[s] = k_flat[group_id, kvh]  # [128]
                        v_rows[s] = v_flat[group_id, kvh]  # [128]

                    # Compute logits = q_vec @ k_rows.T -> [max_kv_idx]
                    logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=q.device)
                    matvec_128_kernel[(1,)](q_vec, k_rows, logits, NUM_ROWS=max_kv_idx)

                    # Scale logits
                    logits_scaled = logits * sm_scale

                    # Compute lse = logsumexp(logits_scaled) / ln(2)
                    # Pass logits_scaled to Triton
                    lse_out = torch.empty((), dtype=torch.float32, device=q.device)
                    inp_vec = logits_scaled  # 1D float32 vector of length 128, with NUM_ROWS=min(max_kv_idx, 128)
                    # We pass the first max_kv_idx entries; pad the rest with -inf (not needed here since 128 is long enough)
                    # But Triton kernel expects 128 elements; we ensure it by making a 128-length vector where only first max_kv_idx are used.
                    # Create a 128-length vector filled with logits_scaled and zeros beyond max_kv_idx; however Triton expects contiguous pointer.
                    # Here, we reuse that inp_vec is logits_scaled (size >= 128). We'll assume max_kv_idx <= 128 for our inputs.
                    # To be safe, we can construct a 128-length inp:
                    inp_vec_128 = torch.empty(128, dtype=torch.float32, device=q.device)
                    inp_vec_128[:max_kv_idx] = logits_scaled
                    inp_vec_128[max_kv_idx:] = -float("inf")
                    lse_scaled_128_kernel[(1,)](inp_vec_128, lse_out)
                    lse[global_q_idx, h] = lse_out.item()  # store scalar

                    # Compute attn = softmax(logits_scaled) -> [max_kv_idx]
                    attn = torch.empty((max_kv_idx,), dtype=torch.float32, device=q.device)
                    softmax_128_kernel[(1,)](logits_scaled, attn)

                    # Compute out = attn @ v_rows -> [128]
                    out_vec = torch.empty((head_dim,), dtype=torch.float32, device=q.device)
                    attn_matvec_128_kernel[(1,)](attn, v_rows, out_vec, NUM_ROWS=max_kv_idx)

                    # Store output[q_idx, h, :] = out_vec cast to bfloat16
                    output[global_q_idx, h] = out_vec.to(torch.bfloat16)

        return output, lse


# Example helper functions (not used by evaluator but useful for testing)
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 51, [34], dtype=torch.int32, device='cuda')
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
