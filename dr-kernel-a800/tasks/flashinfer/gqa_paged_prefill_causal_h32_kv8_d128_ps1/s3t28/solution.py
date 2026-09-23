import math
import torch
import triton
import triton.language as tl


# Triton kernel: logits = q_vec @ k_mat.T
# q_vec_ptr: [HEAD_DIM] float32
# k_mat_ptr: [NUM_KV, HEAD_DIM] float32, row-major (stride_k = HEAD_DIM)
# logits_out_ptr: [NUM_KV] float32
@triton.jit
def _dot_logits_kernel(q_vec_ptr, k_mat_ptr, logits_out_ptr,
                        HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr, stride_k: tl.constexpr):
    for i in range(NUM_KV):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            ki_j = tl.load(k_mat_ptr + i * stride_k + j)
            acc += qj * ki_j
        tl.store(logits_out_ptr + i, acc)


# Triton kernel: compute logsumexp(logit_scaled, axis=0) and divide by ln(2), return into out_lse[0]
# logit_scaled_ptr: [NUM_KV] float32 (padded to VEC_SIZE with -inf)
# out_lse_ptr: [1] float32
@triton.jit
def _lse_kernel(logit_scaled_ptr, out_lse_ptr, NUM_KV: tl.constexpr, VEC_SIZE: tl.constexpr):
    # Reduce max across VEC_SIZE; first compute max over the first NUM_KV entries
    max_val = tl.full((), -float("inf"), tl.float32)
    # Manually loop only NUM_KV (constexpr) to avoid dynamic loops
    for i in range(NUM_KV):
        vi = tl.load(logit_scaled_ptr + i)
        if i == 0:
            max_val = vi
        else:
            max_val = tl.maximum(max_val, vi)

    sum_exp = tl.zeros((), dtype=tl.float32)
    # Sum exp(logit - max) across NUM_KV
    for i in range(NUM_KV):
        vi = tl.load(logit_scaled_ptr + i)
        sum_exp += tl.exp(vi - max_val)

    lse = tl.log(sum_exp) + max_val  # logsumexp
    # Divide by ln(2)
    ln2 = 0.6931471805599453
    lse_div2 = lse / ln2
    tl.store(out_lse_ptr, lse_div2)


# Triton kernel: compute softmax(logit_scaled, axis=0) into attn_out_ptr[VEC_SIZE]
# logit_scaled_ptr: [NUM_KV] float32 (padded to VEC_SIZE with -inf)
# attn_out_ptr: [VEC_SIZE] float32
@triton.jit
def _softmax_kernel(logit_scaled_ptr, attn_out_ptr, NUM_KV: tl.constexpr, VEC_SIZE: tl.constexpr):
    # Compute max over NUM_KV
    max_val = tl.full((), -float("inf"), tl.float32)
    for i in range(NUM_KV):
        vi = tl.load(logit_scaled_ptr + i)
        if i == 0:
            max_val = vi
        else:
            max_val = tl.maximum(max_val, vi)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for i in range(NUM_KV):
        vi = tl.load(logit_scaled_ptr + i)
        sum_exp += tl.exp(vi - max_val)

    # Fill attn with softmax values; write only first NUM_KV positions
    for i in range(VEC_SIZE):
        # If i >= NUM_KV, value should be 0 (padding), but we won't write out for i >= NUM_KV
        if i < NUM_KV:
            vi = tl.load(logit_scaled_ptr + i)
            attn = tl.exp(vi - max_val) / sum_exp
            tl.store(attn_out_ptr + i, attn)
        else:
            # Leave as 0 (implicitly, since we only wrote first NUM_KV; caller discards the rest)
            pass


# Triton kernel: matvec: out[r] = sum_j v_mat[r, j] * attn[j], j over all columns (accumulate across NUM_KV rows)
# v_mat_ptr: [NUM_KV * HEAD_DIM] float32, viewed as [NUM_KV, HEAD_DIM] via stride v_stride = HEAD_DIM
# attn_ptr: [VEC_SIZE] float32 (only first NUM_KV used)
# out_vec_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_mat_ptr, attn_ptr, out_vec_ptr,
                   HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr, VEC_SIZE: tl.constexpr, v_stride: tl.constexpr):
    for r in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(NUM_KV):  # accumulate over all rows; attn[j] valid
            aj = tl.load(attn_ptr + j)
            vj = tl.load(v_mat_ptr + j * v_stride + r)
            acc += aj * vj
        tl.store(out_vec_ptr + r, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants
        self.HEAD_DIM = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA and contiguous, compute in float32
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
        device = q.device
        q = q.contiguous().to(torch.float32)
        k_cache = k_cache.contiguous().to(torch.float32)
        v_cache = v_cache.contiguous().to(torch.float32)
        qo_indptr = qo_indptr.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        total_q = q.shape[0]
        num_qo_heads = self.num_qo_heads
        head_dim = self.HEAD_DIM

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Flatten cached k/v to [num_pages, num_kv_heads, head_dim]
        k_cache_flat = k_cache.view(k_cache.shape[0], self.num_kv_heads, head_dim)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.view(v_cache.shape[0], self.num_kv_heads, head_dim)  # [num_pages, 8, 128]

        len_indptr = qo_indptr.shape[0]

        # We expect len_indptr >= 2; otherwise no segments. The original code asserts len_indptr > 1 in practice.
        # Iterate over segments b from 0 to len_indptr - 2
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            num_q_tokens = q_end - q_start

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_tokens = kv_end - kv_start

            # Handle empty segments
            if num_q_tokens <= 0 or num_kv_tokens <= 0:
                continue

            # Collect kv_ids for this segment and map to cached k/v
            kv_ids = kv_indices[kv_start:kv_end].contiguous()  # [num_kv_tokens]
            num_segments_b = kv_ids.shape[0]  # same as num_kv_tokens

            # Slice query batch
            q_batch = q[q_start:q_end]  # [num_q_tokens, num_qo_heads, head_dim]
            # Prepare output buffer for this segment
            # We will fill output row by row for each q_idx and head h

            # Precompute ln(2) for division
            ln2 = 0.6931471805599453

            # Loop over each query token and each head
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                # Compute causal-like limit
                delta = num_segments_b - num_q_tokens  # typically negative or small
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    # No valid kv for this query position
                    continue

                # Extract q_vec for all heads
                q_vecs = q_batch[q_idx]  # [32, 128]

                # Loop over query heads
                for h in range(num_qo_heads):
                    # corresponding KV head for GQA
                    kv_head = h // self.gqa_ratio  # 8 -> 0..31 mapped to 0..7

                    # Extract query vector for this head
                    q_vec = q_vecs[h]  # [128], float32

                    # Extract k/v rows for this kv_head across max_kv_idx entries
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128], float32
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128], float32

                    # Launch dot_logits_kernel: logits_scaled = q_vec @ k_rows.T -> [max_kv_idx]
                    NUM_KV = max_kv_idx  # compile-time const for this kernel launch
                    logits = torch.empty((NUM_KV,), dtype=torch.float32, device=device)
                    _dot_logits_kernel[(1,)](
                        q_vec, k_rows, logits,
                        HEAD_DIM=self.HEAD_DIM,
                        NUM_KV=NUM_KV,
                        stride_k=self.HEAD_DIM,
                    )

                    # Scale logits
                    logits_scaled = logits * sm_scale  # float32 scalar

                    # Compute lse = logsumexp(logits_scaled) / ln(2)
                    # Pad to VEC_SIZE = 128 with -inf
                    VEC_SIZE = self.HEAD_DIM
                    logit_scaled_pad = torch.empty((VEC_SIZE,), dtype=torch.float32, device=device)
                    logit_scaled_pad[:NUM_KV].copy_(logits_scaled)
                    logit_scaled_pad[NUM_KV:].fill_(-float("inf"))

                    out_lse = torch.empty((1,), dtype=torch.float32, device=device)
                    _lse_kernel[(1,)](
                        logit_scaled_pad, out_lse,
                        NUM_KV=NUM_KV,
                        VEC_SIZE=VEC_SIZE,
                    )
                    lse[global_q_idx, h] = out_lse[0]

                    # Compute softmax on logits_scaled
                    attn = torch.empty((VEC_SIZE,), dtype=torch.float32, device=device)
                    _softmax_kernel[(1,)](
                        logit_scaled_pad, attn,
                        NUM_KV=NUM_KV,
                        VEC_SIZE=VEC_SIZE,
                    )
                    # attn[:NUM_KV] are the softmax values; attn[NUM_KV:] are ignored

                    # Compute out_vec = attn @ v_rows -> [128]
                    # v_rows is [NUM_KV, 128], attn is [VEC_SIZE], only first NUM_KV matter
                    # We pass v_rows as a contiguous [NUM_KV*128] buffer and use stride 128
                    v_mat = v_rows.contiguous().view(-1)  # [NUM_KV*128]
                    out_vec = torch.empty((self.HEAD_DIM,), dtype=torch.float32, device=device)
                    _matvec_kernel[(1,)](
                        v_mat, attn, out_vec,
                        HEAD_DIM=self.HEAD_DIM,
                        NUM_KV=NUM_KV,
                        VEC_SIZE=VEC_SIZE,
                        v_stride=self.HEAD_DIM,
                    )

                    # Store output
                    output[global_q_idx, h] = out_vec  # float32

        # Cast output to bfloat16 to match original return dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# Helper functions from the original (for testing)
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16, device="cuda")
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device="cuda")
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device="cuda")
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 51, [34], dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

# Example: ModelNew entry point
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
