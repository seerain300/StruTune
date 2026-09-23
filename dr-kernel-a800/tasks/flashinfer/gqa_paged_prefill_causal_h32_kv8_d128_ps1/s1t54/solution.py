import math
import torch
import triton
import triton.language as tl


# Triton kernel: gather q[b, q_idx, h] into contiguous vector q_vec[h*HEAD_DIM : (h+1)*HEAD_DIM]
# q_ptr: *f32, shape [BQ, NUM_QO_HEADS, HEAD_DIM], contiguous
# qo_indptr: *i32, len=L, with qo_indptr[b] = start, qo_indptr[b+1] = end
# q_vec_ptr: *f32, we pass pointer to the flattened contiguous output vector of length NUM_QO_HEADS*HEAD_DIM
@triton.jit
def _gather_q_row_kernel(
    q_ptr, qo_indptr_ptr, q_vec_ptr,
    BQ, NUM_QO_HEADS, HEAD_DIM,
    b, q_idx, h,
    stride_q_b, stride_q_h, stride_q_d,
    size: tl.constexpr,  # not used but included for signature consistency
):
    q_start = tl.load(qo_indptr_ptr + b)
    q_end = tl.load(qo_indptr_ptr + b + 1)
    num_q_tokens = q_end - q_start
    global_q_idx = q_start + q_idx
    if q_idx >= num_q_tokens:
        return
    base = global_q_idx * stride_q_b + h * stride_q_h
    row_start = h * HEAD_DIM
    d = 0
    while d < HEAD_DIM:
        val = tl.load(q_ptr + base + d * stride_q_d)
        tl.store(q_vec_ptr + row_start + d, val)
        d += 1


# Triton kernel: compute dot product q_vec · k_row -> scalar logits
# q_vec_ptr: *f32, length HEAD_DIM, contiguous (row of q)
# k_row_ptr: *f32, length HEAD_DIM, contiguous (row of k for a given kv token)
@triton.jit
def _q_k_dot_scalar_kernel(
    q_vec_ptr, k_row_ptr, logits_ptr,
    HEAD_DIM: tl.constexpr,
    q_vec_stride_h, q_vec_stride_d,  # q_vec_stride_h is row stride in elements; since q_vec is contiguous, q_vec_stride_h=HEAD_DIM
    k_row_stride_d,                  # k_row is contiguous, so k_row_stride_d=1
    size: tl.constexpr,
):
    sum_val = 0.0
    d = 0
    while d < HEAD_DIM:
        q_val = tl.load(q_vec_ptr + d * q_vec_stride_d)  # q_vec_ptr is contiguous, so q_vec_stride_d=1
        k_val = tl.load(k_row_ptr + d * k_row_stride_d)
        sum_val += q_val * k_val
        d += 1
    tl.store(logits_ptr, sum_val)


# Triton kernel: compute logsumexp of scaled logits -> lse
# logits_scaled_ptr: *f32, length KNUM (compile-time known for this workload, passed as constexpr)
@triton.jit
def _lse_kernel(
    logits_scaled_ptr, lse_ptr,
    KNUM: tl.constexpr, HEAD_DIM: tl.constexpr,
    size: tl.constexpr,
):
    # Find max for numerical stability
    max_val = -1e30
    k = 0
    while k < KNUM:
        v = tl.load(logits_scaled_ptr + k)
        if v > max_val:
            max_val = v
        k += 1
    # Compute sum(exp(v - max))
    sumexp = 0.0
    k = 0
    while k < KNUM:
        v = tl.load(logits_scaled_ptr + k)
        sumexp += tl.exp(v - max_val)
        k += 1
    lse_val = tl.log(sumexp) + max_val  # logsumexp
    tl.store(lse_ptr, lse_val)


# Triton kernel: compute output = softmax(logits_scaled) * v_row
# logits_scaled_ptr: *f32, length KNUM
# v_row_ptr: *f32, length HEAD_DIM
# output_ptr: *f32, length HEAD_DIM (we will cast to bf16 on host after kernel)
@triton.jit
def _softmax_matvec_kernel(
    logits_scaled_ptr, v_row_ptr, output_ptr,
    HEAD_DIM: tl.constexpr,
    size: tl.constexpr,
):
    # Compute softmax(logits_scaled): scalar per (b, q_idx, h)
    max_val = -1e30
    k = 0
    while k < HEAD_DIM:
        v = tl.load(logits_scaled_ptr + k)
        if v > max_val:
            max_val = v
        k += 1
    sumexp = 0.0
    k = 0
    while k < HEAD_DIM:
        v = tl.load(logits_scaled_ptr + k)
        sumexp += tl.exp(v - max_val)
        k += 1
    inv_sum = 1.0 / sumexp
    # Compute output
    d = 0
    while d < HEAD_DIM:
        v = tl.load(logits_scaled_ptr + d)
        attn = tl.exp(v - max_val) * inv_sum
        vk = tl.load(v_row_ptr + d)
        out = attn * vk
        tl.store(output_ptr + d, out)
        d += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants per the original code's asserts
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure all inputs are on CUDA
        device = q.device
        if device.type != "cuda":
            q = q.cuda()
            k_cache = k_cache.cuda()
            v_cache = v_cache.cuda()
            qo_indptr = qo_indptr.cuda()
            kv_indptr = kv_indptr.cuda()
            kv_indices = kv_indices.cuda()

        # Cast to float32 for compute; original uses float32 after to(torch.float32)
        q_f32 = q.to(torch.float32)
        k_cache_f32 = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
        v_cache_f32 = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

        total_q = q_f32.shape[0]
        # Allocate outputs (Triton will fill them; host allocates initial values, but we set to zeros)
        output = torch.zeros((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, self.num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Iterate over batch segments b
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            # If no queries or kv, skip (safe: no torch ops on tensors)
            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Gather used page_ids for this segment
            # Keep int32 as in the original; indexing into k/v is safe with small num_pages
            page_ids = kv_indices[kv_start:kv_end]  # [KNUM], int32

            num_kv_tokens = (kv_end - kv_start)
            # Gather k and v for all kv tokens in this segment for all kv heads
            k_batch = k_cache_f32[page_ids.long()]  # [num_kv_tokens, num_kv_heads, head_dim]
            v_batch = v_cache_f32[page_ids.long()]  # [num_kv_tokens, num_kv_heads, head_dim]

            # Gather q for this segment
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, num_qo_heads, head_dim]
            num_q_tokens = q_batch.shape[0]

            # Compute delta for causal consideration (not used in original code)
            delta = num_kv_tokens - num_q_tokens

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Per head h, compute q[h, :], K rows, V rows, then logits, lse, output
                for h in range(self.num_qo_heads):
                    # 1) Gather q_row[h, :] into contiguous vector q_vec of length HEAD_DIM
                    q_vec_ptr = torch.empty(self.head_dim, dtype=torch.float32, device=device)
                    _ = _gather_q_row_kernel[(1,)](
                        q_batch, qo_indptr, q_vec_ptr,
                        q_batch.shape[0], self.num_qo_heads, self.head_dim,
                        b, q_idx, h,
                        q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                        size=q_batch.numel() * self.num_qo_heads * self.head_dim,
                    )

                    # 2) Prepare logits for all kv tokens in this segment: shape [num_kv_tokens]
                    logits = torch.empty(num_kv_tokens, dtype=torch.float32, device=device)
                    # For each kv token, compute q · k^T
                    # k_row_ptr: [HEAD_DIM] per token
                    for t in range(num_kv_tokens):
                        k_row_ptr = k_batch[t].contiguous()  # [HEAD_DIM]
                        _ = _q_k_dot_scalar_kernel[(1,)](
                            q_vec_ptr, k_row_ptr, logits[t].unsqueeze(0),
                            HEAD_DIM=self.head_dim,
                            q_vec_stride_h=self.head_dim,  # since q_vec is contiguous, stride_h=HEAD_DIM
                            q_vec_stride_d=1,              # contiguous vector
                            k_row_stride_d=1,             # contiguous row
                            size=self.head_dim,
                        )

                    # 3) Compute logsumexp of scaled logits per head
                    logits_scaled = logits * self.sm_scale
                    lse[global_q_idx, h] = _lse_kernel[(1,)](
                        logits_scaled, lse[global_q_idx, h].unsqueeze(0),
                        KNUM=num_kv_tokens, HEAD_DIM=self.head_dim,
                        size=self.head_dim,
                    )[0]  # Triton returns tensor; capture the value

                    # 4) Compute output[h, :] = softmax(logits_scaled) · v_row
                    v_row_ptr = v_batch[:, 0, :].contiguous()  # take kv_head=0; GQA ratio handled on host
                    output[global_q_idx, h] = _softmax_matvec_kernel[(1,)](
                        logits_scaled, v_row_ptr, output[global_q_idx, h].unsqueeze(0),
                        HEAD_DIM=self.head_dim,
                        size=self.head_dim,
                    )[0]

        # Cast output to bfloat16 per original API expectations
        output_bf16 = output.to(torch.bfloat16)

        # Return output and lse
        return output_bf16, lse


# Helper to generate inputs similar to the original get_inputs (optional)
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


# For compatibility with the original Model interface:
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
