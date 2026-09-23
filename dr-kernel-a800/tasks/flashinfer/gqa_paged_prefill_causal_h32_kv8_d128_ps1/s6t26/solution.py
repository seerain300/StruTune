import math
import torch

# Provide get_inputs that returns exactly 7 tensors to match evaluator expectations.
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


# Triton kernel: process multiple q indices for one batch b and head h.
# Inputs:
#   q_vecs_ptr: *fp32, [num_q_tokens, head_dim]
#   k_seg_ptr: *fp32, [num_kv_tokens, head_dim]
#   v_seg_ptr: *fp32, [num_kv_tokens, head_dim]
#   out_mat_ptr: *fp32, [num_q_tokens, head_dim]
#   lse_vec_ptr: *fp32, [num_q_tokens]
#   qo_start: int32, starting q index in this batch
#   qo_len: int32, number of q tokens in this batch
#   num_kv_tokens: int32, segment length
#   sm_scale: float32
#   head_dim: tl.constexpr
#   CHUNK: tl.constexpr
@triton.jit
def batch_head_kernel(
    q_vecs_ptr,        # *fp32, [num_q_tokens, head_dim]
    k_seg_ptr,         # *fp32, [num_kv_tokens, head_dim]
    v_seg_ptr,         # *fp32, [num_kv_tokens, head_dim]
    out_mat_ptr,       # *fp32, [num_q_tokens, head_dim]
    lse_vec_ptr,       # *fp32, [num_q_tokens]
    qo_start: tl.int32,
    qo_len: tl.int32,
    num_kv_tokens: tl.int32,
    sm_scale: tl.float32,
    head_dim: tl.constexpr,
    CHUNK: tl.constexpr
):
    # Compute lse and output for each q index in this batch
    for q_i in range(0, qo_len):
        global_q_idx = qo_start + q_i
        q_vec = tl.load(q_vecs_ptr + q_i * head_dim + tl.arange(0, head_dim))  # [head_dim]

        # Compute m and l across the segment
        m = -float("inf")
        for start in range(0, num_kv_tokens, CHUNK):
            offs = start + tl.arange(0, CHUNK)
            mask = offs < num_kv_tokens
            q_vec = tl.load(q_vecs_ptr + q_i * head_dim + tl.arange(0, head_dim), mask=mask, other=0.0)
            logits = tl.zeros([CHUNK], dtype=tl.float32)
            for i in range(0, CHUNK):
                idx_i = offs[i]
                valid_i = idx_i < num_kv_tokens
                k_row_ptr = k_seg_ptr + idx_i * head_dim
                k_row = tl.load(k_row_ptr + tl.arange(0, head_dim), mask=valid_i, other=0.0)
                dot = 0.0
                for j in range(0, head_dim):
                    dot += q_vec[j] * k_row[j]
                logits[i] = dot * sm_scale
            m = tl.maximum(m, tl.max(tl.where(mask, logits, -float("inf")), axis=0))
        # Compute l = sum(exp(logits - m))
        l = 0.0
        for start in range(0, num_kv_tokens, CHUNK):
            offs = start + tl.arange(0, CHUNK)
            mask = offs < num_kv_tokens
            q_vec = tl.load(q_vecs_ptr + q_i * head_dim + tl.arange(0, head_dim), mask=mask, other=0.0)
            logits = tl.zeros([CHUNK], dtype=tl.float32)
            for i in range(0, CHUNK):
                idx_i = offs[i]
                valid_i = idx_i < num_kv_tokens
                k_row_ptr = k_seg_ptr + idx_i * head_dim
                k_row = tl.load(k_row_ptr + tl.arange(0, head_dim), mask=valid_i, other=0.0)
                dot = 0.0
                for j in range(0, head_dim):
                    dot += q_vec[j] * k_row[j]
                logits[i] = dot * sm_scale
            l += tl.sum(tl.exp(tl.where(mask, logits - m, -float("inf"))), axis=0)
        ln2 = 1.0 / math.log(2.0)
        lse_val = tl.log(l) / ln2
        tl.store(lse_vec_ptr + q_i, lse_val)

        # Compute output vector for this q index
        out_vec = tl.zeros([head_dim], dtype=tl.float32)
        for start in range(0, num_kv_tokens, CHUNK):
            offs = start + tl.arange(0, CHUNK)
            mask = offs < num_kv_tokens
            q_vec = tl.load(q_vecs_ptr + q_i * head_dim + tl.arange(0, head_dim), mask=mask, other=0.0)
            logits = tl.zeros([CHUNK], dtype=tl.float32)
            for i in range(0, CHUNK):
                idx_i = offs[i]
                valid_i = idx_i < num_kv_tokens
                k_row_ptr = k_seg_ptr + idx_i * head_dim
                v_row_ptr = v_seg_ptr + idx_i * head_dim
                k_row = tl.load(k_row_ptr + tl.arange(0, head_dim), mask=valid_i, other=0.0)
                dot = 0.0
                for j in range(0, head_dim):
                    dot += q_vec[j] * k_row[j]
                logits[i] = dot * sm_scale
            attn = tl.exp(logits - lse_val)
            for i in range(0, CHUNK):
                idx_i = start + i
                valid_i = idx_i < num_kv_tokens
                v_row_ptr = v_seg_ptr + idx_i * head_dim
                v_row = tl.load(v_row_ptr + tl.arange(0, head_dim), mask=valid_i, other=0.0)
                for j in range(0, head_dim):
                    out_vec[j] += attn[i] * v_row[j]
        # Store output vector for this q index
        for j in range(0, head_dim):
            tl.store(out_mat_ptr + q_i * head_dim + j, out_vec[j])


# ModelNew: perform attention using Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        device = q.device
        q = q.to(device).contiguous()
        k_cache = k_cache.to(device).contiguous()
        v_cache = v_cache.to(device).contiguous()

        # Squeeze dim=1
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim]

        num_qo_heads = q.shape[1]
        num_kv_heads = k_cache.shape[2]
        total_q = q.shape[0]
        assert total_q == int(qo_indptr[-1].item()), "Sum of qo_indptr must equal total_q."

        # Output and lse buffers
        head_dim = q.shape[2]
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Process each batch segment
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            if num_q_tokens <= 0 or num_kv_tokens <= 0:
                continue

            kv_indices_b = kv_indices[kv_start:kv_end].to(device)  # [num_kv_tokens]

            # Loop over heads
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # 0..7

                # q vectors for all q indices in this batch
                q_vecs = q[q_start:q_end, h, :].to(torch.float32).contiguous()  # [num_q_tokens, head_dim]
                # Gather k and v rows for this batch segment
                k_rows = k_cache_flat[kv_indices_b, kv_head, :].to(torch.float32).contiguous()  # [num_kv_tokens, head_dim]
                v_rows = v_cache_flat[kv_indices_b, kv_head, :].to(torch.float32).contiguous()  # [num_kv_tokens, head_dim]

                # Allocate output matrix and lse vector for these q indices
                out_mat = torch.empty((num_q_tokens, head_dim), dtype=torch.float32, device=device)
                lse_vec = torch.empty((num_q_tokens,), dtype=torch.float32, device=device)

                # Launch Triton kernel for this batch and head
                CHUNK = 128  # specialize for head_dim=128; mask handles smaller
                batch_head_kernel[(1,)](
                    q_vecs,                 # *fp32 [num_q_tokens, head_dim]
                    k_rows,                 # *fp32 [num_kv_tokens, head_dim]
                    v_rows,                 # *fp32 [num_kv_tokens, head_dim]
                    out_mat,                # *fp32 [num_q_tokens, head_dim]
                    lse_vec,                # *fp32 [num_q_tokens]
                    qo_start=q_start,
                    qo_len=num_q_tokens,
                    num_kv_tokens=num_kv_tokens,
                    sm_scale=float(sm_scale),
                    head_dim=head_dim,
                    CHUNK=CHUNK
                )

                # Store results into output and lse
                # out_mat shape: [num_q_tokens, head_dim]
                for q_i in range(0, num_q_tokens):
                    output[q_start + q_i, h, :] = out_mat[q_i, :].to(torch.bfloat16)
                    lse[q_start + q_i, h] = lse_vec[q_i]

        return output, lse


def run(*args):
    return ModelNew()(*args)
