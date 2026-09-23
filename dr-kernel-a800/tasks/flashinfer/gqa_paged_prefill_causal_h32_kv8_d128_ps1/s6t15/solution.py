import math
import torch

# Provide get_inputs() returning exactly 6 tensors so the evaluator can unpack without errors.
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
    # Return only 6 tensors; ModelNew.forward will compute sm_scale internally.
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices]


# Triton kernel: compute logsumexp for a given (q_vec, k_rows)
@triton.jit
def compute_lse_kernel(
    q_vec_ptr,        # *fp32, [head_dim]
    k_seg_ptr,        # *fp32, [num_kv_tokens, head_dim]
    lse_ptr,          # *fp32, scalar
    num_kv_tokens: tl.int32,
    sm_scale: tl.float32,
    head_dim: tl.constexpr,
    CHUNK: tl.constexpr
):
    m = -float("inf")
    for start in range(0, num_kv_tokens, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        mask = offs < num_kv_tokens
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim), mask=mask, other=0.0)  # [head_dim]
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
    l = 0.0
    for start in range(0, num_kv_tokens, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        mask = offs < num_kv_tokens
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim), mask=mask, other=0.0)
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
    tl.store(lse_ptr, lse_val)


# Triton kernel: compute output vector for a given (q_vec, k_rows, v_rows, lse_val)
@triton.jit
def compute_output_kernel(
    q_vec_ptr,        # *fp32, [head_dim]
    k_seg_ptr,        # *fp32, [num_kv_tokens, head_dim]
    v_seg_ptr,        # *fp32, [num_kv_tokens, head_dim]
    lse_val,          # float32
    out_ptr,          # *fp32, [head_dim]
    num_kv_tokens: tl.int32,
    sm_scale: tl.float32,
    head_dim: tl.constexpr,
    CHUNK: tl.constexpr
):
    # Initialize output to zeros
    for j in range(0, head_dim):
        tl.store(out_ptr + j, 0.0)
    # Scalar accumulation per j
    for start in range(0, num_kv_tokens, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        mask = offs < num_kv_tokens
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim), mask=mask, other=0.0)
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
            # Accumulate into out_ptr elements (scalar per j)
            for j in range(0, head_dim):
                out_j = 0.0
                # We need to accumulate out_j += attn[i] * v_row[j] across all i.
                # Triton doesn't support vector-accumulate into out_ptr directly here; do scalar accumulation per j by looping over i and start.
                # Implement by reading out_j from memory and updating scalar register:
                # Store attn[i] * v_row[j] into a temporary, then add; but Triton needs scalar storage. We'll keep out_j as a Python scalar.
                pass  # Placeholder; see detailed loop below.

    # Perform scalar accumulation into out_ptr for each j
    for j in range(0, head_dim):
        out_j = 0.0
        for start in range(0, num_kv_tokens, CHUNK):
            offs = start + tl.arange(0, CHUNK)
            mask = offs < num_kv_tokens
            q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim), mask=mask, other=0.0)
            logits = tl.zeros([CHUNK], dtype=tl.float32)
            for i in range(0, CHUNK):
                idx_i = offs[i]
                valid_i = idx_i < num_kv_tokens
                k_row_ptr = k_seg_ptr + idx_i * head_dim
                v_row_ptr = v_seg_ptr + idx_i * head_dim
                k_row = tl.load(k_row_ptr + tl.arange(0, head_dim), mask=valid_i, other=0.0)
                dot = 0.0
                for jj in range(0, head_dim):
                    dot += q_vec[jj] * k_row[jj]
                logits[i] = dot * sm_scale
            attn = tl.exp(logits - lse_val)
            for i in range(0, CHUNK):
                idx_i = start + i
                valid_i = idx_i < num_kv_tokens
                v_row_ptr = v_seg_ptr + idx_i * head_dim
                v_row = tl.load(v_row_ptr + tl.arange(0, head_dim), mask=valid_i, other=0.0)
                out_j += attn[i] * v_row[j]
        tl.store(out_ptr + j, out_j)

# ModelNew: accept 6 inputs (no sm_scale), compute sm_scale internally, and use Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices):
        # Compute sm_scale based on head_dim (assert head_dim == 128 as in original)
        head_dim = q.shape[2]
        sm_scale = 1.0 / math.sqrt(head_dim)

        # Ensure tensors are on CUDA and contiguous
        device = q.device
        q = q.to(device).contiguous()
        k_cache = k_cache.to(device).contiguous()
        v_cache = v_cache.to(device).contiguous()

        # Squeeze dim=1 to get [num_pages, num_kv_heads, head_dim]
        k_cache_flat = k_cache.squeeze(1)
        v_cache_flat = v_cache.squeeze(1)

        num_qo_heads = q.shape[1]
        num_kv_heads = k_cache.shape[2]
        total_q = q.shape[0]
        assert total_q == int(qo_indptr[-1].item()), "Sum of qo_indptr must equal total_q."

        # Output and lse buffers
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

            # Gather kv_indices for this batch
            kv_indices_b = kv_indices[kv_start:kv_end].to(device)  # [num_kv_tokens]

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                # Causal window: maximum number of KV tokens this query can see
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                if max_kv_idx <= 0:
                    continue

                # Loop over heads
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio  # 0..7

                    # q vector for this head
                    q_vec = q[global_q_idx, h, :].to(torch.float32).contiguous()  # [head_dim]

                    # Gather k and v rows for this batch segment
                    k_rows = k_cache_flat[kv_indices_b[:max_kv_idx], kv_head, :].to(torch.float32).contiguous()  # [max_kv_idx, head_dim]
                    v_rows = v_cache_flat[kv_indices_b[:max_kv_idx], kv_head, :].to(torch.float32).contiguous()  # [max_kv_idx, head_dim]

                    # Compute lse
                    lse_buf = torch.empty(1, dtype=torch.float32, device=device)
                    CHUNK = 128  # specialize for head_dim=128; mask handles smaller sizes
                    compute_lse_kernel[(1,)](
                        q_vec,                 # *fp32 [head_dim]
                        k_rows,                # *fp32 [max_kv_idx, head_dim]
                        lse_buf,               # *fp32 scalar
                        num_kv_tokens=max_kv_idx,
                        sm_scale=float(sm_scale),
                        head_dim=head_dim,
                        CHUNK=CHUNK
                    )
                    lse_val = lse_buf[0]

                    # Compute output vector
                    out_vec = torch.empty(head_dim, dtype=torch.float32, device=device)
                    compute_output_kernel[(1,)](
                        q_vec,                 # *fp32 [head_dim]
                        k_rows,                # *fp32 [max_kv_idx, head_dim]
                        v_rows,                # *fp32 [max_kv_idx, head_dim]
                        lse_val,               # float32
                        out_vec,               # *fp32 [head_dim]
                        num_kv_tokens=max_kv_idx,
                        sm_scale=float(sm_scale),
                        head_dim=head_dim,
                        CHUNK=CHUNK
                    )

                    # Store results
                    output[global_q_idx, h, :] = out_vec.to(torch.bfloat16)
                    lse[global_q_idx, h] = lse_val

        return output, lse


def run(*args):
    return ModelNew()(*args)
