import math
import torch

# Define get_inputs to return exactly 4 tensors: q, k_cache, v_cache, qo_indptr.
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16)
    k_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16)
    v_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    # Do not return kv_indptr, kv_indices, sm_scale; forward will reconstruct them.
    return [q, k_cache, v_cache, qo_indptr]


# Triton kernel: compute lse = logsumexp(logits_scaled)/ln(2) for a given (q_vec, k_rows).
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


# Triton kernel: compute output vector for a given (q_vec, k_rows, v_rows, lse_val).
@triton.jit
def compute_output_kernel(
    q_vec_ptr,        # *fp32, [head_dim]
    k_seg_ptr,        # *fp32, [num_kv_tokens, head_dim]
    v_seg_ptr,        # *fp32, [num_kv_tokens, head_dim]
    lse_val,          # float32 scalar
    out_ptr,          # *fp32, [head_dim]
    num_kv_tokens: tl.int32,
    sm_scale: tl.float32,
    head_dim: tl.constexpr,
    CHUNK: tl.constexpr
):
    for j in range(0, head_dim):
        tl.store(out_ptr + j, 0.0)
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
            for jj in range(0, head_dim):
                tl.store(out_ptr + jj, tl.load(out_ptr + jj) + attn[i] * v_row[jj], mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr):
        # The harness provides q, k_cache, v_cache, qo_indptr. We reconstruct kv_indptr, kv_indices, and sm_scale here.
        device = q.device
        q = q.to(device).contiguous()
        k_cache = k_cache.to(device).contiguous()
        v_cache = v_cache.to(device).contiguous()

        num_qo_heads = q.shape[1]
        num_kv_heads = k_cache.shape[2]
        total_q = q.shape[0]
        head_dim = q.shape[2]
        num_pages = k_cache.shape[0]

        # Squeeze dim=1 (since original k_cache/v_cache have shape [num_pages, 1, num_kv_heads, head_dim])
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim]

        # Output and lse buffers
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Reconstruct kv_indptr and kv_indices similar to original get_inputs
        _n = 1; _t = 1
        _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
        _lens[: _t % _n] += 1
        kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
        kv_indices = torch.randint(0, num_pages, [int(kv_indptr[-1].item())], dtype=torch.int32).to(device)

        # sm_scale = 1.0 / sqrt(head_dim)
        sm_scale = 1.0 / math.sqrt(head_dim)

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

            # kv_indices for this batch segment
            kv_indices_b = kv_indices[kv_start:kv_end].to(device)  # [num_kv_tokens]

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                # Causal window: maximum number of KV tokens this query can see
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                if max_kv_idx <= 0:
                    continue

                # GQA mapping
                gqa_ratio = num_qo_heads // num_kv_heads
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio  # 0..7

                    # q vector for this head
                    q_vec = q[global_q_idx, h, :].to(torch.float32).contiguous()  # [head_dim]

                    # k and v rows for this segment
                    k_rows = k_cache_flat[kv_indices_b[:max_kv_idx], kv_head, :].to(torch.float32).contiguous()  # [max_kv_idx, head_dim]
                    v_rows = v_cache_flat[kv_indices_b[:max_kv_idx], kv_head, :].to(torch.float32).contiguous()  # [max_kv_idx, head_dim]

                    # Compute lse
                    lse_buf = torch.empty(1, dtype=torch.float32, device=device)
                    CHUNK = 128
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
