import math
import torch

# Provide the original get_inputs function (exact as in the reference).
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


# Triton kernel: compute lse for one (b, q_idx, h)
# Inputs:
#   q_vec_ptr: *fp32, [head_dim]
#   k_seg_ptr: *fp32, [num_kv_tokens, head_dim]
#   num_kv_tokens: int32
#   sm_scale: float32
# Outputs:
#   lse_ptr: *fp32, scalar
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
    # Initialize m and l
    m = -float("inf")
    l = 0.0

    # Iterate over segment in tiles
    for start in range(0, num_kv_tokens, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        mask = offs < num_kv_tokens
        # Load q vector [head_dim]
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))
        # Load k tile [CHUNK, head_dim]
        k_tile = tl.load(k_seg_ptr + offs[:, None] * head_dim + tl.arange(0, head_dim), mask=mask[:, None], other=0.0)
        # Compute logits for this tile
        logits = tl.zeros([CHUNK], dtype=tl.float32)
        for j in range(0, head_dim):
            logits += q_vec[j] * k_tile[:, j]
        logits = logits * sm_scale
        # Update m and l
        m = tl.maximum(m, tl.max(tl.where(mask, logits, -float("inf")), axis=0))
        l += tl.sum(tl.where(mask, tl.exp(logits - m), 0.0), axis=0)

    ln2 = 1.0 / math.log(2.0)
    lse_val = tl.log(l) / ln2
    tl.store(lse_ptr, lse_val)


# Triton kernel: compute output vector for one (b, q_idx, h) using precomputed lse
# Inputs:
#   q_vec_ptr: *fp32, [head_dim]
#   k_seg_ptr: *fp32, [num_kv_tokens, head_dim]
#   v_seg_ptr: *fp32, [num_kv_tokens, head_dim]
#   lse_val: float32
# Outputs:
#   out_ptr: *fp32, [head_dim]
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
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    for start in range(0, num_kv_tokens, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        mask = offs < num_kv_tokens
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))
        k_tile = tl.load(k_seg_ptr + offs[:, None] * head_dim + tl.arange(0, head_dim), mask=mask[:, None], other=0.0)
        v_tile = tl.load(v_seg_ptr + offs[:, None] * head_dim + tl.arange(0, head_dim), mask=mask[:, None], other=0.0)
        logits = tl.zeros([CHUNK], dtype=tl.float32)
        for j in range(0, head_dim):
            logits += q_vec[j] * k_tile[:, j]
        logits = logits * sm_scale
        attn = tl.exp(logits - lse_val)  # [CHUNK]
        # Accumulate out_vec += sum(attn[:, None] * v_tile, axis=0)
        out_vec += tl.sum(attn[:, None] * v_tile, axis=0)

    tl.store(out_ptr + tl.arange(0, head_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Inputs must be on CUDA for Triton."
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()

        # Flatten dim=1 as in the original
        num_pages = k_cache.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k_cache.shape[2]

        # Squeeze dim=1
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim]

        total_q = q.shape[0]
        assert total_q == qo_indptr[-1].item(), "Sum of qo_indptr must equal total_q."

        # Output and lse
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        gqa_ratio = num_qo_heads // num_kv_heads

        # Iterate over batch segments
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            if num_q_tokens <= 0 or num_kv_tokens <= 0:
                continue

            # kv_indices must be on device
            if kv_indices.device != device:
                kv_indices = kv_indices.to(device)
            kv_indices_b = kv_indices[kv_start:kv_end]  # [num_kv_tokens]

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Causal window: max number of kv tokens this query sees
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                if max_kv_idx <= 0:
                    continue

                # For each head
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio  # 0..7

                    # Gather q vector for this head
                    q_vec = q[global_q_idx, h, :].to(torch.float32).contiguous()  # [head_dim]

                    # Gather k and v rows for this batch
                    k_rows = k_cache_flat[kv_indices_b[:max_kv_idx], kv_head, :].to(torch.float32).contiguous()  # [max_kv_idx, head_dim]
                    v_rows = v_cache_flat[kv_indices_b[:max_kv_idx], kv_head, :].to(torch.float32).contiguous()  # [max_kv_idx, head_dim]

                    # Compute lse using Triton
                    lse_buf = torch.empty(1, dtype=torch.float32, device=device)
                    CHUNK = 128  # tile size; 128 matches head_dim in the original
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

                    # Compute output vector using Triton
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
