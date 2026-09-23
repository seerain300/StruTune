import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_kernel_bh(
    q_ptr,           # *fp16, shape [B, H, D]
    k_ptr,           # *fp16, shape [N_total, 1, num_kv_heads, D] (gather by indptr/indices)
    v_ptr,           # *fp16, shape [N_total, 1, num_kv_heads, D]
    kv_indptr_ptr,   # *int32, shape [B+1]
    kv_indices_ptr,  # *int32, shape [num_tokens]
    lse_ptr,         # *fp32, shape [B, H]
    out_ptr,         # *fp32, shape [B, H, D]
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SM_SCALE: tl.constexpr,          # python float, e.g., 1.0 / sqrt(128)
    N_TOTAL: tl.constexpr,           # loop bound (e.g., 128), mask beyond actual_num_tokens
):
    b = tl.program_id(0)  # 0..B-1
    h = tl.program_id(1)  # 0..H-1

    # Load q vector q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    offs = tl.arange(0, D)
    q_vec = tl.load(q_base + offs).to(tl.float32)  # [D]

    # Compute start and end for this batch in kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start

    # GQA mapping: kv_head = h // (H // NUM_KV_HEADS)
    gqa_ratio = H // NUM_KV_HEADS
    kv_head = h // gqa_ratio

    ln2 = 0.6931471805599453  # math.log(2.0)

    # First pass: compute logsumexp across tokens in base-2 and cache per-token scale s_n
    m = -float("inf")
    sumexp = 0.0
    s_vec = tl.zeros((N_TOTAL,), dtype=tl.float32)  # per-token softmax scaling
    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)

        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        offs_d = tl.arange(0, D)
        k_vec = tl.load(k_row_ptr + offs_d).to(tl.float32)  # [D]
        # dot = sum(q_vec * k_vec)
        dot = tl.sum(q_vec * k_vec, axis=0)
        logit = dot * SM_SCALE
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

        # cache per-token softmax scaling s_n = exp(logit - m) / (sumexp * ln2)
        s_vec[nn] = tl.exp(logit - m) / (sumexp * ln2)

    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: accumulate output using cached s_n
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        # Load v_row for this token
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)
        v_row_ptr = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        offs_d = tl.arange(0, D)
        v_row = tl.load(v_row_ptr + offs_d).to(tl.float32)  # [D]

        # Apply cached scaling and accumulate
        out_vec += tl.where(mask_n, s_vec[nn] * v_row, v_row)

    # store output
    tl.store(out_base + offs, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops
        assert TRITON_AVAILABLE, "Triton is not available."

        B, H, D = q.shape
        # Ensure inputs are contiguous
        q_c = q.contiguous()          # [B, H, D]
        k_c = k_cache.contiguous()    # [N, 1, 8, D]
        v_c = v_cache.contiguous()    # [N, 1, 8, D]
        kv_indptr_c = kv_indptr.contiguous()  # [B+1]
        kv_indices_c = kv_indices.contiguous()  # [num_tokens]

        # Allocate outputs as float32; convert to bfloat16 at end
        output = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: grid = (B, H)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_c, k_c, v_c,
            kv_indptr_c, kv_indices_c,
            lse, output,
            B, H, D,
            8,  # NUM_KV_HEADS
            float(sm_scale),  # SM_SCALE as float
            128,  # N_TOTAL loop bound (safe upper limit; mask handles actual_num_tokens)
            num_warps=4,
        )

        # Match original output dtype: bfloat16 for output, float32 for lse
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
