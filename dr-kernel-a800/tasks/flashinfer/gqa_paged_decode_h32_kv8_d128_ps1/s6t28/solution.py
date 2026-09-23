import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_bh_kernel(
    q_ptr,              # *ptr float32, shape [B, H, D]
    k_ptr,              # *ptr float32, shape [N_total, num_kv_heads, D]
    v_ptr,              # *ptr float32, shape [N_total, num_kv_heads, D]
    kv_indices_ptr,     # *ptr int32, shape [num_kv_indices]
    kv_indptr_ptr,      # *ptr int32, shape [B+1]
    lse_ptr,            # *ptr float32, shape [B, H]
    out_ptr,            # *ptr float32, shape [B, H, D]
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    num_kv_heads: tl.constexpr,
    sm_scale,           # scalar float32
    N_TOTAL: tl.constexpr,  # loop bound (e.g., 128), masked for actual_num_tokens
):
    # Program ids for batch and head
    b = tl.program_id(0)  # int32
    h = tl.program_id(1)  # int32

    # 1) Load q[b, h, :] into float32 vector
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d)

    # 2) Compute start and end from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int32

    # GQA mapping: kv_head = h // (H // num_kv_heads) = h // 4
    kv_head = h // 4

    # First pass: streaming logsumexp in base-2
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453  # log(2)

    for nn in range(0, N_TOTAL):
        token_off = start + nn
        valid = nn < actual_num_tokens  # scalar bool

        # Load index if valid; otherwise dummy 0
        idx = tl.load(kv_indices_ptr + token_off, mask=valid, other=0).to(tl.int32)

        # Base pointers for k_row and v_row
        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        # Load k_row and compute dot = q_vec · k_row
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_base + d, mask=valid, other=0.0)
            k_vec[d] = k_val

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse for this (b, h): logsumexp in base-2
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # 3) Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        token_off = start + nn
        valid = nn < actual_num_tokens

        idx = tl.load(kv_indices_ptr + token_off, mask=valid, other=0).to(tl.int32)

        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_base + d, mask=valid, other=0.0)
            k_vec[d] = k_val

        # dot and logit
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]
        logit = dot * sm_scale

        # softmax = exp(logit - m) / (sumexp * ln2)
        softmax = tl.exp(logit - m) / (sumexp * ln2)

        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_val = tl.load(v_row_base + d, mask=valid, other=0.0)
            v_vec[d] = v_val

        out_vec += softmax * v_vec

    tl.store(out_base, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Inputs: q [B, H, D], k_cache [N, 1, num_kv_heads, D], v_cache [N, 1, num_kv_heads, D]
        # We perform all computation in Triton; no PyTorch ops in forward.
        B, H, D = q.shape
        num_kv_heads = 8  # as per original assertions
        GQA_ratio = H // num_kv_heads  # 4

        # Allocate outputs (float32 for computation)
        device = q.device
        lse = torch.empty((B, H), dtype=torch.float32, device=device)
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)

        # Convert inputs to float32 and ensure contiguity for Triton
        q_f32 = q.to(torch.float32).contiguous()
        # Flatten k_cache and v_cache to [N, num_kv_heads, D]
        k_flat = k_cache.view(-1, num_kv_heads, D).to(torch.float32).contiguous()
        v_flat = v_cache.view(-1, num_kv_heads, D).to(torch.float32).contiguous()

        kv_indptr_i32 = kv_indptr.to(torch.int32)
        kv_indices_i32 = kv_indices.to(torch.int32)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_bh_kernel[grid](
            q_f32, k_flat, v_flat, kv_indices_i32, kv_indptr_i32, lse, output,
            B=B, H=H, D=D, num_kv_heads=num_kv_heads,
            sm_scale=float(sm_scale),
            N_TOTAL=128,  # loop bound; mask covers actual_num_tokens
            num_warps=4, num_stages=2
        )

        # Return outputs with expected dtypes
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
