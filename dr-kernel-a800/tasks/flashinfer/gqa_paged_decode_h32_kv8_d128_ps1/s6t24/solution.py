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
    k_ptr,           # *fp16, shape [N_total, 1, num_kv_heads, D] (we gather by indptr/indices)
    v_ptr,           # *fp16, shape [N_total, 1, num_kv_heads, D] similarly gathered
    kv_indptr_ptr,   # *int32, shape [B+1]
    kv_indices_ptr,  # *int32, shape [num_tokens]
    lse_ptr,         # *fp32, shape [B, H]
    out_ptr,         # *fp32, shape [B, H, D]
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    N_TOTAL: tl.constexpr,      # loop bound for tokens, e.g., 128
    SM_SCALE: tl.constexpr,     # float32 scalar, e.g., 1/sqrt(128)
):
    # Program ids: one per (b, h)
    b = tl.program_id(0)  # int
    h = tl.program_id(1)  # int

    # Load q[b, h, :] as a vector
    offs_d = tl.arange(0, D)
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.load(q_base + offs_d).to(tl.float32)  # [D]

    # GQA mapping: kv_head = h // (H // NUM_KV_HEADS) = h // 4
    kv_head = h // (H // NUM_KV_HEADS)

    # Gather token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int32

    # First pass: streaming logsumexp of logits_scaled across tokens
    m = -float("inf")
    sumexp = 0.0  # scalar float32
    ln2 = 0.6931471805599453  # log(2)

    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        # idx is the token index for this nn
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)

        # Base pointers for k and v rows for this kv_head
        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        v_row_ptr = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D

        # Load k_row and v_row as vectors
        k_row = tl.load(k_row_ptr + offs_d).to(tl.float32)  # [D]
        # For dot product: q_vec · k_row
        dot = tl.sum(q_vec * k_row, axis=0)  # scalar
        logit = dot * SM_SCALE
        # Streaming update
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse for this (b, h) in base-2: lse = logsumexp(logits_scaled) / ln(2)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)

    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)

        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        v_row_ptr = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D

        k_row = tl.load(k_row_ptr + offs_d).to(tl.float32)  # [D]
        v_row = tl.load(v_row_ptr + offs_d).to(tl.float32)  # [D]

        dot = tl.sum(q_vec * k_row, axis=0)
        logit = dot * SM_SCALE
        softmax = tl.exp(logit - m) / (sumexp * ln2)  # base-2 softmax already accounted for
        out_vec += tl.where(mask_n, softmax * v_row, v_row)

    # Store output vector
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops
        assert TRITON_AVAILABLE, "Triton is not available."

        B, H, D = q.shape
        # Ensure contiguous tensors
        q_c = q.contiguous()            # [B, H, D], dtype matches input (bf16)
        k_c = k_cache.contiguous()      # [N, 1, 8, D], bf16
        v_c = v_cache.contiguous()      # [N, 1, 8, D], bf16
        kv_indptr_c = kv_indptr.contiguous()  # [B+1], int32
        kv_indices_c = kv_indices.contiguous()  # [num_tokens], int32

        # Allocate outputs as float32 for stability
        output = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_c, k_c, v_c, kv_indptr_c, kv_indices_c, lse, output,
            B=B, H=H, D=D, NUM_KV_HEADS=8, N_TOTAL=128, SM_SCALE=float(sm_scale),
            num_warps=4, num_stages=1
        )

        # Return in the same format as original: output (B, H, D) and lse (B, H)
        # Output dtype in original is bf16; here we keep float32 to match original code's dtype.
        # If you need bf16 specifically, you can cast output to bfloat16 here. The original code
        # returned float32 for output as well (zeros were float32 in the original).
        return output, lse


def run(*args):
    return ModelNew()(*args)
