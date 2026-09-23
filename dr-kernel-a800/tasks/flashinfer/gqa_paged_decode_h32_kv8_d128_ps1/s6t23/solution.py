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
    k_ptr,           # *fp16, shape [N_total, 1, num_kv_heads, D] (we gather rows by index)
    v_ptr,           # *fp16, shape [N_total, 1, num_kv_heads, D]
    kv_indptr_ptr,   # *int32, shape [B+1]
    kv_indices_ptr,  # *int32, shape [num_tokens]
    lse_ptr,         # *fp32, shape [B, H]
    out_ptr,         # *fp32, shape [B, H, D]
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SM_SCALE: tl.constexpr,    # scalar float32
    N_TOTAL: tl.constexpr,     # compile-time loop bound (>= num_tokens)
    LN2: tl.constexpr,         # 0.6931471805599453
):
    b = tl.program_id(0)  # 0..B-1
    h = tl.program_id(1)  # 0..H-1

    # GQA mapping: 32 query heads, 8 kv heads -> kv_head = h // (32//8) = h // 4
    kv_head = h // (H // NUM_KV_HEADS)

    # Load q vector q[b, h, :] (D=128)
    q_base = q_ptr + b * H * D + h * D
    offs_d = tl.arange(0, D)
    q_vec = tl.load(q_base + offs_d).to(tl.float32)  # [D], fp32

    # Compute start/end from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start

    # First pass: streaming logsumexp over tokens in base-2
    m = -float("inf")
    sumexp = 0.0
    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)

        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        v_row_ptr = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D

        k_row = tl.load(k_row_ptr + offs_d).to(tl.float32)  # [D]
        v_row = tl.load(v_row_ptr + offs_d).to(tl.float32)  # [D]

        # dot product
        dot = tl.sum(q_vec * k_row, axis=0)  # scalar
        logit = dot * SM_SCALE

        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse = logsumexp(logit) / ln(2)
    lse_val = m + tl.log(sumexp) / LN2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax and accumulate output
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
        softmax = tl.exp(logit - m) / sumexp  # natural log accumulation
        out_vec += tl.where(mask_n, softmax * v_row, v_row)

    # store output
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops
        assert TRITON_AVAILABLE, "Triton is not available."

        # Shapes from inputs
        B, H, D = q.shape
        # Ensure contiguous tensors
        q_c = q.contiguous()          # [B, H, D]
        k_c = k_cache.contiguous()    # [N_total, 1, 8, D]
        v_c = v_cache.contiguous()    # [N_total, 1, 8, D]
        kv_indptr_c = kv_indptr.contiguous()  # [B+1], int32
        kv_indices_c = kv_indices.contiguous()  # [num_tokens], int32

        # Allocate outputs
        output = torch.empty((B, H, D), dtype=torch.float32, device=q.device)  # we compute in fp32
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        LN2 = 0.6931471805599453
        _forward_kernel_bh[grid](
            q_c, k_c, v_c, kv_indptr_c, kv_indices_c, lse, output,
            B=B, H=H, D=D, NUM_KV_HEADS=8, SM_SCALE=float(sm_scale), N_TOTAL=128, LN2=LN2,
            num_warps=4, num_stages=2
        )

        # Return output and lse as float32 (original code returns lse float32)
        return output, lse


def run(*args):
    return ModelNew()(*args)
