import torch
import math
import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    qn_ptr,   # *f32 [H, Dq]
    qp_ptr,   # *f32 [H, Dp]
    Kc_ptr,   # *f32 [T, Dq]
    Kp_ptr,   # *f32 [T, Dp]
    logits_ptr,  # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,  # 512
    Dp: tl.constexpr,  # 64
    BLOCK_T: tl.constexpr = 128,
):
    # One program per (head, tile-of-tokens)
    h = tl.program_id(0)
    t_start = tl.program_id(1) * BLOCK_T

    offs_t = t_start + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # Loop over feature dims in tiles
    for i in range(0, Dq, 128):
        offs_q = i + tl.arange(0, 128)
        mask_q = offs_q < Dq

        qn_vec = tl.load(qn_ptr + h * Dq + offs_q, mask=mask_q, other=0.0)  # [128]
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_q[None, :], mask=mask_t[:, None] & mask_q[None, :], other=0.0)  # [BLOCK_T, 128]
        acc += tl.sum(Kc_tile * qn_vec[None, :], axis=1)  # [BLOCK_T]

    for i in range(0, Dp, 32):
        offs_p = i + tl.arange(0, 32)
        mask_p = offs_p < Dp

        qp_vec = tl.load(qp_ptr + h * Dp + offs_p, mask=mask_p, other=0.0)  # [32]
        Kp_tile = tl.load(Kp_ptr + offs_t[:, None] * Dp + offs_p[None, :], mask=mask_t[:, None] & mask_p[None, :], other=0.0)  # [BLOCK_T, 32]
        acc += tl.sum(Kp_tile * qp_vec[None, :], axis=1)  # [BLOCK_T]

    tl.store(logits_ptr + h * T + offs_t, acc, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_ptr,   # *f32 [H, T]
    attn_ptr,     # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr = 128,
):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Load logits for this row
    logits = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))  # [BLOCK_T]
    # Stable softmax
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    attn = exp_logits / sum_exp
    tl.store(attn_ptr + h * T + offs_t, attn, mask=mask_t)


@triton.jit
def lse_row_kernel(
    logits_ptr,   # *f32 [H, T]
    lse_ptr,      # *f32 [H]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr = 128,
):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Load logits for this row
    logits = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val
    sum_exp = tl.sum(tl.exp(logits), axis=0)
    lse_val = tl.log(sum_exp) / 0.6931471805599453  # ln(2)
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def gemv_row_kernel(
    attn_ptr,     # *f32 [H, T]
    Kc_ptr,       # *f32 [T, Dq]
    out_ptr,      # *f32 [H, Dq]
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,  # 512
    BLOCK_D: tl.constexpr = 128,
    BLOCK_T: tl.constexpr = 128,
):
    # One program per head h
    h = tl.program_id(0)
    # Initialize output vector
    out_vec = tl.zeros((Dq,), dtype=tl.float32)

    # Loop over tokens in tiles
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        attn = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]

        # Loop over feature dim in tiles
        for j in range(0, Dq, BLOCK_D):
            offs_j = j + tl.arange(0, BLOCK_D)
            mask_j = offs_j < Dq

            # Load Kc[j, :] for this tile
            Kc_tile = tl.load(Kc_ptr + offs_j * T + (offs_t[:, None] * 1), mask=mask_j[:, None] & mask_t[None, :], other=0.0)
            # Multiply and reduce across tokens
            contrib = tl.sum(Kc_tile * attn[None, :], axis=1)  # [BLOCK_D]
            out_vec += contrib

    tl.store(out_ptr + h * Dq + tl.arange(0, Dq), out_vec, mask=tl.arange(0, Dq) < Dq)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA device for Triton kernels"

    # Dimensions
    B = q_nope.shape[0]
    H = q_nope.shape[1]
    Dq = q_nope.shape[2]  # 512
    Dp = q_pe.shape[2]    # 64

    # Output and lse
    output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    for b in range(B):
        # Compute L_tokens and indices for this batch
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            # No KV cache for this batch element
            output[b] = torch.zeros((H, Dq), dtype=torch.float32, device=device)
            lse[b] = torch.full((H,), -float("inf"), dtype=torch.float32, device=device)
            continue

        tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(device).to(torch.int32)
        Kc_b = ckv_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, Dq]
        Kp_b = kpe_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, Dp]

        # Prepare qn and qp
        qn = q_nope[b].to(torch.float32)  # [H, Dq]
        qp = q_pe[b].to(torch.float32)    # [H, Dp]

        # Allocate intermediate
        logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

        # Launch fused_logits_kernel
        grid = (H, (L_tokens + 127) // 128)
        fused_logits_kernel[grid](
            qn, qp, Kc_b, Kp_b, logits,
            H=H, T=L_tokens, Dq=Dq, Dp=Dp,
            BLOCK_T=128,
            num_warps=4, num_stages=2
        )

        # Scale logits by sm_scale
        logits_scaled = logits * sm_scale

        # Row-wise softmax
        attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
        grid_softmax = (H,)
        softmax_row_kernel[grid_softmax](
            logits_scaled, attn,
            H=H, T=L_tokens,
            BLOCK_T=128,
            num_warps=4, num_stages=2
        )

        # Row-wise LSE / ln(2)
        lse_row = torch.empty((H,), dtype=torch.float32, device=device)
        grid_lse = (H,)
        lse_row_kernel[grid_lse](
            logits_scaled, lse_row,
            H=H, T=L_tokens,
            BLOCK_T=128,
            num_warps=4, num_stages=2
        )
        lse[b] = lse_row  # shape [H]

        # Per-head GEMV: out[h, :] = attn[h, :] @ Kc[:, :]
        out_b = torch.empty((H, Dq), dtype=torch.float32, device=device)
        grid_gemv = (H,)
        gemv_row_kernel[grid_gemv](
            attn, Kc_b, out_b,
            H=H, T=L_tokens, Dq=Dq,
            BLOCK_D=128, BLOCK_T=128,
            num_warps=4, num_stages=2
        )
        output[b] = out_b

    # Cast output to bfloat16 to match original
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


# Helper to generate inputs (ensure CUDA tensors for Triton)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)], 0).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
