import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    qn_ptr,   # *f32, [H, Dq]
    qp_ptr,   # *f32, [H, Dp]
    Kc_ptr,   # *f32, [T, Dq]
    Kp_ptr,   # *f32, [T, Dp]
    logits_ptr,  # *f32, [H, T]
    H: tl.constexpr,  # heads (16)
    Dq: tl.constexpr, # cache dim for Kc (512)
    Dp: tl.constexpr, # cache dim for Kp (64)
    T: tl.constexpr,  # tokens
    BLOCK_T: tl.constexpr,  # tile over tokens
):
    # Grid: (H, ceil_div(T, BLOCK_T))
    h = tl.program_id(0)
    tile_t = tl.program_id(1)
    offs_t = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Load q vectors for this head h
    qn_vec = tl.load(qn_ptr + h * Dq + tl.arange(0, Dq), mask=True, other=0.0)  # [Dq]
    qp_vec = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp), mask=True, other=0.0)  # [Dp]

    # Accumulate logits for tokens in this tile
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    # Loop over tokens (compile-time constant T)
    for t in range(0, T):
        # Load Kc[t, :] and Kp[t, :]
        Kc_t = tl.load(Kc_ptr + t * Dq + tl.arange(0, Dq), mask=True, other=0.0)  # [Dq]
        Kp_t = tl.load(Kp_ptr + t * Dp + tl.arange(0, Dp), mask=True, other=0.0)  # [Dp]
        # Compute dot products
        dot_qn = 0.0
        for i in range(0, Dq):
            dot_qn += qn_vec[i] * Kc_t[i]
        dot_qp = 0.0
        for j in range(0, Dp):
            dot_qp += qp_vec[j] * Kp_t[j]
        acc[tile_t * BLOCK_T + t] = dot_qn + dot_qp

    # Store acc into logits[h, t]
    tl.store(logits_ptr + h * T + offs_t, acc, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_scaled_ptr,  # *f32, [H, T]
    attn_ptr,           # *f32, [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # Grid: (H, ceil_div(T, BLOCK_T))
    h = tl.program_id(0)
    tile_t = tl.program_id(1)
    offs_t = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Load logits_scaled for this head's tile
    logits = tl.load(logits_scaled_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
    # Stable softmax
    row_max = tl.max(logits, axis=0)
    logits = logits - row_max
    exp_vals = tl.exp(logits)
    denom = tl.sum(exp_vals, axis=0)
    attn = exp_vals / denom
    tl.store(attn_ptr + h * T + offs_t, attn, mask=mask_t)


@triton.jit
def lse_row_kernel(
    logits_scaled_ptr,  # *f32, [H, T]
    lse_ptr,            # *f32, [H]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # Grid: (H,)
    h = tl.program_id(0)
    # Compute max and sum(exp(.)) across tokens for head h
    max_val = tl.full((), -float('inf'), tl.float32)
    for t in range(0, T):
        val = tl.load(logits_scaled_ptr + h * T + t)
        max_val = tl.maximum(max_val, val)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for t in range(0, T):
        val = tl.load(logits_scaled_ptr + h * T + t)
        sum_exp += tl.exp(val - max_val)
    lse_h = tl.log(sum_exp) + max_val  # logsumexp
    lse_h = lse_h / math.log(2.0)      # divide by ln(2)
    tl.store(lse_ptr + h, lse_h)


@triton.jit
def gemv_row_kernel(
    attn_ptr,   # *f32, [H, T]
    Kc_ptr,     # *f32, [T, Dq]
    out_ptr,    # *f32, [H, Dq]
    H: tl.constexpr,
    Dq: tl.constexpr,
    T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (H, ceil_div(Dq, BLOCK_D))
    h = tl.program_id(0)
    tile_d = tl.program_id(1)
    offs_d = tile_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < Dq

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    # Reduce over tokens
    for t in range(0, T):
        attn_t = tl.load(attn_ptr + h * T + t)  # scalar
        Kc_t = tl.load(Kc_ptr + t * Dq + offs_d, mask=mask_d, other=0.0)  # [BLOCK_D]
        acc += attn_t * Kc_t

    tl.store(out_ptr + h * Dq + offs_d, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors."

        # Dimensions as per original
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16."
        assert q_nope.shape[2] == 512, "head_dim_ckv must be 512."
        assert q_pe.shape[2] == 64, "head_dim_kpe must be 64."

        B = q_nope.shape[0]
        H = 16
        Dq = 512
        Dp = 64

        device = q_nope.device

        # Outputs
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # we'll fill via Triton
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Tile sizes (power-of-two)
        BLOCK_T = 128
        BLOCK_D = 128

        for b in range(B):
            # Tokens for this batch
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather token indices and cache rows
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)  # [L_tokens]
            Kc_b = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, Dq]
            Kp_b = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, Dp]

            # Load q vectors for this batch element
            qn = q_nope[b].contiguous().to(torch.float32)  # [H, Dq]
            qp = q_pe[b].contiguous().to(torch.float32)   # [H, Dp]

            # Allocate intermediates
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # 1) Compute logits[h, t] = qn[h] · Kc[t] + qp[h] · Kp[t]
            grid_logits = (H, triton.cdiv(L_tokens, BLOCK_T))
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, Dq=Dq, Dp=Dp, T=L_tokens,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Scale by sm_scale
            logits_scaled = logits * sm_scale

            # 2) Row-wise softmax over tokens
            grid_softmax = (H, triton.cdiv(L_tokens, BLOCK_T))
            softmax_row_kernel[grid_softmax](
                logits_scaled, attn,
                H=H, T=L_tokens,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # 3) Row-wise logsumexp and divide by ln(2)
            grid_lse = (H,)
            lse[b] = torch.empty((H,), dtype=torch.float32, device=device)
            lse_row_kernel[grid_lse](
                logits_scaled, lse[b],
                H=H, T=L_tokens,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # 4) Per-head GEMV: out[h, :] = attn[h, :] @ Kc[:, :] -> Triton
            out = torch.empty((H, Dq), dtype=torch.float32, device=device)
            grid_gemv = (H, triton.cdiv(Dq, BLOCK_D))
            gemv_row_kernel[grid_gemv](
                attn, Kc_b, out,
                H=H, Dq=Dq, T=L_tokens,
                BLOCK_D=BLOCK_D,
                num_warps=4, num_stages=2
            )

            # Store per-head output for this batch
            output[b] = out  # [H, Dq]

        # Return output and lse; cast output to bfloat16 to match original
        return output.to(torch.bfloat16), lse


# Helper to generate inputs (CUDA tensors for Triton)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)], 0).to(torch.int32).cuda()
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).cuda()
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
