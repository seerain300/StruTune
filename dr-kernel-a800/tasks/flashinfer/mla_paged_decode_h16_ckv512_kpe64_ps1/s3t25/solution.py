import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -------------------------
# Triton kernels
# -------------------------

# Fused logits computation: logits[h, t] = dot(qn[h, :], Kc[t, :]) + dot(qp[h, :], Kp[t, :])
# One Triton program per head (h). Iterates over tokens in tiles of BLOCK_T.
@triton.jit
def fused_logits_kernel(
    qn_ptr,            # *f32 [H, Dq]
    qp_ptr,            # *f32 [H, Dp]
    Kc_ptr,            # *f32 [T, Dq]
    Kp_ptr,            # *f32 [T, Dp]
    logits_ptr,        # *f32 [H, T]
    H: tl.constexpr,   # number of heads
    T: tl.constexpr,   # number of tokens
    Dq: tl.constexpr,  # 512
    Dp: tl.constexpr,  # 64
    BLOCK_T: tl.constexpr,  # e.g., 128 (power of 2)
):
    h = tl.program_id(0)
    # Each program handles one head h
    # Preload qn[h, :] and qp[h, :]
    # qn[h, :] has Dq elements; use mask for safety
    offs_qn = tl.arange(0, Dq)
    qn_row = tl.load(qn_ptr + h * Dq + offs_qn)  # [Dq]
    offs_qp = tl.arange(0, Dp)
    qp_row = tl.load(qp_ptr + h * Dp + offs_qp)  # [Dp]

    # Iterate over tokens in tiles
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T

        # Load Kc_sub [BLOCK_T, Dq]
        Kc_sub_ptrs = Kc_ptr + offs_t[:, None] * Dq + tl.arange(0, Dq)[None, :]  # shape (BLOCK_T, Dq)
        Kc_sub = tl.load(Kc_sub_ptrs, mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dq]

        # Load Kp_sub [BLOCK_T, Dp]
        Kp_sub_ptrs = Kp_ptr + offs_t[:, None] * Dp + tl.arange(0, Dp)[None, :]  # shape (BLOCK_T, Dp)
        Kp_sub = tl.load(Kp_sub_ptrs, mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dp]

        # Compute two dot-products: qn[h, :] @ Kc_sub and qp[h, :] @ Kp_sub
        dot1 = tl.sum(qn_row[None, :] * Kc_sub, axis=1)  # [BLOCK_T]
        dot2 = tl.sum(qp_row[None, :] * Kp_sub, axis=1)  # [BLOCK_T]
        logits_tile = dot1 + dot2  # [BLOCK_T]

        # Store logits[h, offs_t]
        logits_ptrs = logits_ptr + h * T + offs_t
        tl.store(logits_ptrs, logits_tile, mask=mask_t)


# Row-wise softmax: attn[h, :] = softmax(logits_scaled[h, :])
# One Triton program per head h. Iterate over tokens in tiles of BLOCK_T (power-of-2).
@triton.jit
def softmax_row_kernel(
    logits_ptr,        # *f32 [H, T]
    attn_ptr,          # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,  # e.g., 128 (power of 2)
):
    h = tl.program_id(0)
    # First pass: compute max for numerical stability
    max_val = -float("inf")
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
        current_max = tl.max(logits_tile, axis=0)
        max_val = tl.maximum(max_val, current_max)

    # Second pass: compute sum of exp(logits - max)
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
        exp_tile = tl.exp(logits_tile - max_val)
        sum_exp += tl.sum(exp_tile, axis=0)

    inv_sum = 1.0 / sum_exp

    # Third pass: write softmax
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
        attn_tile = tl.exp(logits_tile - max_val) * inv_sum
        tl.store(attn_ptr + h * T + offs_t, attn_tile, mask=mask_t)


# Row-wise logsumexp: lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
# One Triton program per head h. Iterate over tokens in tiles of BLOCK_T (power-of-2).
@triton.jit
def lse_row_kernel(
    logits_ptr,        # *f32 [H, T]
    lse_ptr,           # *f32 [H]
    H: tl.constexpr,
    T: tl.constexpr,
    sm_scale,          # f32 scalar
    BLOCK_T: tl.constexpr,  # e.g., 128 (power of 2)
):
    h = tl.program_id(0)
    max_val = -float("inf")
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
        scaled_tile = logits_tile * sm_scale
        current_max = tl.max(scaled_tile, axis=0)
        max_val = tl.maximum(max_val, current_max)

    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
        scaled_tile = logits_tile * sm_scale
        sum_exp += tl.sum(tl.exp(scaled_tile - max_val), axis=0)

    lse_val = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr + h, lse_val)


# Per-head matmul: out[h, :] = attn[h, :] @ Kc[:, :]
# One Triton program per head h. Iterate over Kc rows in tiles of BLOCK_D (power-of-2).
@triton.jit
def matmul_row_kernel(
    attn_ptr,          # *f32 [H, T]
    Kc_ptr,            # *f32 [T, Dq]
    out_ptr,           # *f32 [H, Dq]
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,  # 512 (power-of-2 or not; we use masked tiles)
    BLOCK_D: tl.constexpr,  # e.g., 128 (power of 2)
):
    h = tl.program_id(0)
    # Accumulator for out[h, :]
    out_vec = tl.zeros((Dq,), dtype=tl.float32)

    for t_start in range(0, T, BLOCK_T):  # BLOCK_T is power-of-2 tile for attn rows
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        attn_vec = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        # Iterate over Kc feature tiles
        for d_start in range(0, Dq, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            Kc_sub = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :], mask=mask_t[:, None], other=0.0)  # [BLOCK_T, BLOCK_D]
            acc = tl.sum(Kc_sub * attn_vec[:, None], axis=0)  # [BLOCK_D]
            out_vec += acc  # accumulate

    # Store out[h, :]
    out_ptrs = out_ptr + h * Dq + tl.arange(0, Dq)
    tl.store(out_ptrs, out_vec)


# -------------------------
# ModelNew forward
# -------------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, use_triton=True):
        # The evaluation harness may pass 8 args; we ignore 'use_triton' but keep signature.
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA."

        B, H, Dq = q_nope.shape
        _, _, Dp = q_pe.shape
        assert H == 16, "num_qo_heads must be 16"
        assert Dq == 512, "head_dim_ckv must be 512"
        assert Dp == 64, "head_dim_kpe must be 64"

        # Allocate output tensors
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # compute in float32
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            # Get number of tokens and token indices for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            # Ensure L_tokens >= 0
            if L_tokens <= 0:
                # No KV tokens for this batch element: set output zeros and lse to -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather Kc and Kp for this batch from cache
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int64)  # indices into [N, 1, D]
            # Gather: [L_tokens, Dq] and [L_tokens, Dp]
            Kc_b = ckv_cache[tok_idx, 0].to(torch.float32).contiguous()  # [L_tokens, 512]
            Kp_b = kpe_cache[tok_idx, 0].to(torch.float32).contiguous()  # [L_tokens, 64]
            # Ensure shapes
            assert Kc_b.shape[1] == Dq, "Kc_b second dim must be 512"
            assert Kp_b.shape[1] == Dp, "Kp_b second dim must be 64"

            # Load qn and qp for this batch head, cast to float32
            qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Allocate logits [H, T]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel: one program per head
            fused_logits_kernel[(H,)](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp, BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Launch lse_row_kernel: compute lse per head
            lse_row_kernel[(H,)](
                logits, lse[b],
                H=H, T=L_tokens, sm_scale=float(sm_scale), BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Launch softmax_row_kernel to compute attn per head
            attn = torch.empty_like(logits, dtype=torch.float32, device=device)  # [H, T]
            softmax_row_kernel[(H,)](
                logits, attn,
                H=H, T=L_tokens, BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Compute per-head outputs out[h, :] = attn[h, :] @ Kc_b[:, :]
            # One program per head
            for h in range(H):
                out_vec = torch.empty((Dq,), dtype=torch.float32, device=device)
                matmul_row_kernel[(1,)](
                    attn[h], Kc_b, out_vec,
                    H=H, T=L_tokens, Dq=Dq, BLOCK_D=128,
                    num_warps=4, num_stages=2
                )
                output[b, h] = out_vec

        # Cast output back to bfloat16 to match original code
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


# -------------------------
# Optional: keep original run for compatibility if needed
# -------------------------
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    return ModelNew().forward(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, use_triton=True)

def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device="cuda")
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).cuda()
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# -------------------------
# Entry point: fused_operator
# -------------------------
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    out, lse = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return out if isinstance(out, (tuple, list)) else [out]


def run(*args):
    return ModelNew()(*args)
