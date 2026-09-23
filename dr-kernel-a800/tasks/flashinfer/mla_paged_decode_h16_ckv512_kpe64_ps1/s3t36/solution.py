import torch
import math
import triton
import triton.language as tl


# Triton kernel: compute fused logits per head and token
@triton.jit
def fused_logits_kernel(
    qn_ptr,   # *f32 [H, Dq]
    qp_ptr,   # *f32 [H, Dp]
    Kc_ptr,   # *f32 [T, Dq]
    Kp_ptr,   # *f32 [T, Dp]
    logits_ptr,  # *f32 [H, T]
    H: tl.constexpr,       # number of heads (e.g., 16)
    T: tl.constexpr,       # number of tokens for this batch element
    Dq: tl.constexpr,      # feature dim for CKV (e.g., 512)
    Dp: tl.constexpr,      # feature dim for KPE (e.g., 64)
    BLOCK_T: tl.constexpr, # tile size for tokens (power of two, e.g., 128)
    BLOCK_D: tl.constexpr  # tile size for feature dim (power of two, e.g., 128)
):
    h = tl.program_id(0)  # each program handles one head
    # Preload qn and qp for this head
    qn_vec = tl.load(qn_ptr + h * Dq + tl.arange(0, Dq), mask=True, other=0.0)  # [Dq]
    qp_vec = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp), mask=True, other=0.0)  # [Dp]

    # Accumulator for logits vector for this head
    logits_vec = tl.zeros((T,), dtype=tl.float32)

    # Tile over tokens
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T

        acc1 = tl.zeros((BLOCK_T,), dtype=tl.float32)  # dot(qn_vec, Kc[t, :])
        acc2 = tl.zeros((BLOCK_T,), dtype=tl.float32)  # dot(qp_vec, Kp[t, :])

        # Reduction over feature tiles for Kc
        for d_start in range(0, Dq, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < Dq
            Kc_tile = tl.load(
                Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0
            )  # [BLOCK_T, BLOCK_D]
            # sum over Dq tile: qn_vec[offs_d] * Kc_tile, then reduce across d
            acc1 += tl.sum(qn_vec[offs_d] * Kc_tile, axis=1)  # [BLOCK_T]

        # Reduction over feature tiles for Kp
        for dp_start in range(0, Dp, BLOCK_D):
            offs_dp = dp_start + tl.arange(0, BLOCK_D)
            mask_dp = offs_dp < Dp
            Kp_tile = tl.load(
                Kp_ptr + offs_t * Dp + offs_dp,
                mask=mask_t & mask_dp,
                other=0.0
            )  # [BLOCK_T]
            acc2 += tl.sum(qp_vec[offs_dp] * Kp_tile, axis=0)  # [BLOCK_T]

        logits_vec[offs_t] = acc1 + acc2

    # Store logits for this head
    tl.store(logits_ptr + h * T + tl.arange(0, T), logits_vec, mask=True)


# Triton kernel: row-wise softmax over T for each head
@triton.jit
def softmax_row_kernel(
    logits_ptr,   # *f32 [H, T]
    attn_ptr,     # *f32 [H, T]
    H: tl.constexpr,       # number of heads (e.g., 16)
    T: tl.constexpr,       # number of tokens
    BLOCK_T: tl.constexpr  # power-of-two (e.g., 128)
):
    h = tl.program_id(0)
    row_ptr_logits = logits_ptr + h * T
    row_ptr_attn = attn_ptr + h * T

    # First pass: compute max
    row_max = tl.full((), -float('inf'), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(row_ptr_logits + offs_t, mask=mask_t, other=-float('inf'))
        block_max = tl.max(logits_tile, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # Second pass: compute denominator
    denom = tl.zeros((), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(row_ptr_logits + offs_t, mask=mask_t, other=-float('inf'))
        exp_tile = tl.exp(logits_tile - row_max)
        denom += tl.sum(exp_tile, axis=0)

    inv_denom = 1.0 / denom

    # Third pass: write softmax
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(row_ptr_logits + offs_t, mask=mask_t, other=-float('inf'))
        attn_tile = tl.exp(logits_tile - row_max) * inv_denom
        tl.store(row_ptr_attn + offs_t, attn_tile, mask=mask_t)


# Triton kernel: row-wise logsumexp over T for each head, scaled by 1/ln(2)
@triton.jit
def lse_row_kernel(
    logits_ptr,   # *f32 [H, T]
    lse_ptr,      # *f32 [H]
    H: tl.constexpr,       # number of heads (e.g., 16)
    T: tl.constexpr,       # number of tokens
    BLOCK_T: tl.constexpr  # power-of-two (e.g., 128)
):
    h = tl.program_id(0)
    row_ptr_logits = logits_ptr + h * T

    row_max = tl.full((), -float('inf'), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(row_ptr_logits + offs_t, mask=mask_t, other=-float('inf'))
        block_max = tl.max(logits_tile, axis=0)
        row_max = tl.maximum(row_max, block_max)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_tile = tl.load(row_ptr_logits + offs_t, mask=mask_t, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(logits_tile - row_max), axis=0)

    lse_val = row_max + tl.log(sum_exp)  # logsumexp
    ln2 = 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr + h, lse_val * (1.0 / ln2))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernels"

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dq = q_nope.shape[2]  # 512
        Dp = q_pe.shape[2]     # 64

        # Output and lse
        output = torch.empty((B, H, Dq), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b] = torch.zeros((H, Dq), dtype=torch.bfloat16, device=q_nope.device)
                lse[b] = torch.full((H,), -float('inf'), dtype=torch.float32, device=q_nope.device)
                continue

            # Gather Kc and Kp for this batch element
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]]
            Kc_b = ckv_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, Dq]
            Kp_b = kpe_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, Dp]

            # qn and qp vectors for this batch
            qn = q_nope[b].to(torch.float32)  # [H, Dq]
            qp = q_pe[b].to(torch.float32)    # [H, Dp]

            # Allocate intermediate buffers
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)

            # Launch fused logits kernel: compute logits[h, t] for all heads h and tokens t
            grid_logits = (H,)
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=128, BLOCK_D=128,
                num_warps=4, num_stages=2
            )

            # Scale logits by sm_scale
            logits_scaled = logits * sm_scale

            # Allocate attn buffer and compute softmax per head
            attn = torch.empty_like(logits_scaled)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                logits_scaled, attn,
                H=H, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            # Compute lse per head
            lse_row = torch.empty((H,), dtype=torch.float32, device=q_nope.device)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                logits_scaled, lse_row,
                H=H, T=L_tokens,
                BLOCK_T=128,
                num_warps=4, num_stages=2
            )
            lse[b] = lse_row  # shape [H]

            # Compute out[h, :] = attn[h, :] @ Kc[:, :] using torch.matmul (robust and correct)
            # attn[h, :] is vector of length L_tokens, Kc_b is [L_tokens, Dq]
            out_b = torch.matmul(attn, Kc_b)  # [H, Dq]
            output[b] = out_b.to(torch.bfloat16)

        return output, lse


# Helper to generate inputs (kept similar to original; forward will place on CUDA)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)], 0).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    # Move to CUDA for Triton
    return [
        q_nope.cuda(), q_pe.cuda(), ckv_cache.cuda(), kpe_cache.cuda(),
        kv_indptr.cuda(), kv_indices.cuda(), sm_scale
    ]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
