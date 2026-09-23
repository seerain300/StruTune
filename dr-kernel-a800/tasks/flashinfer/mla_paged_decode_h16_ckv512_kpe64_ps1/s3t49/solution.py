import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_ptr,       # *f32 [H, Dq]
    qp_ptr,       # *f32 [H, Dp]
    Kc_ptr,       # *f32 [T, Dq]
    Kp_ptr,       # *f32 [T, Dp]
    out_ptr,      # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,
    Dp: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    ts = tl.program_id(1)
    offs_t = ts * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Load qn[h, :] and qp[h, :]
    qn = tl.load(qn_ptr + h * Dq + tl.arange(0, Dq), mask=True, other=0.0)  # [Dq]
    qp = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp), mask=True, other=0.0)  # [Dp]

    # Accumulate logits for these tokens
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(T):
        # Load Kc[t, :] and Kp[t, :]
        Kc_t = tl.load(Kc_ptr + t * Dq + tl.arange(0, Dq), mask=True, other=0.0)  # [Dq]
        Kp_t = tl.load(Kp_ptr + t * Dp + tl.arange(0, Dp), mask=True, other=0.0)  # [Dp]
        # Compute dot products
        dot_qn = tl.sum(qn * Kc_t)
        dot_qp = tl.sum(qp * Kp_t)
        acc += dot_qn + dot_qp

    # Store to output
    tl.store(out_ptr + h * T + offs_t, acc, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    in_ptr,       # *f32 [H, T]
    out_ptr,      # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    ts = tl.program_id(1)
    offs_t = ts * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Row pointer for this head
    row_ptr = in_ptr + h * T
    x = tl.load(row_ptr + offs_t, mask=mask_t, other=-float('inf'))
    # Compute row-wise max
    row_max = tl.max(x, axis=0)
    x = x - row_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    attn = exp_x / sum_exp
    tl.store(out_ptr + h * T + offs_t, attn, mask=mask_t)


@triton.jit
def lse_row_kernel(
    in_ptr,       # *f32 [H, T]
    out_ptr,      # *f32 [H]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    row_ptr = in_ptr + h * T
    x = tl.load(row_ptr, mask=True, other=-float('inf'))
    row_max = tl.max(x, axis=0)
    sum_exp = tl.sum(tl.exp(x - row_max), axis=0)
    ln2 = 1.4426950408889634  # 1 / ln(2)
    lse = row_max + tl.log(sum_exp) / ln2
    tl.store(out_ptr + h, lse)


@triton.jit
def perhead_gemv_kernel(
    attn_ptr,     # *f32 [H, T]
    Kc_ptr,       # *f32 [T, Dq]
    out_ptr,      # *f32 [H, Dq] (we will store per (b,h) via offset)
    H: tl.constexpr,
    T: tl.constexpr,
    Dq: tl.constexpr,
    BLOCK_DO: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    do = tl.program_id(1)
    offs_do = do * BLOCK_DO + tl.arange(0, BLOCK_DO)
    mask_do = offs_do < Dq

    acc = tl.zeros((BLOCK_DO,), dtype=tl.float32)
    for t in range(0, T, BLOCK_T):
        offs_t = t + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        attn_vec = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_do[None, :],  # [BLOCK_T, BLOCK_DO]
                          mask=mask_t[:, None] & mask_do[None, :], other=0.0)
        # acc[offs_do] += sum_t attn_vec[t] * Kc_tile[t, offs_do]
        # Compute per offs_do scalar accumulation for this tile
        for i in range(BLOCK_T):
            acc += attn_vec[i] * Kc_tile[i, :]
    tl.store(out_ptr + h * Dq + offs_do, acc, mask=mask_do)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA"
        device = q_nope.device

        B, H, Dq = q_nope.shape
        _, _, Dp = q_pe.shape
        N = ckv_cache.shape[0]
        assert Dq == 512 and Dp == 64 and H == 16, "Expected H=16, Dq=512, Dp=64"

        # Allocate output and lse
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # will be cast to bfloat16
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Iterate batch elements
        BLOCK_T = 128  # power-of-two for tl.arange
        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            # If no tokens, skip
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int64)

            # Load Kc and Kp for these tokens
            Kc_b = ckv_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, 512]
            Kp_b = kpe_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, 64]

            # Load q_nope[b] and q_pe[b]
            qn_b = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp_b = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Allocate intermediates
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            lse_row = torch.empty((H,), dtype=torch.float32, device=device)

            # 1) Compute logits_scaled = qn[h, :] · Kc[t, :] + qp[h, :] · Kp[t, :]
            grid = (H, triton.cdiv(L_tokens, BLOCK_T))
            compute_logits_kernel[grid](
                qn_b, qp_b, Kc_b, Kp_b, logits, H, L_tokens, Dq, Dp, float(sm_scale), BLOCK_T
            )

            # 2) Row-wise softmax over tokens
            grid = (H, triton.cdiv(L_tokens, BLOCK_T))
            softmax_row_kernel[grid](logits, attn, H, L_tokens, BLOCK_T)

            # 3) Row-wise logsumexp divided by ln(2)
            grid = (H,)
            lse_row_kernel[grid](attn, lse_row, H, L_tokens, BLOCK_T)
            lse[b] = lse_row  # shape [H]

            # 4) Per-head GEMV: out[h, :] = attn[h, :] @ Kc_b[:, :]
            # We store per (b,h) slice in output[b, :, :]
            grid = (H, triton.cdiv(Dq, 128))
            perhead_gemv_kernel[grid](attn, Kc_b, output[b], H, L_tokens, Dq, 128, BLOCK_T)

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse

def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)], 0)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
