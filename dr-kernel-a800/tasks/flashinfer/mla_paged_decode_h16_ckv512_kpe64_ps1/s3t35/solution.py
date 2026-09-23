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
    H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr
):
    h = tl.program_id(0)
    t_block = tl.program_id(1)

    offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # Reduce over Dq in tiles
    for d_start in range(0, Dq, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < Dq
        qn_vec = tl.load(qn_ptr + h * Dq + offs_d, mask=mask_d, other=0.0)  # [BLOCK_D]
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)
        acc += tl.sum(Kc_tile * qn_vec[None, :], axis=1)

    # Reduce over Dp in tiles
    for d_start in range(0, Dp, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < Dp
        qp_vec = tl.load(qp_ptr + h * Dp + offs_d, mask=mask_d, other=0.0)  # [BLOCK_D]
        Kp_tile = tl.load(Kp_ptr + offs_t[:, None] * Dp + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)
        acc += tl.sum(Kp_tile * qp_vec[None, :], axis=1)

    tl.store(logits_ptr + h * T + offs_t, acc, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_ptr,  # *f32 [H, T]
    attn_ptr,    # *f32 [H, T]
    H: tl.constexpr, T: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask = offs_t < T

    logits_row = tl.load(logits_ptr + h * T + offs_t, mask=mask, other=-float('inf'))
    max_val = tl.max(logits_row, axis=0)
    x = logits_row - max_val
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    attn_row = num / den
    tl.store(attn_ptr + h * T + offs_t, attn_row, mask=mask)


@triton.jit
def lse_row_kernel(
    logits_ptr,   # *f32 [H, T]
    lse_ptr,      # *f32 [H]
    H: tl.constexpr, T: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask = offs_t < T

    logits_row = tl.load(logits_ptr + h * T + offs_t, mask=mask, other=-float('inf'))
    max_val = tl.max(logits_row, axis=0)
    x = logits_row - max_val
    sum_exp = tl.sum(tl.exp(x), axis=0)
    lse_val = max_val + tl.log(sum_exp)
    ln2 = 0.6931471805599453  # math.log(2.0)
    tl.store(lse_ptr + h, lse_val / ln2)


@triton.jit
def matmul_row_kernel(
    attn_ptr,   # *f32 [H, T]
    Kc_ptr,     # *f32 [T, Dq]
    out_ptr,    # *f32 [H, Dq]
    H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr
):
    h = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < Dq

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        attn_vec = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0)  # [BLOCK_T, BLOCK_D]
        acc += tl.sum(Kc_tile * attn_vec[:, None], axis=0)

    tl.store(out_ptr + h * Dq + offs_d, acc, mask=mask_d)


def get_inputs():
    # Ensure tensors are on CUDA for Triton execution
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)], dim=0).int()
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Expect CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton execution"

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dq = q_nope.shape[2]
        Dp = q_pe.shape[2]
        # Safety checks to match original assumptions
        assert H == 16, "num_qo_heads must be 16"
        assert Dq == 512, "head_dim_ckv must be 512"
        assert Dp == 64, "head_dim_kpe must be 64"

        # Output buffers (compute in fp32, return bfloat16 as in original)
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Tiling (power-of-two)
        BLOCK_T = 128
        BLOCK_D = 128

        for b in range(B):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp for this batch element
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).to(torch.long).contiguous()  # [L_tokens]
            Kc_b = ckv_cache[tok_idx].squeeze(1).to(torch.float32).contiguous()  # [L_tokens, 512]
            Kp_b = kpe_cache[tok_idx].squeeze(1).to(torch.float32).contiguous()  # [L_tokens, 64]

            # Prepare qn, qp for all heads
            qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Intermediate: logits [H, T]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)

            # Kernel 1: fused logits
            grid_logits = (H, triton.cdiv(L_tokens, BLOCK_T))
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
                num_warps=4, num_stages=2
            )

            # Scale
            logits_scaled = logits * sm_scale

            # Kernel 2: softmax per row
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                logits_scaled, attn,
                H=H, T=L_tokens,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Kernel 3: lse per row
            lse_b = torch.empty((H,), dtype=torch.float32, device=q_nope.device)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                logits_scaled, lse_b,
                H=H, T=L_tokens,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )
            lse[b] = lse_b

            # Kernel 4: out[h, :] = attn[h, :] @ Kc_b
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=q_nope.device)
            grid_mm = (H,)
            matmul_row_kernel[grid_mm](
                attn, Kc_b, out_b,
                H=H, T=L_tokens, Dq=Dq,
                BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
                num_warps=4, num_stages=2
            )
            output[b] = out_b

        # Match original output dtype
        output = output.to(torch.bfloat16)
        return output, lse.to(torch.float32)


def run(*args):
    return ModelNew()(*args)
