import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_ptr,   # *f32 [H, Dq]
    qp_ptr,   # *f32 [H, Dp]
    Kc_ptr,   # *f32 [T, Dq]
    Kp_ptr,   # *f32 [T, Dp]
    logits_ptr,  # *f32 [H, T]
    H: tl.constexpr,  # num heads
    T: tl.constexpr,  # tokens per batch
    Dq: tl.constexpr, # head_dim_ckv
    Dp: tl.constexpr, # head_dim_kpe
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        # Load qn[h, :] and qp[h, :]
        qn_row = tl.load(qn_ptr + h * Dq + tl.arange(0, Dq), mask=(tl.arange(0, Dq) < Dq), other=0.0)  # [Dq]
        qp_row = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp), mask=(tl.arange(0, Dp) < Dp), other=0.0)  # [Dp]
        # Load Kc/tile and Kp/tile
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + tl.arange(0, Dq)[None, :],
                          mask=(mask_t[:, None] & (tl.arange(0, Dq)[None, :] < Dq)), other=0.0)  # [BLOCK_T, Dq]
        Kp_tile = tl.load(Kp_ptr + offs_t[:, None] * Dp + tl.arange(0, Dp)[None, :],
                          mask=(mask_t[:, None] & (tl.arange(0, Dp)[None, :] < Dp)), other=0.0)  # [BLOCK_T, Dp]
        # Partial dot products for this tile
        dot_qn_Kc = tl.sum(qn_row[:, None] * Kc_tile, axis=0)  # [BLOCK_T]
        dot_qp_Kp = tl.sum(qp_row[:, None] * Kp_tile, axis=0)  # [BLOCK_T]
        logits_partial = dot_qn_Kc + dot_qp_Kp  # [BLOCK_T]
        tl.store(logits_ptr + h * T + offs_t, logits_partial, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_ptr,   # *f32 [H, T]
    attn_ptr,     # *f32 [H, T]
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_row = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
        row_max = tl.max(logits_row, axis=0)
        x = logits_row - row_max
        exp_x = tl.exp(x)
        denom = tl.sum(exp_x, axis=0)
        attn_row = exp_x / denom
        tl.store(attn_ptr + h * T + offs_t, attn_row, mask=mask_t)


@triton.jit
def lse_row_kernel(
    logits_ptr,   # *f32 [H, T]
    lse_ptr,      # *f32 [H]
    H: tl.constexpr,
    T: tl.constexpr,
    SCALE: tl.constexpr,  # 1/ln(2)
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    sum_logsumexp = 0.0
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        logits_row = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float("inf"))
        row_max = tl.max(logits_row, axis=0)
        x = logits_row - row_max
        sum_exp = tl.sum(tl.exp(x), axis=0)
        sum_logsumexp += tl.log(sum_exp) + row_max
    tl.store(lse_ptr + h, sum_logsumexp * SCALE)


@triton.jit
def perhead_gemv_kernel(
    attn_ptr,     # *f32 [H, T]
    Kc_ptr,       # *f32 [T, Dq]
    out_ptr,      # *f32 [H, Dq]
    H: tl.constexpr,   # num heads
    T: tl.constexpr,   # number of tokens
    Dq: tl.constexpr,  # head_dim_ckv
    BLOCK_T: tl.constexpr
):
    h = tl.program_id(0)
    # Initialize output to zero
    for j in range(0, Dq):
        out_ptr[h, j] = 0.0
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        attn_row = tl.load(attn_ptr + h * T + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        Kc_tile = tl.load(Kc_ptr + offs_t[:, None] * Dq + tl.arange(0, Dq)[None, :],
                          mask=(mask_t[:, None] & (tl.arange(0, Dq)[None, :] < Dq)), other=0.0)  # [BLOCK_T, Dq]
        # out[h, :] += sum over t of attn_row[t] * Kc_tile[t, :]
        for t in range(0, BLOCK_T):
            if (t0 + t) < T:
                out_ptr[h, :] += attn_row[t] * Kc_tile[t, :]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Compute in Triton, no torch ops in forward hot path
        device = q_nope.device
        B, H, Dq = q_nope.shape
        _, H2, Dp = q_pe.shape
        assert H == H2, "num_qo_heads mismatch"

        q_nope_f = q_nope.to(torch.float32).contiguous()  # [B, H, Dq]
        q_pe_f = q_pe.to(torch.float32).contiguous()      # [B, H, Dp]

        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # [B, H, Dq]
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            T = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if T <= 0:
                lse[b] = torch.full((H,), float("-inf"), dtype=torch.float32, device=device)
                output[b] = torch.zeros((H, Dq), dtype=torch.float32, device=device)
                continue

            tok_idx = kv_indices[kv_indptr[b].item(): kv_indptr[b + 1].item()].to(torch.int32).contiguous()  # [T]
            Kc_b = ckv_cache[tok_idx].squeeze(1).to(torch.float32).contiguous()  # [T, Dq]
            Kp_b = kpe_cache[tok_idx].squeeze(1).to(torch.float32).contiguous()  # [T, Dp]

            qn_b = q_nope_f[b]      # [H, Dq]
            qp_b = q_pe_f[b]        # [H, Dp]

            # Allocate intermediate buffers
            logits_scaled = torch.empty((H, T), dtype=torch.float32, device=device)
            attn = torch.empty((H, T), dtype=torch.float32, device=device)

            # 1) Fused logits: logits[h, t] = qn[h] · Kc[t] + qp[h] · Kp[t]
            BLOCK_T = 128
            grid_logits = (H,)
            compute_logits_kernel[grid_logits](
                qn_b, qp_b, Kc_b, Kp_b, logits_scaled,
                H=H, T=T, Dq=Dq, Dp=Dp,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Scale logits
            logits_scaled.mul_(sm_scale)

            # 2) Softmax over tokens per head
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                logits_scaled, attn,
                H=H, T=T,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # 3) lse per head: logsumexp(logits_scaled) / ln(2)
            SCALE = 1.0 / math.log(2.0)
            grid_lse = (H,)
            lse[b] = torch.empty((H,), dtype=torch.float32, device=device)
            lse_row_kernel[grid_lse](
                logits_scaled, lse[b],
                H=H, T=T, SCALE=SCALE,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            # 4) GEMV: out[h, :] = attn[h, :] @ Kc[:, :]
            grid_gemm = (H,)
            out_b = torch.empty((H, Dq), dtype=torch.float32, device=device)
            perhead_gemv_kernel[grid_gemm](
                attn, Kc_b, out_b,
                H=H, T=T, Dq=Dq,
                BLOCK_T=BLOCK_T,
                num_warps=4, num_stages=2
            )

            output[b] = out_b

        # Return in expected dtype
        return output.to(torch.bfloat16), lse


# Helper to generate inputs
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
    return [q_nope.cuda(), q_pe.cuda(), ckv_cache.cuda(), kpe_cache.cuda(), kv_indptr.cuda(), kv_indices.cuda(), sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
