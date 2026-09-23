import torch
import math
import triton
import triton.language as tl


# Matmul A[M,K] @ B[K,N] -> C[M,N], fp32
@triton.jit
def _matmul_fp32(A_ptr, B_ptr, C_ptr,
                 M, N, K,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel A: qn [H, 512] x Kc_gather [M, 512] -> logits_qn [H, M]
@triton.jit
def matmul_qn_kc_kernel(qn_ptr, Kc_ptr, logits_qn_ptr,
                        H: tl.constexpr, M,
                        stride_qn0, stride_qn1,
                        stride_kc0, stride_kc1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)  # tile along H
    pid_m = tl.program_id(1)  # tile along M

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    Kq = 512
    for k0 in range(0, Kq, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qn_ptr + (offs_h[:, None] * stride_qn0 + offs_k[None, :] * stride_qn1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kq), other=0.0)
        kc_ptrs = Kc_ptr + (offs_m[None, :] * stride_kc0 + offs_k[:, None] * stride_kc1)
        b = tl.load(kc_ptrs, mask=(offs_k[:, None] < Kq) & (offs_m[None, :] < M), other=0.0)
        acc += tl.dot(a, b)  # (BLOCK_H, BLOCK_K) @ (BLOCK_K, BLOCK_M) -> (BLOCK_H, BLOCK_M)

    out_ptrs = logits_qn_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Kernel B: qp [H, 64] x Kp_gather [M, 64] -> logits_qp [H, M]
@triton.jit
def matmul_qp_kp_kernel(qp_ptr, Kp_ptr, logits_qp_ptr,
                        H: tl.constexpr, M,
                        stride_qp0, stride_qp1,
                        stride_kp0, stride_kp1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    Kp = 64
    for k0 in range(0, Kp, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qp_ptr + (offs_h[:, None] * stride_qp0 + offs_k[None, :] * stride_qp1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kp), other=0.0)
        kp_ptrs = Kp_ptr + (offs_m[None, :] * stride_kp0 + offs_k[:, None] * stride_kp1)
        b = tl.load(kp_ptrs, mask=(offs_k[:, None] < Kp) & (offs_m[None, :] < M), other=0.0)
        acc += tl.dot(a, b)

    out_ptrs = logits_qp_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Elementwise add and scale: C[H,M] = A[H,M] + B[H,M]; C *= sm_scale
@triton.jit
def add_scale_kernel(A_ptr, B_ptr, C_ptr,
                     H: tl.constexpr, M,
                     stride_a0, stride_a1,
                     stride_b0, stride_b1,
                     stride_c0, stride_c1,
                     scale: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * 16 + tl.arange(0, 16)  # since H=16, tile along H
    offs_m = pid_m * 256 + tl.arange(0, 256)  # tile along M

    a_ptrs = A_ptr + (offs_h[:, None] * stride_a0 + offs_m[None, :] * stride_a1)
    b_ptrs = B_ptr + (offs_h[:, None] * stride_b0 + offs_m[None, :] * stride_b1)
    c_ptrs = C_ptr + (offs_h[:, None] * stride_c0 + offs_m[None, :] * stride_c1)

    a = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    c = a + b
    c = c * scale
    tl.store(c_ptrs, c, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Softmax per row along M for [H, M] input; output [H, M]
# We implement one program per row; loop over M in tiles.
@triton.jit
def softmax_row_kernel(in_ptr, out_ptr,
                       H: tl.constexpr, M,
                       stride_in0, stride_in1,
                       stride_out0, stride_out1):
    h = tl.program_id(0)
    # First pass: compute row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        in_ptrs = in_ptr + (h * stride_in0 + offs_m * stride_in1)
        vals = tl.load(in_ptrs, mask=(offs_m < M), other=-float("inf"))
        tile_max = tl.max(vals, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    # Second pass: compute sum of exp(x - row_max)
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        in_ptrs = in_ptr + (h * stride_in0 + offs_m * stride_in1)
        vals = tl.load(in_ptrs, mask=(offs_m < M), other=0.0)
        e = tl.exp(vals - row_max)
        sum_exp += tl.sum(e, axis=0)

    # Third pass: write normalized softmax
    inv_sum = 1.0 / sum_exp
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        in_ptrs = in_ptr + (h * stride_in0 + offs_m * stride_in1)
        out_ptrs = out_ptr + (h * stride_out0 + offs_m * stride_out1)
        vals = tl.load(in_ptrs, mask=(offs_m < M), other=0.0)
        e = tl.exp(vals - row_max) * inv_sum
        tl.store(out_ptrs, e, mask=(offs_m < M))


# Matmul attn [H, M] x Kc_gather [M, 512] -> out [H, 512]
@triton.jit
def matmul_attn_kc_kernel(attn_ptr, Kc_ptr, out_ptr,
                          H: tl.constexpr, M, N: tl.constexpr,
                          stride_attn0, stride_attn1,
                          stride_kc0, stride_kc1,
                          stride_out0, stride_out1,
                          BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        attn_ptrs = attn_ptr + (offs_h[:, None] * stride_attn0 + offs_k[None, :] * stride_attn1)
        a = tl.load(attn_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < M), other=0.0)

        kc_ptrs = Kc_ptr + (offs_k[:, None] * stride_kc0 + offs_n[None, :] * stride_kc1)
        b = tl.load(kc_ptrs, mask=(offs_k[:, None] < M) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)  # (BLOCK_H, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_H, BLOCK_N)

    out_ptrs = out_ptr + (offs_h[:, None] * stride_out0 + offs_n[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_n[None, :] < N))


# LSE per row: lse[h] = log(sum(exp(x[h, :]))) / ln(2)
# Implement one program per row; loop over M in tiles to compute sum(exp(x)).
@triton.jit
def lse_row_kernel(in_ptr, lse_ptr,
                   H: tl.constexpr, M,
                   stride_in0, stride_in1):
    h = tl.program_id(0)
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        in_ptrs = in_ptr + (h * stride_in0 + offs_m * stride_in1)
        vals = tl.load(in_ptrs, mask=(offs_m < M), other=0.0)
        sum_exp += tl.sum(tl.exp(vals), axis=0)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure device consistency
        device = q_nope.device

        # Flatten paged caches to token-level and cast to float32
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        num_pages, page_size, _ = ckv_cache.shape
        topk = sparse_indices.shape[-1]

        # Sanity checks (not hard assertions in model, but ensure shapes)
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert page_size == 64
        assert topk == 2048

        # Flatten caches
        Kc_all = ckv_cache.reshape(num_pages * page_size, head_dim_ckv).to(torch.float32).contiguous()
        Kp_all = kpe_cache.reshape(num_pages * page_size, head_dim_kpe).to(torch.float32).contiguous()

        # Allocate outputs
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute inv ln(2) for base-2 logsumexp
        INV_LN2 = 1.4426950408889634

        for t in range(num_tokens):
            indices = sparse_indices[t].to(torch.int32)  # [topk]
            valid_mask = indices != -1
            tok_idx = indices[valid_mask].to(torch.long)  # [M]

            if tok_idx.numel() == 0:
                # output[t] zero, lse[t] -inf
                output[t].zero_()
                lse[t] = float("-inf")
                continue

            # Gather Kc_gather and Kp_gather: [M, 512] and [M, 64]
            Kc_gather = Kc_all[tok_idx]  # [M, 512]
            Kp_gather = Kp_all[tok_idx]  # [M, 64]

            # Prepare queries
            qn = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[t].to(torch.float32).contiguous()    # [16, 64]

            # Launch matmul_qn_kc: [H, 512] x [M, 512] -> [H, M]
            logits_qn = torch.empty((num_qo_heads, tok_idx.numel()), dtype=torch.float32, device=device)
            grid_qn = (triton.cdiv(num_qo_heads, 16), triton.cdiv(tok_idx.numel(), 256))
            matmul_qn_kc_kernel[grid_qn](
                qn, Kc_gather, logits_qn,
                num_qo_heads, tok_idx.numel(),
                qn.stride(0), qn.stride(1),
                Kc_gather.stride(0), Kc_gather.stride(1),
                logits_qn.stride(0), logits_qn.stride(1),
                BLOCK_H=16, BLOCK_M=256, BLOCK_K=32,
            )

            # Launch matmul_qp_kp: [H, 64] x [M, 64] -> [H, M]
            logits_qp = torch.empty((num_qo_heads, tok_idx.numel()), dtype=torch.float32, device=device)
            grid_qp = (triton.cdiv(num_qo_heads, 16), triton.cdiv(tok_idx.numel(), 256))
            matmul_qp_kp_kernel[grid_qp](
                qp, Kp_gather, logits_qp,
                num_qo_heads, tok_idx.numel(),
                qp.stride(0), qp.stride(1),
                Kp_gather.stride(0), Kp_gather.stride(1),
                logits_qp.stride(0), logits_qp.stride(1),
                BLOCK_H=16, BLOCK_M=256, BLOCK_K=64,
            )

            # Elementwise add and scale
            logits = torch.empty((num_qo_heads, tok_idx.numel()), dtype=torch.float32, device=device)
            grid_add = (triton.cdiv(num_qo_heads, 16), triton.cdiv(tok_idx.numel(), 256))
            add_scale_kernel[grid_add](
                logits_qn, logits_qp, logits,
                num_qo_heads, tok_idx.numel(),
                logits_qn.stride(0), logits_qn.stride(1),
                logits_qp.stride(0), logits_qp.stride(1),
                logits.stride(0), logits.stride(1),
                scale=sm_scale,
            )

            # Softmax per row (stable) along dim=1
            attn = torch.empty((num_qo_heads, tok_idx.numel()), dtype=torch.float32, device=device)
            grid_softmax = (num_qo_heads,)
            softmax_row_kernel[grid_softmax](
                logits, attn,
                num_qo_heads, tok_idx.numel(),
                logits.stride(0), logits.stride(1),
                attn.stride(0), attn.stride(1),
            )

            # Output matmul: attn [H, M] x Kc_gather [M, 512] -> [H, 512]
            out_tile = torch.empty((num_qo_heads, 512), dtype=torch.float32, device=device)
            grid_out = (triton.cdiv(num_qo_heads, 16), triton.cdiv(512, 128))
            matmul_attn_kc_kernel[grid_out](
                attn, Kc_gather, out_tile,
                num_qo_heads, tok_idx.numel(), 512,
                attn.stride(0), attn.stride(1),
                Kc_gather.stride(0), Kc_gather.stride(1),
                out_tile.stride(0), out_tile.stride(1),
                BLOCK_H=16, BLOCK_N=128, BLOCK_K=32,
            )
            output[t] = out_tile

            # Compute LSE per row (base-2): logsumexp / ln(2)
            lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            grid_lse = (num_qo_heads,)
            lse_row_kernel[grid_lse](
                logits, lse_row,
                num_qo_heads, tok_idx.numel(),
                logits.stride(0), logits.stride(1),
            )
            lse[t] = lse_row

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
