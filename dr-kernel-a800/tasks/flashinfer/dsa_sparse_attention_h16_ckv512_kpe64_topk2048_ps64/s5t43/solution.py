import torch
import math
import triton
import triton.language as tl


# Generic fp32 matmul: A[M, K] @ B[K, N] -> C[M, N]
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


# Kernel A: qn [H, Kq] x Kc_gather [M, Kq] -> logits_qn [H, M], Kq=512
@triton.jit
def matmul_qn_kc_kernel(qn_ptr, Kc_ptr, logits_qn_ptr,
                        H: tl.constexpr, M, Kq: tl.constexpr,
                        stride_qn0, stride_qn1,
                        stride_kc0, stride_kc1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    for k0 in range(0, Kq, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qn_ptr + (offs_h[:, None] * stride_qn0 + offs_k[None, :] * stride_qn1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kq), other=0.0)  # [BH, BK]
        kc_ptrs = Kc_ptr + (offs_m[None, :] * stride_kc0 + offs_k[:, None] * stride_kc1)    # [BM, BK]
        b = tl.load(kc_ptrs, mask=(offs_m[None, :] < M) & (offs_k[:, None] < Kq), other=0.0)
        acc += tl.dot(a, b)  # (BH, BM)

    out_ptrs = logits_qn_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Kernel B: qp [H, Kp] x Kp_gather [M, Kp] -> logits_qp [H, M], Kp=64
@triton.jit
def matmul_qp_kp_kernel(qp_ptr, Kp_ptr, logits_qp_ptr,
                        H: tl.constexpr, M, Kp: tl.constexpr,
                        stride_qp0, stride_qp1,
                        stride_kp0, stride_kp1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    for k0 in range(0, Kp, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qp_ptr + (offs_h[:, None] * stride_qp0 + offs_k[None, :] * stride_qp1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kp), other=0.0)  # [BH, BK]
        kp_ptrs = Kp_ptr + (offs_m[None, :] * stride_kp0 + offs_k[:, None] * stride_kp1)    # [BM, BK]
        b = tl.load(kp_ptrs, mask=(offs_m[None, :] < M) & (offs_k[:, None] < Kp), other=0.0)
        acc += tl.dot(a, b)

    out_ptrs = logits_qp_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Elementwise add and scale: C[H,M] = (A[H,M] + B[H,M]) * scale
@triton.jit
def add_scale_kernel(A_ptr, B_ptr, C_ptr,
                     H: tl.constexpr, M,
                     stride_a0, stride_a1,
                     stride_b0, stride_b1,
                     stride_c0, stride_c1,
                     scale: tl.float32):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * 16 + tl.arange(0, 16)
    offs_m = pid_m * 256 + tl.arange(0, 256)

    a_ptrs = A_ptr + (offs_h[:, None] * stride_a0 + offs_m[None, :] * stride_a1)
    b_ptrs = B_ptr + (offs_h[:, None] * stride_b0 + offs_m[None, :] * stride_b1)
    c_ptrs = C_ptr + (offs_h[:, None] * stride_c0 + offs_m[None, :] * stride_c1)

    A = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    B = tl.load(b_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    C = (A + B) * scale
    tl.store(c_ptrs, C, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Softmax per row: attn[H,M] = softmax(C[H,M], dim=1) (stable: subtract max), one program per row
@triton.jit
def softmax_row_kernel(C_ptr, attn_ptr,
                       H: tl.constexpr, M,
                       stride_c0, stride_c1,
                       stride_a0, stride_a1):
    h = tl.program_id(0)
    # Pass 1: row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Pass 2: row sum of exp(x - row_max)
    row_sum = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        row_sum += tl.sum(e, axis=0)

    inv_row_sum = 1.0 / row_sum

    # Pass 3: write normalized outputs
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        y = tl.exp(x - row_max) * inv_row_sum
        attn_ptrs = attn_ptr + (h * stride_a0 + offs_m * stride_a1)
        tl.store(attn_ptrs, y, mask=offs_m < M)


# LSE per row: lse[h] = logsumexp(C[h, :]) / ln(2), stable (one program per row)
@triton.jit
def lse_row_kernel(C_ptr, lse_ptr,
                   H: tl.constexpr, M,
                   stride_c0, stride_c1):
    h = tl.program_id(0)
    row_max = -float("inf")
    # Pass 1: row max
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Pass 2: sum of exp(x - row_max)
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        sum_exp += tl.sum(e, axis=0)

    lse_val = row_max + tl.log(sum_exp)  # natural log logsumexp
    # Divide by ln(2): 1 / 0.69314718056
    lse_val = lse_val / 0.6931471805599453
    tl.store(lse_ptr + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        device = q_nope.device

        # Input shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape  # [T, 16, 512]
        num_qo_heads_qpe, head_dim_kpe = q_pe.shape[1], q_pe.shape[2]  # [T, 16, 64]
        assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64

        # Flatten paged KV cache to token-level: [num_pages, 64, dim] -> [num_tokens*64, dim]
        total_tokens = ckv_cache.shape[0] * ckv_cache.shape[1]  # num_tokens * 64
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).contiguous().to(torch.float32)  # [num_tokens*64, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).contiguous().to(torch.float32)  # [num_tokens*64, 64]

        # Prepare output
        output = torch.empty(
            (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
        )
        lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Process each token
        for t in range(num_tokens):
            # Gather valid indices: indices != -1
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices != -1
            valid_indices = indices[valid_mask]  # [M]
            M = valid_indices.numel()

            if M == 0:
                # No valid keys, output zeros and lse -inf
                output[t].zero_()
                lse[t].fill_(-float("inf"))
                continue

            # Map tok_idx to flattened index (each group of 64 is a "page")
            tok_idx = valid_indices  # already in [0, num_tokens*64)
            # Gather Kc/Kp rows
            Kc_gather = Kc_all[tok_idx]  # [M, 512]
            Kp_gather = Kp_all[tok_idx]  # [M, 64]

            # Prepare q_nope[t], q_pe[t] as float32
            qn = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[t].to(torch.float32).contiguous()   # [16, 64]

            # Allocate intermediates
            logits_qn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            logits_qp = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            logits = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)

            # Launch matmul kernels (qn @ Kc_gather.T -> [H=16, M])
            # Choose conservative tiles to avoid SMEM limits
            BLOCK_H = 16
            BLOCK_M = 64
            BLOCK_K = 32
            grid_qn = (num_qo_heads // BLOCK_H, (M + BLOCK_M - 1) // BLOCK_M)
            matmul_qn_kc_kernel[grid_qn](
                qn, Kc_gather, logits_qn,
                H=16, M=M, Kq=512,
                stride_qn0=qn.stride(0), stride_qn1=qn.stride(1),
                stride_kc0=Kc_gather.stride(0), stride_kc1=Kc_gather.stride(1),
                stride_out0=logits_qn.stride(0), stride_out1=logits_qn.stride(1),
                BLOCK_H=BLOCK_H, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
                num_stages=2, num_warps=4
            )

            # (qp @ Kp_gather.T -> [H=16, M])
            grid_qp = (num_qo_heads // BLOCK_H, (M + BLOCK_M - 1) // BLOCK_M)
            matmul_qp_kp_kernel[grid_qp](
                qp, Kp_gather, logits_qp,
                H=16, M=M, Kp=64,
                stride_qp0=qp.stride(0), stride_qp1=qp.stride(1),
                stride_kp0=Kp_gather.stride(0), stride_kp1=Kp_gather.stride(1),
                stride_out0=logits_qp.stride(0), stride_out1=logits_qp.stride(1),
                BLOCK_H=BLOCK_H, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
                num_stages=2, num_warps=4
            )

            # Add and scale
            logits.mul_(0)  # clear
            logits = logits_qn + logits_qp  # [H, M]
            # Scale by sm_scale
            grid_add = (num_qo_heads // 16, (M + 256 - 1) // 256)
            add_scale_kernel[grid_add](
                logits_qn, logits_qp, logits,
                H=16, M=M,
                stride_a0=logits_qn.stride(0), stride_a1=logits_qn.stride(1),
                stride_b0=logits_qp.stride(0), stride_b1=logits_qp.stride(1),
                stride_c0=logits.stride(0), stride_c1=logits.stride(1),
                scale=float(sm_scale),
                num_warps=4
            )

            # Softmax per row on logits
            attn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            grid_softmax = (num_qo_heads,)  # one program per row
            softmax_row_kernel[grid_softmax](
                logits, attn,
                H=16, M=M,
                stride_c0=logits.stride(0), stride_c1=logits.stride(1),
                stride_a0=attn.stride(0), stride_a1=attn.stride(1),
                num_warps=4
            )

            # Final matmul: attn @ Kc_gather -> output[t]
            output_t = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            # Use a slightly larger M tile but keep K small to avoid SMEM issues
            grid_out = (num_qo_heads // 16, (M + 128 - 1) // 128)
            _matmul_fp32[grid_out](
                attn, Kc_gather,
                output_t,
                M=M, N=head_dim_ckv, K=M,
                stride_am=attn.stride(0), stride_ak=attn.stride(1),
                stride_bk=Kc_gather.stride(0), stride_bn=Kc_gather.stride(1),
                stride_cm=output_t.stride(0), stride_cn=output_t.stride(1),
                BLOCK_M=128, BLOCK_N=16, BLOCK_K=32,
                num_stages=2, num_warps=4
            )

            # Store output (bfloat16)
            output[t] = output_t.to(torch.bfloat16)

            # LSE per row: logsumexp(logits_scaled[h, :]) / ln(2)
            lse_t = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            grid_lse = (num_qo_heads,)
            lse_row_kernel[grid_lse](
                logits, lse_t,
                H=16, M=M,
                stride_c0=logits.stride(0), stride_c1=logits.stride(1),
                num_warps=4
            )
            lse[t] = lse_t

        return output, lse


def run(*args):
    return ModelNew()(*args)
