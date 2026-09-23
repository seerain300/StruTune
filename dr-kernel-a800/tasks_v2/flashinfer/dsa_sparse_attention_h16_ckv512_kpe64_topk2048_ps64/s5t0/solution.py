import torch
import math
import triton
import triton.language as tl


# Triton matmul kernel: A: [M, K], B: [K, N], C: [M, N]
# We specialize for fp32 and small H (num_qo_heads), arbitrary M, N, K.
# Each program computes a tile of (BLOCK_M x BLOCK_N) and reduces over K in BLOCK_K chunks.
@triton.jit
def _matmul_fp32(A_ptr, B_ptr, C_ptr,
                 M, N, K,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 stride_cm, stride_cn,
                 sm_scale,  # scalar
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)  # along M
    pid_n = tl.program_id(1)  # along N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    # store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel A: qn [H, Kq] x Kc [M, Kq] -> logits_qn [H, M]
# We fix H=16, Kq=512. M is runtime.
@triton.jit
def matmul_qn_kc_kernel(qn_ptr, Kc_ptr, logits_qn_ptr,
                        H, M,
                        stride_qn0, stride_qn1,
                        stride_kc0, stride_kc1,
                        stride_out0, stride_out1,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # N dimension is M, H is passed as runtime, we tile over H and M
    pid_h = tl.program_id(0)  # along H
    pid_m = tl.program_id(1)  # along M

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # BLOCK_H is a constexpr; we set it to H for simplicity
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # We'll just use a single tile for H (BLOCK_H=16) and loop over M in tiles
    # Initialize accumulator for BLOCK_M across H
    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    Kq = 512  # constant from the problem
    for k0 in range(0, Kq, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # qn: [H, Kq], row-major
        q_ptrs = qn_ptr + (offs_h[:, None] * stride_qn0 + offs_k[None, :] * stride_qn1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kq), other=0.0)

        # Kc: [M, Kq], row-major
        kc_ptrs = Kc_ptr + (offs_m[None, :] * stride_kc0 + offs_k[:, None] * stride_kc1)
        b = tl.load(kc_ptrs, mask=(offs_k[:, None] < Kq) & (offs_m[None, :] < M), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)  # (BLOCK_H, BLOCK_K) @ (BLOCK_K, BLOCK_M) -> (BLOCK_H, BLOCK_M)

    # Store into logits_qn [H, M]
    out_ptrs = logits_qn_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Kernel B: qp [H, Kp] x Kp [M, Kp] -> logits_qp [H, M]
# H=16, Kp=64, M runtime.
@triton.jit
def matmul_qp_kp_kernel(qp_ptr, Kp_ptr, logits_qp_ptr,
                        H, M,
                        stride_qp0, stride_qp1,
                        stride_kp0, stride_kp1,
                        stride_out0, stride_out1,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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


# Softmax kernel: input logits [H, M], output attn [H, M] = softmax(input * sm_scale) along dim=1
# We implement stable softmax: subtract max, exp, sum, divide.
@triton.jit
def softmax_kernel(logits_ptr, attn_ptr,
                   H, M,
                   stride_in0, stride_in1,
                   stride_out0, stride_out1,
                   sm_scale,  # scalar
                   BLOCK_M: tl.constexpr):
    pid_h = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    row_ptr = logits_ptr + pid_h * stride_in0
    # Load row as vector
    logits_row = tl.load(row_ptr + offs_m * stride_in1, mask=offs_m < M, other=-float("inf"))
    # Scale
    logits_scaled = logits_row * sm_scale
    # Stable softmax
    m = tl.max(logits_scaled, axis=0)
    logits_scaled = logits_scaled - m
    exp_logits = tl.exp(logits_scaled)
    sum_exp = tl.sum(exp_logits, axis=0)
    attn_row = exp_logits / sum_exp
    # Store
    out_row_ptr = attn_ptr + pid_h * stride_out0
    tl.store(out_row_ptr + offs_m * stride_out1, attn_row, mask=offs_m < M)


# Output matmul: attn [H, M] x Kc [M, 512] -> out [H, 512]
@triton.jit
def matmul_attn_kc_kernel(attn_ptr, Kc_ptr, out_ptr,
                          H, M,
                          stride_attn0, stride_attn1,
                          stride_kc0, stride_kc1,
                          stride_out0, stride_out1,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)

    Kq = 512
    for k0 in range(0, Kq, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = attn_ptr + (offs_h[:, None] * stride_attn0 + offs_k[None, :] * stride_attn1)
        a = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kq), other=0.0)

        b_ptrs = Kc_ptr + (offs_k[:, None] * stride_kc0 + offs_n[None, :] * stride_kc1)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < Kq) & (offs_n[None, :] < 512), other=0.0)

        acc += tl.dot(a, b)

    out_ptrs = out_ptr + (offs_h[:, None] * stride_out0 + offs_n[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_n[None, :] < 512))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        q_nope: [num_tokens, 16, 512], bfloat16
        q_pe: [num_tokens, 16, 64], bfloat16
        ckv_cache: [num_pages, 64, 512], bfloat16
        kpe_cache: [num_pages, 64, 64], bfloat16
        sparse_indices: [num_tokens, 2048], int32
        sm_scale: float32 scalar
        Returns:
        output: [num_tokens, 16, 512], bfloat16
        lse: [num_tokens, 16], float32 (logsumexp in base 2)
        """
        device = q_nope.device

        # Flatten paged caches to [total_kv_tokens, dim]
        total_pages = ckv_cache.shape[0]
        total_tokens = total_pages * 64  # since each "token" is a 64-length vector per page
        Kc_all = ckv_cache.reshape(-1, 512).to(torch.float32).contiguous()  # [total_tokens, 512]
        Kp_all = kpe_cache.reshape(-1, 64).to(torch.float32).contiguous()  # [total_tokens, 64]

        num_tokens = q_nope.shape[0]
        H = 16
        Kq = 512
        Kp = 64
        topk = sparse_indices.shape[-1]  # 2048
        assert q_nope.shape[1] == H and q_nope.shape[2] == Kq
        assert q_pe.shape[1] == H and q_pe.shape[2] == Kp
        assert ckv_cache.shape[1] == 64 and ckv_cache.shape[2] == Kq
        assert kpe_cache.shape[1] == 64 and kpe_cache.shape[2] == Kp

        # Output buffers (fp32 for compute, bf16 for return)
        output = torch.empty((num_tokens, H, 512), dtype=torch.float32, device=device)
        lse = torch.full((num_tokens, H), -float("inf"), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [topk], int32
            valid_mask = indices != -1
            if not valid_mask.any():
                # No valid entries: output zeros, lse -inf
                output[t] = 0.0
                lse[t] = -float("inf")
                continue

            valid_indices = indices[valid_mask].to(torch.long)  # [M]
            M = valid_indices.numel()

            # Gather rows from flattened caches
            Kc_gather = Kc_all[valid_indices]  # [M, 512], fp32
            Kp_gather = Kp_all[valid_indices]  # [M, 64], fp32

            # Prepare queries (fp32)
            qn = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[t].to(torch.float32).contiguous()   # [16, 64]

            # Allocate intermediate results
            logits_qn = torch.empty((H, M), dtype=torch.float32, device=device)
            logits_qp = torch.empty((H, M), dtype=torch.float32, device=device)
            attn = torch.empty((H, M), dtype=torch.float32, device=device)

            # Launch matmul for qn @ Kc
            grid_qn = (triton.cdiv(H, 16), triton.cdiv(M, 64))
            matmul_qn_kc_kernel[grid_qn](
                qn, Kc_gather,
                logits_qn,
                H, M,
                qn.stride(0), qn.stride(1),
                Kc_gather.stride(0), Kc_gather.stride(1),
                logits_qn.stride(0), logits_qn.stride(1),
                BLOCK_H=16, BLOCK_M=64, BLOCK_K=32,
            )

            # Launch matmul for qp @ Kp
            grid_qp = (triton.cdiv(H, 16), triton.cdiv(M, 64))
            matmul_qp_kp_kernel[grid_qp](
                qp, Kp_gather,
                logits_qp,
                H, M,
                qp.stride(0), qp.stride(1),
                Kp_gather.stride(0), Kp_gather.stride(1),
                logits_qp.stride(0), logits_qp.stride(1),
                BLOCK_H=16, BLOCK_M=64, BLOCK_K=32,
            )

            # Compute logits = logits_qn + logits_qp
            logits = logits_qn + logits_qp  # [H, M]
            logits = logits * sm_scale

            # Softmax along dim=1
            grid_softmax = (H,)
            softmax_kernel[grid_softmax](
                logits,
                attn,
                H, M,
                logits.stride(0), logits.stride(1),
                attn.stride(0), attn.stride(1),
                sm_scale,
                BLOCK_M=64,
            )

            # Output: attn @ Kc
            out = torch.empty((H, 512), dtype=torch.float32, device=device)
            grid_out = (triton.cdiv(H, 16), triton.cdiv(512, 128))
            matmul_attn_kc_kernel[grid_out](
                attn, Kc_gather, out,
                H, M,
                attn.stride(0), attn.stride(1),
                Kc_gather.stride(0), Kc_gather.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_H=16, BLOCK_N=128, BLOCK_K=32,
            )

            # Store output
            output[t] = out

            # Compute LSE = logsumexp(logits_scaled, dim=1) / ln(2)
            # Use PyTorch for this reduction; very small H=16
            lse_row = torch.logsumexp(logits, dim=1) / math.log(2.0)  # [H]
            lse[t] = lse_row

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
