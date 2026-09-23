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


# Kernel A: qn [H, Kq] x Kc_gather [M, Kq] -> logits_qn [H, M]
# Kq=512
@triton.jit
def matmul_qn_kc_kernel(qn_ptr, Kc_ptr, logits_qn_ptr,
                        H: tl.constexpr, M,
                        stride_qn0, stride_qn1,
                        stride_kc0, stride_kc1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

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


# Kernel B: qp [H, Kp] x Kp_gather [M, Kp] -> logits_qp [H, M]
# Kp=64
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

    a = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    c = (a + b) * scale
    tl.store(c_ptrs, c, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Per-row stable softmax: X[H,M] -> OUT[H,M]
@triton.jit
def softmax_row_kernel(X_ptr, Out_ptr,
                       H: tl.constexpr, M,
                       stride_x0, stride_x1,
                       stride_out0, stride_out1):
    # One program per row (h), iterate over M in chunks
    h = tl.program_id(0)

    # Pass 1: row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)

    # Pass 2: row sum of exp(x - row_max)
    row_sum = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        row_sum += tl.sum(e, axis=0)

    inv_row_sum = 1.0 / row_sum

    # Pass 3: write normalized outputs
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        out_ptrs = Out_ptr + (h * stride_out0 + offs_m * stride_out1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        y = tl.exp(x - row_max) * inv_row_sum
        tl.store(out_ptrs, y, mask=offs_m < M)


# Per-row logsumexp of X[H,M] / ln(2): OUT[H]
@triton.jit
def lse_row_kernel(X_ptr, Out_ptr,
                   H: tl.constexpr, M,
                   stride_x0, stride_x1,
                   stride_out0,
                   inv_ln2: tl.float32):
    # One program per row (h)
    h = tl.program_id(0)

    # Pass 1: row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)

    # Pass 2: sum_exp
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        sum_exp += tl.sum(e, axis=0)

    lse_val = row_max + tl.log(sum_exp)  # natural log
    tl.store(Out_ptr + h * stride_out0, lse_val * inv_ln2)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        device = q_nope.device
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        head_dim_kpe = q_pe.shape[-1]
        assert head_dim_kpe == 64

        # Flatten caches and convert to fp32
        num_pages, seq_len, _ = ckv_cache.shape
        assert seq_len == 64, "kpe_cache seq_len must be 64"
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [num_pages*64, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [num_pages*64, 64]

        # Prepare outputs
        output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        inv_ln2 = 1.0 / math.log(2.0)  # for logsumexp scaling

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices != -1
            valid_indices = indices[valid_mask]
            M = valid_indices.numel()
            if M == 0:
                # Nothing to do; output[t] already zero, lse[t] stays -inf
                continue

            # Gather Kc and Kp
            tok_idx = valid_indices.to(torch.long)  # [M]
            Kc_gather = Kc_all[tok_idx]  # [M, 512]
            Kp_gather = Kp_all[tok_idx]  # [M, 64]

            # Prepare qn and qp as fp32 contiguous
            qn = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[t].to(torch.float32).contiguous()   # [16, 64]

            # Allocate intermediates
            logits_qn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            logits_qp = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            C = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            attn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)

            # Launch matmul for logits_qn: qn @ Kc_gather.T
            H = num_qo_heads
            BLOCK_H = 16
            BLOCK_M = 256
            BLOCK_K = 64
            grid_qn = (triton.cdiv(H, BLOCK_H), triton.cdiv(M, BLOCK_M))
            matmul_qn_kc_kernel[grid_qn](
                qn, Kc_gather, logits_qn,
                H, M,
                qn.stride(0), qn.stride(1),
                Kc_gather.stride(0), Kc_gather.stride(1),
                logits_qn.stride(0), logits_qn.stride(1),
                BLOCK_H, BLOCK_M, BLOCK_K
            )

            # Launch matmul for logits_qp: qp @ Kp_gather.T
            logits_qp = torch.empty((H, M), dtype=torch.float32, device=device)  # to avoid reuse bug
            grid_qp = (triton.cdiv(H, BLOCK_H), triton.cdiv(M, BLOCK_M))
            matmul_qp_kp_kernel[grid_qp](
                qp, Kp_gather, logits_qp,
                H, M,
                qp.stride(0), qp.stride(1),
                Kp_gather.stride(0), Kp_gather.stride(1),
                logits_qp.stride(0), logits_qp.stride(1),
                BLOCK_H, BLOCK_M, BLOCK_K
            )

            # Launch add_scale_kernel: C = (logits_qn + logits_qp) * sm_scale
            grid_add = (triton.cdiv(H, 16), triton.cdiv(M, 256))
            add_scale_kernel[grid_add](
                logits_qn, logits_qp, C,
                H, M,
                logits_qn.stride(0), logits_qn.stride(1),
                logits_qp.stride(0), logits_qp.stride(1),
                C.stride(0), C.stride(1),
                float(sm_scale)
            )

            # Launch softmax_row_kernel to get attn
            attn = torch.empty((H, M), dtype=torch.float32, device=device)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                C, attn,
                H, M,
                C.stride(0), C.stride(1),
                attn.stride(0), attn.stride(1)
            )

            # Compute output[t] = attn @ Kc_gather
            out_row = torch.empty((H, head_dim_ckv), dtype=torch.float32, device=device)
            BLOCK_M_out = 128
            BLOCK_K_out = 64
            grid_out = (triton.cdiv(H, BLOCK_M_out), triton.cdiv(head_dim_ckv, BLOCK_M_out))
            # Here K = M, N = head_dim_ckv
            _matmul_fp32[grid_out](
                attn, Kc_gather, out_row,
                M, head_dim_ckv, M,
                attn.stride(0), attn.stride(1),
                Kc_gather.stride(0), Kc_gather.stride(1),
                out_row.stride(0), out_row.stride(1),
                BLOCK_M_out, BLOCK_K_out, BLOCK_K_out
            )
            output[t] = out_row.to(torch.bfloat16)  # write one row at a time

            # Compute lse[t, :]
            lse_row = torch.empty((H,), dtype=torch.float32, device=device)
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                C, lse_row,
                H, M,
                C.stride(0), C.stride(1),
                lse_row.stride(0),
                inv_ln2
            )
            lse[t] = lse_row  # already per-row

        return output, lse


def run(*args):
    return ModelNew()(*args)
