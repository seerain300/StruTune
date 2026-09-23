import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _matmul_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias, where A: (M, K), B: (K, N), C: (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BM, BK)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # (BK, BN)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_1d_kernel(
    A_ptr, B_ptr, C_ptr,
    NUMEL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NUMEL
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, convW_ptr, convBias_ptr, convOut_ptr,
    B, H, S,
    stride_bxm, stride_bxn,
    stride_convW_h, stride_convW_k,
    stride_convOut_bm, stride_convOut_bn, stride_convOut_bk,
):
    # One program per (b, h). Compute conv_out[b, h, s] for s in 0..S-1.
    b = tl.program_id(0)
    h = tl.program_id(1)

    # convW: (H, 4); kernel_size = 4 (fixed). convBias: (H).
    # causal: padding = 3 on the left (since kernel_size = 4).
    for s in range(0, S):
        acc = tl.zeros((), dtype=tl.float32)
        for k in range(0, 4):
            in_pos = s + 3 - k  # left padding
            valid = in_pos >= 0
            val = tl.load(Bx_ptr + b * stride_bxm + h * stride_bxn + in_pos * stride_bxn, mask=valid, other=0.0)
            w = tl.load(convW_ptr + h * stride_convW_h + k * stride_convW_k)
            acc += val * w
        bias = tl.load(convBias_ptr + h)
        acc += bias
        tl.store(convOut_ptr + b * stride_convOut_bm + h * stride_convOut_bk + s * stride_convOut_bn, acc)


@triton.jit
def _matmul_linear_outproj_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias, A: (M,K), B: (K,N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BM, BK)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # (BK, BN)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)  # (H, 1, 4) or (H, 4); we will use last 4 columns
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, S, H = x.shape
        K = H
        N1 = 3 * H

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        A = x.view(B * S, H)  # (M=BS, K=H)
        B_w = in_proj_weight.t().view(H, N1)  # (K=H, N=3H)
        Bias_in = in_proj_bias.view(N1)

        BCx = torch.empty((B * S, N1), dtype=torch.float32, device=x.device)

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_in = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _matmul_linear_kernel[grid_in](
            A, B_w, Bias_in, BCx,
            B * S, N1, K,
            A.stride(0), A.stride(1),
            B_w.stride(0), B_w.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        BCx = BCx.view(B, S, N1)  # (B, S, 3H)

        # Split into B, C, x_proj of shape (B, S, H)
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # 2) Element-wise gate: Bx = B * x_proj
        Bx_flat = torch.empty((B * S * H,), dtype=torch.float32, device=x.device)
        A1 = B_tensor.contiguous().view(B * S * H)
        A2 = x_proj.contiguous().view(B * S * H)
        BLOCK = 1024
        grid_mul = (triton.cdiv(B * S * H, BLOCK),)
        _elementwise_mul_1d_kernel[grid_mul](
            A1, A2, Bx_flat,
            B * S * H,
            BLOCK_SIZE=BLOCK,
        )
        Bx = Bx_flat.view(B, S, H)  # (B, S, H)

        # 3) Grouped causal conv: kernel_size=4, groups=H
        # Extract conv_weight from in_proj_weight's last 4 columns: (H, 4)
        convW = in_proj_weight[:, -4:].contiguous()  # (H, 4), float32
        convOut = torch.empty((B, H, S), dtype=torch.float32, device=x.device)

        grid_conv = (B, H)
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, conv_bias, convOut,
            B, H, S,
            Bx.stride(0), Bx.stride(2),
            convW.stride(0), convW.stride(1),
            convOut.stride(0), convOut.stride(2), convOut.stride(1),
        )

        # 4) Output gating: y = C * conv_out, align convOut to (B, S, H)
        convOut_T = convOut.transpose(-1, -2).contiguous()  # (B, S, H)
        y = torch.empty((B, S, H), dtype=torch.float32, device=x.device)

        y_flat = y.contiguous().view(B * S * H)
        A3 = C_tensor.contiguous().view(B * S * H)
        A4 = convOut_T.contiguous().view(B * S * H)
        grid_mul2 = (triton.cdiv(B * S * H, BLOCK),)
        _elementwise_mul_1d_kernel[grid_mul2](
            A3, A4, y_flat,
            B * S * H,
            BLOCK_SIZE=BLOCK,
        )
        y = y_flat.view(B, S, H)  # (B, S, H)

        # 5) Final out-proj: y @ out_proj_weight^T + out_proj_bias
        y_flat2 = y.contiguous().view(B * S * H)
        B_w_out = out_proj_weight.t().contiguous().view(H, H)  # (K=H, N=H)
        Bias_out = out_proj_bias.view(H)

        output_flat = torch.empty((B * S * H,), dtype=torch.float32, device=x.device)

        BLOCK_M2 = 128
        BLOCK_N2 = 64
        BLOCK_K2 = 32
        grid_out = (triton.cdiv(B * S, BLOCK_M2), triton.cdiv(H, BLOCK_N2))
        _matmul_linear_outproj_kernel[grid_out](
            y_flat2.view(B * S, H), B_w_out, Bias_out, output_flat,
            B * S, H, H,
            y_flat2.view(B * S, H).stride(0), y_flat2.view(B * S, H).stride(1),
            B_w_out.stride(0), B_w_out.stride(1),
            output_flat.view(B * S, H).stride(0), output_flat.view(B * S, H).stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
        )

        output = output_flat.view(B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
