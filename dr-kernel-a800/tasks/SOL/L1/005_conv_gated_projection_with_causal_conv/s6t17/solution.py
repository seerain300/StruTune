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
    # Compute C = A @ B + Bias, where A: (M,K), B: (K,N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
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
def _elementwise_mul_1d_kernel(A_ptr, B_ptr, C_ptr, SIZE, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < SIZE
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr,      # input Bx_padded: (B, H, S)
    W_ptr,      # conv weight: (H, 4)
    Bias_ptr,   # bias: (H)
    C_ptr,      # output: (B, H, S)
    B, H, S,
    stride_xb, stride_xh, stride_xs,
    stride_wh, stride_wk,
    stride_cb, stride_ch, stride_cs,
    BLOCK_S: tl.constexpr,
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Tile over sequence positions
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # k in {0,1,2,3}
        for k in range(0, 4):
            in_pos = offs_s - (k + 1)  # causal left pad: s - k - 1
            valid = in_pos >= 0
            x_ptrs = X_ptr + b * stride_xb + h * stride_xh + in_pos * stride_xs
            x_vals = tl.load(x_ptrs, mask=mask_s & valid, other=0.0)
            w_ptr = W_ptr + h * stride_wh + k * stride_wk
            w_val = tl.load(w_ptr)  # scalar
            acc += x_vals * w_val

        bias_val = tl.load(Bias_ptr + h)  # scalar
        acc += bias_val

        c_ptrs = C_ptr + b * stride_cb + h * stride_ch + offs_s * stride_cs
        tl.store(c_ptrs, acc, mask=mask_s)


@triton.jit
def _matmul_linear_outproj_kernel(
    A_ptr,  # y: (M, K) with M=B*S, K=H
    B_ptr,  # out_proj_weight^T: (K, N) with N=H
    Bias_ptr,  # out_proj_bias: (N)
    C_ptr,  # output: (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
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
        # Ensure float32 and contiguous for Triton
        x = x.contiguous().to(torch.float32)                                # (B, S, H)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)     # (3H, H)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)         # (3H,)

        B, S, H = x.shape
        M = B * S
        N_in = 3 * H

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        A = x.view(M, H)                           # (M, K)
        B_w = in_proj_weight.t().contiguous().view(H, N_in)  # (K, N_in=3H)
        BCx = torch.empty((M, N_in), dtype=torch.float32, device=x.device)

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_in = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_in, BLOCK_N))
        _matmul_linear_kernel[grid_in](
            A, B_w, in_proj_bias, BCx,
            M, N_in, H,
            A.stride(0), A.stride(1),
            B_w.stride(0), B_w.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        BCx = BCx.view(B, S, 3 * H)  # (B, S, 3H)

        # 2) Split BCx into B, C, x_proj
        B_tensor = BCx[:, :, :H]            # (B, S, H)
        C_tensor = BCx[:, :, H:2 * H]      # (B, S, H)
        x_proj = BCx[:, :, 2 * H:]         # (B, S, H)

        # 3) Elementwise gate: Bx = B * x_proj
        Bx = _triton_elementwise_mul_1d_kernel(B_tensor.view(-1), x_proj.view(-1), B_tensor.view(-1), B_tensor.numel(), BLOCK_SIZE=1024).view(B, S, H)

        # 4) Grouped causal conv:
        # Use left padding on Bx: pad 3 zeros on the left along S
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0), mode='constant', value=0.0)  # (B, H, S+3)
        convW = in_proj_weight[:, -4:].contiguous().to(torch.float32)                # (H, 4)
        conv_bias_t = conv_bias.contiguous().to(torch.float32)                       # (H,)

        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)     # (B, H, S)

        grid_conv = (B * H,)
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, convW, conv_bias_t, conv_out,
            B, H, S,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            convW.stride(0), convW.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128
        )

        # 5) Output gating: y = C * conv_out
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_1d_kernel(C_tensor.view(-1), conv_out_T.view(-1), C_tensor.view(-1), C_tensor.numel(), BLOCK_SIZE=1024).view(B, S, H)

        # 6) Final out-projection: y @ out_proj_weight^T + out_proj_bias
        A_out = y.view(M, H)                              # (M, K)
        B_out = out_proj_weight.t().contiguous().view(H, H)  # (K, N=H)
        out_bias = out_proj_bias.contiguous().to(torch.float32)  # (N=H,)

        output = torch.empty((M, H), dtype=torch.float32, device=x.device)
        grid_out = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _matmul_linear_outproj_kernel[grid_out](
            A_out, B_out, out_bias, output,
            M, H, H,
            A_out.stride(0), A_out.stride(1),
            B_out.stride(0), B_out.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        return output.view(B, S, H)


def run(*args):
    return ModelNew()(*args)
