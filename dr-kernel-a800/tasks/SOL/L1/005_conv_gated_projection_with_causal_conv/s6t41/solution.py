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
    # Compute C = A @ B + Bias, A: (M,K), B: (K,N)
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
        # a: (BLOCK_M, BLOCK_K), b: (BLOCK_K, BLOCK_N)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # C = A * B, elementwise over a 2D matrix of size (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    c = a * b

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, Bias_ptr, C_ptr,
    B, H, S,
    stride_xb, stride_xh, stride_xs,
    stride_w_h, stride_w_k,
    stride_cb, stride_ch, stride_cs,
    BLOCK_S: tl.constexpr,
):
    # Each program handles one (b, h) row; computes conv out for s in [0, S)
    pid_bh = tl.program_id(0)  # ranges over B * H
    b = pid_bh // H
    h = pid_bh % H

    # Output vector of length S
    offs_s = tl.arange(0, BLOCK_S)
    s_mask = offs_s < S

    conv_out = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Accumulate over kernel size 4
    for k in range(4):
        # scalar weight for this h and k
        w = tl.load(W_ptr + h * stride_w_h + k * stride_w_k)
        # read X[b, h, s + 3 - k], zero-padded when s + 3 - k < 0 or >= S
        s_in = offs_s + (3 - k)
        x_ptrs = X_ptr + b * stride_xb + h * stride_xh + s_in * stride_xs
        x = tl.load(x_ptrs, mask=s_mask, other=0.0)
        conv_out += w * x

    # add bias[h]
    bias = tl.load(Bias_ptr + h)
    conv_out += bias

    # write back conv_out[b, h, :]
    c_ptrs = C_ptr + b * stride_cb + h * stride_ch + offs_s * stride_cs
    tl.store(c_ptrs, conv_out, mask=s_mask)


# Note: out-proj is implemented in Triton via matmul in forward; we do not use any torch.nn.functional in forward.


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only fused forward:
        1) BCx = in_proj(x) via Triton matmul
        2) Split into B, C, x_proj
        3) Bx = B * x_proj via Triton elementwise
        4) conv_out = grouped causal conv1d(Bx) via Triton kernel (left pad handled in-kernel)
        5) y = C * conv_out_T (conv_out transposed to (B, S, H)) via Triton elementwise
        6) output = out_proj(y) via Triton matmul
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # Ensure inputs are contiguous for Triton kernels
        x_c = x.contiguous()
        in_proj_weight_c = in_proj_weight.contiguous()
        in_proj_bias_c = in_proj_bias.contiguous()
        conv_weight_c = conv_weight.contiguous()
        conv_bias_c = conv_bias.contiguous()
        out_proj_weight_c = out_proj_weight.contiguous()
        out_proj_bias_c = out_proj_bias.contiguous()

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        B, S, H = x_c.shape
        K = H
        N = 3 * H
        M = B * S

        A = x_c.view(M, H)  # (M, K), contiguous
        B_w = in_proj_weight_c.t().view(H, N)  # (K, N), contiguous
        Bias = in_proj_bias_c.view(N)          # (N), contiguous

        BCx = torch.empty((M, N), dtype=torch.float32, device=x_c.device)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_linear_kernel[grid](
            A, B_w, Bias, BCx,
            M, N, K,
            A.stride(0), A.stride(1),
            B_w.stride(0), B_w.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        BCx = BCx.view(B, S, N)  # (B, S, 3H)

        # Split into B, C, x_proj
        BCx_B = BCx[:, :, :H]           # (B, S, H)
        BCx_C = BCx[:, :, H:2*H]        # (B, S, H)
        x_proj = BCx[:, :, 2*H:]        # (B, S, H)

        # 2) Elementwise gate: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=x_c.device)
        grid2 = (triton.cdiv(B, 1), triton.cdiv(S, 1), triton.cdiv(H, 1))
        # Flatten to 2D over (B*S, H)
        A2 = BCx_B.contiguous().view(B*S, H)
        B2 = x_proj.contiguous().view(B*S, H)
        C2 = Bx  # (B*S, H)

        _elementwise_mul_2d_kernel[(triton.cdiv(B*S, 128), triton.cdiv(H, 128))](
            A2, B2, C2,
            B*S, H,
            A2.stride(0), A2.stride(1),
            B2.stride(0), B2.stride(1),
            C2.stride(0), C2.stride(1),
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )
        Bx = Bx  # (B, S, H)

        # 3) Grouped causal conv with kernel_size=4, groups=H: conv_out: (B, H, S)
        # conv_weight is in_proj_weight[:, -4:], reshaped to (H, 4)
        conv_w = in_proj_weight_c[:, -4:].contiguous()  # (H, 4)

        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x_c.device)
        grid3 = (B * H,)
        _grouped_causal_conv1d_kernel[grid3](
            Bx.contiguous(), conv_w, conv_bias_c, conv_out,
            B, H, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_w.stride(0), conv_w.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=2, num_stages=2,
        )  # (B, H, S)

        # 4) Output gating: y = C * conv_out_T (conv_out is (B, H, S); transpose to (B, S, H))
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = torch.empty((B, S, H), dtype=torch.float32, device=x_c.device)

        A3 = BCx_C.contiguous().view(B*S, H)  # (B*S, H)
        B3 = conv_out_T.contiguous().view(B*S, H)  # (B*S, H)
        C3 = y.view(B*S, H)  # (B*S, H)

        _elementwise_mul_2d_kernel[(triton.cdiv(B*S, 128), triton.cdiv(H, 128))](
            A3, B3, C3,
            B*S, H,
            A3.stride(0), A3.stride(1),
            B3.stride(0), B3.stride(1),
            C3.stride(0), C3.stride(1),
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )
        y = y  # (B, S, H)

        # 5) Final out-proj: output = y @ out_proj_weight^T + out_proj_bias
        K2 = H
        M2 = B * S
        A4 = y.view(M2, K2).contiguous()           # (M2, K2)
        B4 = out_proj_weight_c.t().contiguous().view(K2, H)  # (K2, H)
        Bias2 = out_proj_bias_c.view(H)            # (H)

        output = torch.empty((M2, H), dtype=torch.float32, device=x_c.device)

        _matmul_linear_kernel[(triton.cdiv(M2, 64), triton.cdiv(H, 64))](
            A4, B4, Bias2, output,
            M2, H, K2,
            A4.stride(0), A4.stride(1),
            B4.stride(0), B4.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )
        output = output.view(B, S, H)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
