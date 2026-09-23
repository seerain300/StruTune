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
    # Computes C = A @ B + Bias, where
    # A: (M, K) row-major, B: (K, N) row-major, Bias: (N,)
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
def _elementwise_mul_2d(A_ptr, B_ptr, C_ptr,
                         Bsz, Hsz, Ssz,
                         stride_ab, stride_as, stride_ah,
                         stride_bb, stride_bs, stride_bh,
                         stride_cb, stride_cs, stride_ch,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Elementwise C = A * B on (Bsz, Ssz, Hsz) tensors
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_s = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)

    for b in range(0, Bsz):
        a_ptrs = A_ptr + b * stride_ab + offs_s[:, None] * stride_as + offs_h[None, :] * stride_ah
        b_ptrs = B_ptr + b * stride_bb + offs_s[:, None] * stride_bs + offs_h[None, :] * stride_bh
        c_ptrs = C_ptr + b * stride_cb + offs_s[:, None] * stride_cs + offs_h[None, :] * stride_ch

        mask = (offs_s[:, None] < Ssz) & (offs_h[None, :] < Hsz)
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b_ = tl.load(b_ptrs, mask=mask, other=0.0)
        c = a * b_
        tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d(X_ptr, W_ptr, Bias_ptr, Y_ptr,
                            Ssz, Hsz, K: tl.constexpr,
                            stride_xb, stride_xh, stride_xs,
                            stride_yb, stride_yh, stride_ys):
    # Grouped causal conv for one group: for each (b, h), compute y[b, h, s] = sum_k W[h, k] * X[b, h, s + k] + Bias[h]
    # X: (B, H, Ssz+3), Y: (B, H, Ssz), W: (H, K), Bias: (H,)
    pid_bh = tl.program_id(0)
    b = pid_bh // Hsz
    h = pid_bh % Hsz

    # Each program handles one (b, h), loop over s
    s = 0
    while s < Ssz:
        # sum over kernel 4 elements: s0 = s+3, s1 = s+2, s2 = s+1, s3 = s
        # X offsets for these 4 positions
        s0 = s + 3
        s1 = s + 2
        s2 = s + 1
        s3 = s

        x0 = tl.load(X_ptr + b * stride_xb + h * stride_xh + s0 * stride_xs, mask=True, other=0.0)
        x1 = tl.load(X_ptr + b * stride_xb + h * stride_xh + s1 * stride_xs, mask=True, other=0.0)
        x2 = tl.load(X_ptr + b * stride_xb + h * stride_xh + s2 * stride_xs, mask=True, other=0.0)
        x3 = tl.load(X_ptr + b * stride_xb + h * stride_xh + s3 * stride_xs, mask=True, other=0.0)

        w = tl.load(W_ptr + h * stride_wh + tl.arange(0, K))
        # K is small (4), load w0..w3
        w0 = w[0]
        w1 = w[1]
        w2 = w[2]
        w3 = w[3]

        acc = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + tl.load(Bias_ptr + h)

        tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys, acc)

        s += 1


@triton.jit
def _matmul_linear_out_proj(A_ptr, B_ptr, Bias_ptr, C_ptr,
                             M, N, K,
                             stride_am, stride_ak,
                             stride_bk, stride_bn,
                             stride_cm, stride_cn,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Computes C = A @ B + Bias, where
    # A: (M, K) row-major, B: (K, N) row-major, Bias: (N,)
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


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the given operation flow:
        1) Triple linear projection: x -> (B, C, x_proj) via in_proj
        2) Element-wise gating: Bx = B * x_proj
        3) Grouped causal 1D convolution on Bx with kernel_size=4 (groups=H, causal)
        4) Output gating: y = C * conv_out
        5) Final output projection: y -> out_proj(y)
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        B, S, H = x.shape

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        # A: (B*S, H), B: (H, 3H)
        M = B * S
        N_in = 3 * H

        A = x.contiguous().view(M, H).to(torch.float32)
        B_w = in_proj_weight.t().contiguous().view(H, N_in).to(torch.float32)
        Bias_in = in_proj_bias.contiguous().view(N_in).to(torch.float32)

        BCx = torch.empty((M, N_in), dtype=torch.float32, device=x.device)

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_in = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_in, BLOCK_N))
        _matmul_linear_kernel[grid_in](
            A, B_w, Bias_in, BCx,
            M, N_in, H,
            A.stride(0), A.stride(1),
            B_w.stride(0), B_w.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        BCx = BCx.view(B, S, N_in)

        # Split into B, C, x_proj: (B, S, H) each
        B_ = BCx[:, :, :H]
        C_ = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # 2) Element-wise gating: Bx = B_ * x_proj
        Bx = _triton_elementwise_mul_2d(B_, x_proj, (B, S, H), H, S, H,
                                         B_.stride(0), B_.stride(2), B_.stride(1),
                                         x_proj.stride(0), x_proj.stride(2), x_proj.stride(1),
                                         stride_cb, stride_cs, stride_ch,
                                         BLOCK_M=128, BLOCK_N=64)

        # 3) Grouped causal conv: kernel_size=4, groups=H
        # Extract conv_weight from in_proj_weight's last 4 columns: (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().view(H, 4).to(torch.float32)  # (H, 4)
        conv_b = conv_bias.contiguous().view(H).to(torch.float32)  # (H,)

        # Pad Bx on sequence dim with 3 zeros (left pad for causal)
        Bx_for_conv = Bx.transpose(-1, -2).contiguous()  # (B, H, S)
        Bx_padded = torch.empty((B, H, S + 3), dtype=torch.float32, device=x.device)
        # We will feed the padded tensor to Triton which will not read out-of-range via masks

        # Launch grouped causal conv kernel: one program per (b, h)
        grid_conv = (B * H,)
        _grouped_causal_conv1d[grid_conv](
            Bx_padded, conv_w, conv_b, conv_out,
            S, H, K=4,
            stride_xb=Bx_padded.stride(0), stride_xh=Bx_padded.stride(1), stride_xs=Bx_padded.stride(2),
            stride_yb=conv_out.stride(0), stride_yh=conv_out.stride(1), stride_ys=conv_out.stride(2),
        )

        # 4) Output gating: y = C_ * conv_out, with conv_out shape (B, H, S)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_2d(C_, conv_out_T, (B, S, H), H, S, H,
                                        C_.stride(0), C_.stride(2), C_.stride(1),
                                        conv_out_T.stride(0), conv_out_T.stride(2), conv_out_T.stride(1),
                                        stride_cb, stride_cs, stride_ch,
                                        BLOCK_M=128, BLOCK_N=64)

        # 5) Final out-proj: y (B, S, H) -> output (B, S, H)
        M_out = B * S
        A_out = y.contiguous().view(M_out, H).to(torch.float32)
        B_out = out_proj_weight.t().contiguous().view(H, H).to(torch.float32)
        Bias_out = out_proj_bias.contiguous().view(H).to(torch.float32)

        output = torch.empty((M_out, H), dtype=torch.float32, device=x.device)

        grid_out = (triton.cdiv(M_out, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _matmul_linear_out_proj[grid_out](
            A_out, B_out, Bias_out, output,
            M_out, H, H,
            A_out.stride(0), A_out.stride(1),
            B_out.stride(0), B_out.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        return output.view(B, S, H)


def run(*args):
    return ModelNew()(*args)
