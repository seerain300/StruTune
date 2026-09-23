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
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    Bsz, S, H,
    stride_as, stride_ax, stride_ay,
    stride_bs, stride_bx, stride_by,
    stride_cs, stride_cx, stride_cy,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one (b, h) pair across sequence dimension tiles
    b = pid // H
    h = pid % H
    # Loop over S in tiles
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        a_ptrs = A_ptr + b * stride_as + h * stride_ay + offs * stride_ax
        b_ptrs = B_ptr + b * stride_bs + h * stride_by + offs * stride_bx
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b_val = tl.load(b_ptrs, mask=mask, other=0.0)
        c = a * b_val
        c_ptrs = C_ptr + b * stride_cs + h * stride_cy + offs * stride_cx
        tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, conv_w_ptr, conv_bias_ptr, out_ptr,
    Bsz, S, H,
    stride_bx_row, stride_bx_col,
    stride_out_row, stride_out_col,
    BLOCK_S: tl.constexpr,
):
    # Each program handles one (b, h) pair, outputs across S
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    # conv_w_ptr is (H, 4), conv_bias_ptr is (H,)
    # We compute out[b, h, :] of length S
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
        # kernel_size = 4, causal padding = 3 on left
        for k in range(4):
            # For causal, input index is s + 3 - k
            s_in = offs + 3 - k
            valid = s_in < S
            # Address of Bx[b, h, s_in] if valid, else 0
            ptrs = Bx_ptr + b * stride_bx_row + h * stride_bx_col + s_in * stride_bx_col
            val = tl.load(ptrs, mask=mask & valid, other=0.0)
            w = tl.load(conv_w_ptr + h * 4 + k, mask=True, other=0.0)
            acc += val * w
        # Add bias
        bias = tl.load(conv_bias_ptr + h, mask=True, other=0.0)
        acc += bias
        # Store to out[b, h, s]
        out_ptrs = out_ptr + b * stride_out_row + h * stride_out_col + offs * stride_out_col
        tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor) -> torch.Tensor:
        # x: (B, S, H), float32 by default
        assert TRITON_AVAILABLE, "Triton is not available"

        Bsz, S, H = x.shape
        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        in_proj_weight_T = in_proj_weight.t().contiguous()      # (H, 3H)
        in_proj_bias_c = in_proj_bias.contiguous()              # (3H,)
        M = Bsz * S
        K = H
        N = 3 * H
        A = x.contiguous().view(M, K)                           # (M, K)
        BCx = torch.empty((M, N), dtype=torch.float32, device=x.device)
        grid_in = (triton.cdiv(M, 128), triton.cdiv(N, 64))
        _matmul_linear_kernel[grid_in](
            A, in_proj_weight_T, in_proj_bias_c, BCx,
            M, N, K,
            A.stride(0), A.stride(1),
            in_proj_weight_T.stride(0), in_proj_weight_T.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )
        BCx = BCx.view(Bsz, S, N)                                # (B, S, 3H)

        # 2) Split BCx into B, C, x_proj along last dim
        # Using PyTorch slicing to obtain views (no F.conv1d/F.linear here)
        B_part = BCx[:, :, :H]                                  # (B, S, H)
        C_part = BCx[:, :, H:2*H]                               # (B, S, H)
        x_proj = BCx[:, :, 2*H:]                                # (B, S, H)

        # 3) Elementwise gate: Bx = B_part * x_proj
        Bx = torch.empty((Bsz, S, H), dtype=torch.float32, device=x.device)
        # Launch Triton elementwise kernel
        grid_mul = (Bsz * S,)                                   # one program per (b, s) across h=H handled in loop inside kernel
        # Note: Triton elementwise kernel expects flattened memory. We pass strides of 3D tensor.
        stride_as, stride_ax, stride_ay = Bx.stride()           # for elementwise, we use strides of flattened view; we pass pointers directly and let Triton infer
        stride_bs, stride_bx, stride_by = Bx.stride()           # same
        stride_cs, stride_cx, stride_cy = Bx.stride()
        # We need to launch with BLOCK_S as a multiple of H; but elementwise kernel above loops inside. Use a simple 1D grid with one program per (b, h) across S.
        # Better: implement per-(b,h) loop in Triton:
        grid_mul = (Bsz * H,)
        _elementwise_mul_2d_kernel[grid_mul](
            B_part, x_proj, Bx,
            Bsz, S, H,
            1, 1, 1,                                         # stride placeholders; we will pass actual strides via pointer arithmetic in-kernel
            1, 1, 1,
            1, 1, 1,
            BLOCK_S=128,
            num_warps=2, num_stages=1,
        )

        # 4) Grouped causal conv: conv_out = grouped conv(Bx, conv_weight, conv_bias)
        # conv_weight is (H, 1, 4), we take (H, 4)
        conv_w = conv_weight.view(H, 4).contiguous().to(torch.float32)  # (H, 4)
        conv_bias_c = conv_bias.contiguous().to(torch.float32)          # (H,)
        # Build output conv_out: (B, H, S)
        conv_out = torch.empty((Bsz, H, S), dtype=torch.float32, device=x.device)
        grid_conv = (Bsz * H,)
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx, conv_w, conv_bias_c, conv_out,
            Bsz, S, H,
            S, 1,                                        # stride_bx_row=S, stride_bx_col=1 for (B,H,S) flattened view
            conv_out.stride(0), conv_out.stride(1),    # stride_out_row=H, stride_out_col=1
            BLOCK_S=128,
            num_warps=2, num_stages=2,
        )

        # 5) Output gating: y = C_part * conv_out.T
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = torch.empty((Bsz, S, H), dtype=torch.float32, device=x.device)
        grid_gate = (Bsz * H,)
        _elementwise_mul_2d_kernel[grid_gate](
            C_part, conv_out_T, y,
            Bsz, S, H,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            BLOCK_S=128,
            num_warps=2, num_stages=1,
        )

        # 6) Final out-proj: output = y @ out_proj_weight^T + out_proj_bias
        # out_proj_weight: (H, H), out_proj_bias: (H,)
        out_proj_weight_T = out_proj_weight.t().contiguous().to(torch.float32)  # (H, H)
        out_proj_bias_c = out_proj_bias.contiguous().to(torch.float32)          # (H,)
        M_out = Bsz * S
        K_out = H
        N_out = H
        A_out = y.contiguous().view(M_out, K_out)                               # (M_out, H)
        output = torch.empty((M_out, N_out), dtype=torch.float32, device=y.device)
        grid_out = (triton.cdiv(M_out, 128), triton.cdiv(N_out, 64))
        _matmul_linear_kernel[grid_out](
            A_out, out_proj_weight_T, out_proj_bias_c, output,
            M_out, N_out, K_out,
            A_out.stride(0), A_out.stride(1),
            out_proj_weight_T.stride(0), out_proj_weight_T.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )
        output = output.view(Bsz, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
