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
    # A: (M, K), row-major: stride_am for rows, stride_ak for cols
    # B: (K, N), row-major: stride_bk for rows, stride_bn for cols
    # C: (M, N), row-major: stride_cm for rows, stride_cn for cols
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
        # Accumulate in float32
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_1d_kernel(
    A_ptr, B_ptr, Out_ptr,
    M,  # total number of elements (flattened)
    BLOCK_SIZE: tl.constexpr,
):
    # Out = A * B, elementwise over 1D arrays of length M
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    out = a * b
    tl.store(Out_ptr + offs, out, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, ConvW_ptr, Bias_ptr, Out_ptr,
    B, H, S,  # dims
    stride_bb, stride_bh, stride_bs,  # strides for Bx: (B, H, S)
    conv_stride_h, conv_stride_k,    # strides for ConvW: (H, 4)
    out_stride_b, out_stride_h, out_stride_s,  # strides for Out: (B, H, S)
    BLOCK_S: tl.constexpr,  # tile over output sequence positions
):
    # Each program handles one (b, h) pair and computes output across s in [0, S)
    pid_bh = tl.program_id(0)  # combined over B*H
    b = pid_bh // H
    h = pid_bh % H

    # Accumulator for output at this (b, h)
    acc = tl.zeros((S,), dtype=tl.float32)

    # Loop over output sequence positions in tiles
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        # Compute convolution sum for each s in tile
        for k in range(4):  # kernel_size=4
            # idx = s + 3 - k (left padding = 3)
            idx = offs_s + 3 - k  # vector
            # Valid if 0 <= idx < S
            valid = (idx >= 0) & (idx < S) & mask_s
            # Load from Bx[b, h, idx], masked invalid loads return 0
            bx_ptr = Bx_ptr + b * stride_bb + h * stride_bh + idx * stride_bs
            val = tl.load(bx_ptr, mask=valid, other=0.0)

            # Load conv weight for this (h, k): ConvW[h, k]
            w = tl.load(ConvW_ptr + h * conv_stride_h + k * conv_stride_k)
            # Accumulate: acc += val * w for each s in the tile
            acc += val * w  # val is vector, w is scalar

        # Add bias[h]
        bias_val = tl.load(Bias_ptr + h * conv_stride_h)
        acc += bias_val

    # Store results back to Out[b, h, s] for s in [0, S)
    out_base = Out_ptr + b * out_stride_b + h * out_stride_h
    for s in range(S):
        tl.store(out_base + s * out_stride_s, acc[s])


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias.
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Ensure float32 compute
    A = x.contiguous().view(M, H).to(torch.float32)              # (M, K)
    B_w = in_proj_weight.contiguous().t().view(H, N).to(torch.float32)  # (K, N)
    Bias = in_proj_bias.contiguous().view(N).to(torch.float32)   # (N)

    C = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, B_w, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return C.view(B, S, 3 * H)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that avoids PyTorch functional ops in host code.
        Computes the same steps as the original: in_proj, gate B, grouped causal conv, gate with C, out_proj.
        Returns output of shape (B, S, H), cast to float32.
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias -> (B, S, 3H)
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)
        B, S, N = BCx.shape
        H = N // 3  # hidden_size

        # Split into B, C, x_proj
        B_channel = BCx[:, :, :H]                 # (B, S, H)
        C_channel = BCx[:, :, H:2 * H]           # (B, S, H)
        x_proj = BCx[:, :, 2 * H:]               # (B, S, H)

        # 2) Elementwise gate: Bx = B_channel * x_proj
        total_elems = B * S * H
        Bx_flat = torch.empty((total_elems,), dtype=torch.float32, device=x.device)
        _elementwise_mul_1d_kernel[(triton.cdiv(total_elems, 1024),)](
            B_channel.view(-1), x_proj.view(-1), Bx_flat,
            total_elems,
            BLOCK_SIZE=1024,
        )
        Bx = Bx_flat.view(B, S, H)  # (B, S, H), float32

        # 3) Grouped causal conv: conv_weight derived from in_proj_weight's last 4 columns: (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_bias_f = conv_bias.contiguous().to(torch.float32)          # (H)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * H,)
        _grouped_causal_conv1d_kernel[grid](
            Bx, conv_w, conv_bias_f, conv_out,
            B, H, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_w.stride(0), conv_w.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
        )

        # 4) Output gating: y = C_channel * conv_out; conv_out shape (B, H, S), C_channel (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y_flat = torch.empty((total_elems,), dtype=torch.float32, device=x.device)
        _elementwise_mul_1d_kernel[(triton.cdiv(total_elems, 1024),)](
            C_channel.view(-1), conv_out_T.view(-1), y_flat,
            total_elems,
            BLOCK_SIZE=1024,
        )
        y = y_flat.view(B, S, H)  # (B, S, H), float32

        # 5) Final out-proj: y @ out_proj_weight^T + out_proj_bias -> (B, S, H)
        K2 = H
        N2 = H
        M2 = B * S

        A2 = y.contiguous().view(M2, K2).to(torch.float32)         # (M2, K2)
        B2 = out_proj_weight.contiguous().t().view(K2, N2).to(torch.float32)  # (K2, N2)
        Bias2 = out_proj_bias.contiguous().view(N2).to(torch.float32)   # (N2)

        Output = torch.empty((M2, N2), dtype=torch.float32, device=x.device)

        BLOCK_M2 = 128
        BLOCK_N2 = 64
        BLOCK_K2 = 32
        grid2 = (triton.cdiv(M2, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _matmul_linear_kernel[grid2](
            A2, B2, Bias2, Output,
            M2, N2, K2,
            A2.stride(0), A2.stride(1),
            B2.stride(0), B2.stride(1),
            Output.stride(0), Output.stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=3,
        )

        output = Output.view(B, S, H)
        return output


def run(*args):
    return ModelNew()(*args)
