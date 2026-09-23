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
    """
    Compute C = A @ B + Bias, where:
    - A is (M, K)
    - B is (K, N)
    - C is (M, N)
    All in float32.
    """
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
def _elementwise_mul_1d_kernel(
    A_ptr, B_ptr, Out_ptr,
    N,  # total number of elements
    BLOCK: tl.constexpr,
):
    """
    Elementwise multiply: Out[i] = A[i] * B[i] for i in [0, N).
    A and B are flattened 1D tensors of length N.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    out = a * b
    tl.store(Out_ptr + offs, out, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, ConvW_ptr, Bias_ptr, Out_ptr,
    B, H, S,
    stride_bb, stride_bh, stride_bs,
    conv_stride_h, conv_stride_k,
    out_stride_bo, out_stride_bh, out_stride_bs,
):
    """
    Grouped causal 1D convolution over (B, H, S) with kernel_size=4 and groups=H.
    Input Bx_ptr: (B, H, S), ConvW_ptr: (H, 4), Bias_ptr: (H), Out_ptr: (B, H, S).
    We emulate left-padding by using masked loads for idx < 0.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Loop over output positions s = 0..S-1
    for s_i in range(S):
        acc = 0.0
        # Accumulate over k in 0..3
        for k in range(4):
            idx = s_i + 3 - k  # pad=3 for kernel_size=4
            # Build pointer to Bx[b, h, idx]
            bx_ptr = Bx_ptr + b * stride_bb + h * stride_bh + idx * stride_bs
            # Masked load to emulate zero padding when idx < 0
            val = tl.load(bx_ptr, mask=(idx >= 0), other=0.0)
            # Load conv weight for this (h, k)
            w_ptr = ConvW_ptr + h * conv_stride_h + k * conv_stride_k
            w = tl.load(w_ptr)
            acc += val * w
        # Add bias
        bias_val = tl.load(Bias_ptr + h)
        acc += bias_val
        # Store to Out[b, h, s_i]
        out_ptr = Out_ptr + b * out_stride_bo + h * out_stride_bh + s_i * out_stride_bs
        tl.store(out_ptr, acc)


@triton.jit
def _out_proj_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + Bias, where:
    - A is (M, K)
    - B is (K, N)
    - C is (M, N)
    All in float32. Same as _matmul_linear_kernel.
    """
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
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that avoids any torch functional ops in host code.
        Computes: in_proj, elementwise gating, grouped causal conv, output gating, out_proj.
        Returns output of shape (B, S, H).
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias -> (B, S, 3H)
        B, S, H = x.shape
        K_in = H
        N_in = 3 * H
        M = B * S

        # Cast inputs to float32 for computation
        A = x.contiguous().view(M, H).to(torch.float32)
        Bw = in_proj_weight.t().contiguous().view(H, N_in).to(torch.float32)  # (K, N) = (H, 3H)
        Bias = in_proj_bias.contiguous().view(N_in).to(torch.float32)

        BCx = torch.empty((M, N_in), dtype=torch.float32, device=x.device)

        # Launch in_proj matmul kernel
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_in = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_in, BLOCK_N))
        _matmul_linear_kernel[grid_in](
            A, Bw, Bias, BCx,
            M, N_in, K_in,
            A.stride(0), A.stride(1),
            Bw.stride(0), Bw.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        BCx = BCx.view(B, S, 3 * H)

        # Split channels: B_channel, C_channel, x_proj
        B_channel = BCx[:, :, :H]                 # (B, S, H)
        C_channel = BCx[:, :, H:2 * H]            # (B, S, H)
        x_proj = BCx[:, :, 2 * H:]                # (B, S, H)

        # 2) Elementwise gate: Bx = B_channel * x_proj
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        # Flatten to 1D for elementwise kernel
        A_flat = B_channel.reshape(-1).contiguous()
        B_flat = x_proj.reshape(-1).contiguous()
        Out_flat = Bx.reshape(-1).contiguous()
        BLOCK = 1024
        grid_mul = (triton.cdiv(B_flat.numel(), BLOCK),)
        _elementwise_mul_1d_kernel[grid_mul](
            A_flat, B_flat, Out_flat,
            B_flat.numel(),
            BLOCK=BLOCK,
            num_warps=4,
        )

        # 3) Grouped causal conv: conv_weight from in_proj_weight's last 4 columns -> (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_bias_f = conv_bias.contiguous().to(torch.float32)          # (H,)

        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)

        # Launch grouped causal conv kernel
        grid_conv = (B, H)
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx, conv_w, conv_bias_f, conv_out,
            B, H, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_w.stride(0), conv_w.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4,
        )

        # 4) Output gating: y = C_channel * conv_out (elementwise), align shapes by transposing conv_out to (B, S, H)
        conv_out_T = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        C_flat = C_channel.reshape(-1).contiguous()  # (B*S*H,)
        conv_flat = conv_out.reshape(B * H * S).contiguous()  # (B*H*S,)
        Out_flat2 = conv_out_T.reshape(-1).contiguous()
        grid_gate = (triton.cdiv(C_flat.numel(), BLOCK),)
        _elementwise_mul_1d_kernel[grid_gate](
            C_flat, conv_flat, Out_flat2,
            C_flat.numel(),
            BLOCK=BLOCK,
            num_warps=4,
        )

        # 5) Final out-proj: y @ out_proj_weight^T + out_proj_bias -> (B, S, H)
        y = conv_out_T  # (B, S, H), float32
        M2 = B * S
        K2 = H
        N2 = H
        A2 = y.view(M2, K2).to(torch.float32)
        B2 = out_proj_weight.t().contiguous().view(K2, N2).to(torch.float32)
        Bias2 = out_proj_bias.contiguous().view(N2).to(torch.float32)
        Output = torch.empty((M2, N2), dtype=torch.float32, device=x.device)

        grid_out = (triton.cdiv(M2, BLOCK_M), triton.cdiv(N2, BLOCK_N))
        _out_proj_kernel[grid_out](
            A2, B2, Bias2, Output,
            M2, N2, K2,
            A2.stride(0), A2.stride(1),
            B2.stride(0), B2.stride(1),
            Output.stride(0), Output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        output = Output.view(B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
