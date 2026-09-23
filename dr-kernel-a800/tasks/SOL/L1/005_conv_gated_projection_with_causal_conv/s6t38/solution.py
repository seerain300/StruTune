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
    # Compute C = A @ B + Bias, where A: (M,K), B: (K,N), Bias: (N), C: (M,N)
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
    A_ptr, B_ptr, C_ptr,
    L,  # total number of elements
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    a = tl.load(A_ptr + offs, mask=mask, other=1.0)
    b = tl.load(B_ptr + offs, mask=mask, other=1.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, ConvW_ptr, Bias_ptr, Out_ptr,
    B, H, S,
    stride_bb, stride_bh, stride_bs,
    stride_ow,  # stride for H dimension in Out (B, H, S)
    K: tl.constexpr,  # kernel_size (4 here)
):
    # One program per (b, h). Output is Out[b, h, s] for s in [0..S-1]
    b = tl.program_id(0)
    h = tl.program_id(1)
    for s_i in range(0, S):
        acc = 0.0
        for k in range(0, K):
            idx = s_i + K - 1 - k  # equivalent to s + 3 - k for kernel_size=4
            valid = (idx >= 0) & (idx < S)
            bx_ptr = Bx_ptr + b * stride_bb + h * stride_bh + idx * stride_bs
            val = tl.load(bx_ptr) if valid else 0.0
            w_ptr = ConvW_ptr + h * K + k
            w = tl.load(w_ptr)
            acc += val * w
        bias_val = tl.load(Bias_ptr + h)
        out_ptr = Out_ptr + b * stride_ow + h * S + s_i
        tl.store(out_ptr, acc)


@triton.jit
def _matmul_linear_kernel_outproj(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias, where A: (M,K), B: (K,N), Bias: (N), C: (M,N)
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
        Triton-only forward implementing the original computation:
          1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias -> (B, S, 3H)
          2) Split BCx into B, C, x_proj (each (B, S, H))
          3) Bx = B * x_proj
          4) Grouped causal conv with kernel_size=4, groups=H:
             conv_out: (B, H, S)
          5) y = C * conv_out  (C is (B, S, H), conv_out is (B, H, S); we transpose conv_out to (B, S, H) for elementwise multiply)
          6) out_proj: y @ out_proj_weight^T + out_proj_bias -> (B, S, H)
        All computations performed by Triton kernels; no torch functional calls in forward.
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # Ensure float32 compute
        x_f = x.float()
        in_proj_weight_f = in_proj_weight.float()
        in_proj_bias_f = in_proj_bias.float()
        conv_weight_f = conv_weight.float()  # (H, 1, 4)
        conv_bias_f = conv_bias.float()      # (H)
        out_proj_weight_f = out_proj_weight.float()  # (H, H)
        out_proj_bias_f = out_proj_bias.float()      # (H)

        B, S, H = x_f.shape
        device = x_f.device

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias -> (B, S, 3H)
        M = B * S
        K = H
        N = 3 * H

        A = x_f.view(M, K).contiguous()  # (M, K)
        B_w = in_proj_weight_f.t().contiguous().view(K, N)  # (K, N)
        Bias = in_proj_bias_f.contiguous().view(N)          # (N)

        BCx = torch.empty((M, N), dtype=torch.float32, device=device)

        BLOCK_M = 128
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
            num_warps=4, num_stages=3,
        )

        BCx = BCx.view(B, S, N)  # (B, S, 3H)

        # 2) Split into channels (no functional ops, just slicing)
        B_channel = BCx[:, :, :H]                 # (B, S, H)
        C_channel = BCx[:, :, H:2*H]              # (B, S, H)
        x_proj = BCx[:, :, 2*H:]                  # (B, S, H)

        # 3) Elementwise gate: Bx = B_channel * x_proj, (B, S, H)
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        L = B * S * H
        _elementwise_mul_1d_kernel[(triton.cdiv(L, 1024),)](
            B_channel, x_proj, Bx, L, BLOCK=1024, num_warps=4,
        )

        # 4) Grouped causal conv: conv_weight derived from conv_weight's last 4 values per channel
        #    conv_weight: (H, 1, 4) -> flatten channels and kernel to (H, 4)
        ConvW = conv_weight_f.view(H, 4).contiguous()  # (H, 4)

        # Output conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=device)

        grid_conv = (B, H)
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx, ConvW, conv_bias_f, conv_out,
            B, H, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_out.stride(1),  # stride for H dimension in Out (B, H, S)
            K=4,
            num_warps=4,
        )

        # 5) Output gating: y = C_channel * conv_out  (elementwise)
        #    conv_out: (B, H, S); C_channel: (B, S, H)
        conv_out_T = torch.empty((B, S, H), dtype=torch.float32, device=device)
        # Flatten pointers for Triton elementwise kernel
        _elementwise_mul_1d_kernel[(triton.cdiv(B * S * H, 1024),)](
            C_channel, conv_out.view(B, H, S).contiguous().view(B*S, H), conv_out_T, B * S * H, BLOCK=1024, num_warps=4,
        )

        y = conv_out_T  # (B, S, H)

        # 6) Final out-proj: y @ out_proj_weight^T + out_proj_bias -> (B, S, H)
        M2 = B * S
        K2 = H
        N2 = H

        A2 = y.view(M2, K2).contiguous()             # (M, K)
        B2 = out_proj_weight_f.t().contiguous().view(K2, N2)  # (K, N)
        Bias2 = out_proj_bias_f.contiguous().view(N2)          # (N)

        Output = torch.empty((M2, N2), dtype=torch.float32, device=device)

        BLOCK_M2 = 128
        BLOCK_N2 = 64
        BLOCK_K2 = 32
        grid2 = (triton.cdiv(M2, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _matmul_linear_kernel_outproj[grid2](
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
