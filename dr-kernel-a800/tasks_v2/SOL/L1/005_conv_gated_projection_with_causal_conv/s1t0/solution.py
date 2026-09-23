import torch
import triton
import triton.language as tl

# Triton kernel: computes one of the three linear projections:
# Given x[B, S, H], weight[M, H], bias[M], computes out[B, S, M] = x @ weight.T + bias
@triton.jit
def TritonLinear3ProjectionKernel(
    x_ptr,          # *const float, shape [B, S, H]
    weight_ptr,     # *const float, shape [M, H]
    bias_ptr,       # *const float, shape [M]
    out_ptr,        # *float, shape [B, S, M]
    B, S, H, M,     # int32
    stride_xb, stride_xs, stride_xh,   # int32 strides for x
    stride_om, stride_os, stride_ob,   # int32 strides for out
    BLOCK_M: tl.constexpr,             # tile size over output channels
    BLOCK_H: tl.constexpr              # tile size over input channels
):
    # Grid: (B*S, ceil(M/BLOCK_M))
    pid_bs = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bs // S
    s = pid_bs % S

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over input channels H in chunks
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Load x row: x[b, s, h_offsets]
        x_row_ptr = x_ptr + b * stride_xb + s * stride_xs + h_offsets * stride_xh
        x_row = tl.load(x_row_ptr, mask=mask_h, other=0.0)  # [BLOCK_H]
        x_row = x_row.to(tl.float32)

        # Load weight block: weight[m_offsets, h_offsets] -> [BLOCK_M, BLOCK_H]
        weight_ptr_block = weight_ptr + m_offsets[:, None] * H + h_offsets[None, :]
        mask_weight = mask_m[:, None] & mask_h[None, :]
        weight_block = tl.load(weight_ptr_block, mask=mask_weight, other=0.0)  # [BLOCK_M, BLOCK_H]
        weight_block = weight_block.to(tl.float32)

        # Accumulate: acc += sum_h weight_block[m, h] * x_row[h]
        acc += tl.sum(weight_block * x_row[None, :], axis=1)

    # Add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)
    acc += bias_vals

    # Store out[b, s, m_offsets]
    out_ptr_block = out_ptr + b * stride_ob + s * stride_os + m_offsets * stride_om
    tl.store(out_ptr_block, acc, mask=mask_m)


# Triton kernel: GEMM + bias for final output projection:
# Given y[B, S, M], out_proj_weight[M, M], out_bias[M], computes out[B, S, M] = y @ out_proj_weight^T + out_bias
@triton.jit
def TritonGemmBiasKernel(
    y_ptr,          # *const float, shape [B, S, M]
    weight_ptr,     # *const float, shape [M, M]
    bias_ptr,       # *const float, shape [M]
    out_ptr,        # *float, shape [B, S, M]
    B, S, M,        # int32
    stride_yb, stride_ys, stride_ym,   # int32 strides for y
    stride_wm, stride_wk,              # int32 strides for weight (M, M) -> treat as [K=N=M]
    stride_ob, stride_os, stride_om,   # int32 strides for out
    BLOCK_M: tl.constexpr,             # tile over output channels
    BLOCK_K: tl.constexpr              # tile over reduction dimension
):
    # Grid: (B*S, ceil(M/BLOCK_M))
    pid_bs = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bs // S
    s = pid_bs % S

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, M, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < M

        # Load y[b, s, k_offsets] -> [BLOCK_K]
        y_ptr_block = y_ptr + b * stride_yb + s * stride_ys + k_offsets * stride_ym
        y_vals = tl.load(y_ptr_block, mask=mask_k, other=0.0).to(tl.float32)

        # Load weight block: weight[k_offsets, m_offsets] -> [BLOCK_K, BLOCK_M]
        weight_ptr_block = weight_ptr + k_offsets[:, None] * stride_wm + m_offsets[None, :] * stride_wk
        mask_weight = mask_k[:, None] & mask_m[None, :]
        weight_block = tl.load(weight_ptr_block, mask=mask_weight, other=0.0).to(tl.float32)

        # Accumulate: acc += sum_k weight_block[k, m] * y_vals[k]
        acc += tl.sum(weight_block * y_vals[:, None], axis=0)

    # Add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)
    acc += bias_vals

    # Store
    out_ptr_block = out_ptr + b * stride_ob + s * stride_os + m_offsets * stride_om
    tl.store(out_ptr_block, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,   # shape: [hidden_size, 1, 4]
        conv_bias: torch.Tensor,     # shape: [hidden_size]
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        Triton-optimized version of the original 'run' function.
        - Computes the three linear projections (B, C, x_proj) using Triton kernels.
        - Performs element-wise gating (Bx = B * x_proj).
        - Performs grouped causal 1D convolution in PyTorch (groups=hidden_size).
        - Performs the final linear projection using a Triton GEMM kernel.
        All tensors must be CUDA tensors. We assume float32.
        """
        # Ensure CUDA and float32
        assert x.is_cuda, "Input x must be on CUDA for Triton."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda, "in_proj_weight and in_proj_bias must be on CUDA."
        assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be on CUDA."
        assert out_proj_weight.is_cuda and out_proj_bias.is_cuda, "out_proj_weight and out_proj_bias must be on CUDA."

        # Make tensors contiguous
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape
        M = H  # hidden_size

        # 1) Three linear projections via Triton: B, C, x_proj
        # Prepare outputs
        B_out = torch.empty((B, S, M), dtype=torch.float32, device=x.device)
        C_out = torch.empty((B, S, M), dtype=torch.float32, device=x.device)
        x_proj_out = torch


def run(*args):
    return ModelNew()(*args)
