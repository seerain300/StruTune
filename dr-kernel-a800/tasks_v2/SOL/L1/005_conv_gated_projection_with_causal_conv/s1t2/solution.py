import torch
import triton
import triton.language as tl


# Triton kernel: linear projection
# Computes out[B, S, M] = x @ weight[:M, :].T + bias[:M]
@triton.jit
def TritonLinearProjectionKernel(
    x_ptr,          # *const float, shape [B, S, H]
    weight_ptr,     # *const float, shape [M, H]
    bias_ptr,       # *const float, shape [M]
    out_ptr,        # *float, shape [B, S, M]
    B, S, H, M,     # int32
    stride_xb, stride_xs, stride_xh,   # int32 strides for x
    stride_ob, stride_os, stride_om,   # int32 strides for out
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

        # Load x row: x[b, s, h_offsets] -> [BLOCK_H]
        x_row_ptr = x_ptr + b * stride_xb + s * stride_xs + h_offsets * stride_xh
        x_row = tl.load(x_row_ptr, mask=mask_h, other=0.0).to(tl.float32)  # [BLOCK_H]

        # Load weight block: weight[m_offsets, h_offsets] -> [BLOCK_M, BLOCK_H]
        weight_ptr_block = weight_ptr + m_offsets[:, None] * H + h_offsets[None, :]
        mask_weight = mask_m[:, None] & mask_h[None, :]
        weight_block = tl.load(weight_ptr_block, mask=mask_weight, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_H]

        # Accumulate: acc += sum_h weight_block[m, h] * x_row[h]
        acc += tl.sum(weight_block * x_row[None, :], axis=1)

    # Add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)
    acc += bias_vals

    # Store out[b, s, m_offsets]
    out_ptr_block = out_ptr + b * stride_ob + s * stride_os + m_offsets * stride_om
    tl.store(out_ptr_block, acc, mask=mask_m)


# Triton kernel: elementwise multiplication y = a * b
# Assumes a, b, out have shape (B*S, M), but we will pass (B, S, M) by using strides (bs, M) mapping.
@triton.jit
def TritonElementwiseMulKernel(
    a_ptr, b_ptr, out_ptr,
    total_elems,  # int32, total number of elements = B*S*M
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)


# Triton kernel: pad left (pad=3) and grouped causal conv with K=4 and groups=M (channels)
# Computes conv_out[B, M, S] = sum_{k=0..3} Bx_padded[b, m, t + k] * weight[m, 0, k] + bias[m]
@triton.jit
def TritonPadAndGroupedCausalConvKernel(
    Bx_ptr,          # *const float, shape [B, S, H]
    weight_ptr,      # *const float, shape [M, 1, 4]
    bias_ptr,        # *const float, shape [M]
    out_ptr,         # *float, shape [B, M, S]
    B, S, H, M,      # int32
    stride_bx_b, stride_bx_s, stride_bx_h,   # strides for Bx
    stride_wm, stride_wk, stride_wc,         # strides for weight (M, 1, 4)
    stride_ob, stride_om, stride_os,         # strides for out (B, M, S)
    BLOCK_T: tl.constexpr                    # tile size over S (time dimension)
):
    # Grid: (B, M, ceil(S/BLOCK_T))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_t = tl.program_id(2)

    b = pid_b
    m = pid_m

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < S

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # K=4 causal kernel: sum over k=0..3 of Bx[b, m, t + k] * weight[m, 0, k]
    for k in range(0, 4):
        t_in = t_offsets + k
        mask_valid = (t_in < S) & mask_t

        bx_ptr = Bx_ptr + b * stride_bx_b + m * stride_bx_h + t_in * stride_bx_s
        bx_vals = tl.load(bx_ptr, mask=mask_valid, other=0.0).to(tl.float32)

        # Load weight scalar: weight[m, 0, k]
        w_ptr = weight_ptr + m * stride_wm + 0 * stride_wc + k * stride_wk
        w_val = tl.load(w_ptr).to(tl.float32)

        acc += bx_vals * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + m).to(tl.float32)
    acc += bias_val

    # Store conv_out[b, m, t_offsets]
    out_ptr_block = out_ptr + b * stride_ob + m * stride_om + t_offsets * stride_os
    tl.store(out_ptr_block, acc, mask=mask_t)


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
    stride_wm, stride_wk,              # int32 strides for weight (M, M)
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
        conv_weight: torch.Tensor,   # shape: [H, 1, 4]
        conv_bias: torch.Tensor,     # shape: [H]
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor
    ):
        """
        Full Triton implementation of the original 'run' function.
        All computations are performed by Triton kernels. Forward only allocates and launches kernels.
        """
        assert x.is_cuda, "Input must be CUDA tensor."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda, "in_proj_weight and in_proj_bias must be CUDA tensors."
        assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be CUDA tensors."
        assert out_proj_weight.is_cuda and out_proj_bias.is_cuda, "out_proj_weight and out_proj_bias must be CUDA tensors."

        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, S, H = x.shape
        M = H  # hidden_size

        # 1) Three linear projections via Triton: B_out, C_out, x_proj_out
        B_out = torch.empty((B, S, M), dtype=torch.float32, device=x.device)
        C_out = torch.empty((B, S, M), dtype=torch.float32, device=x.device)
        x_proj_out = torch.empty((B, S, M), dtype=torch.float32, device=x.device)

        # Launch for B
        TritonLinearProjectionKernel[(B*S, triton.cdiv(M, 64))](
            x, in_proj_weight[:M, :], in_proj_bias[:M], B_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_M=64, BLOCK_H=32
        )

        # Launch for C
        TritonLinearProjectionKernel[(B*S, triton.cdiv(M, 64))](
            x, in_proj_weight[M:2*M, :], in_proj_bias[M:2*M], C_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_M=64, BLOCK_H=32
        )

        # Launch for x_proj
        TritonLinearProjectionKernel[(B*S, triton.cdiv(M, 64))](
            x, in_proj_weight[2*M:3*M, :], in_proj_bias[2*M:3*M], x_proj_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            BLOCK_M=64, BLOCK_H=32
        )

        # 2) Element-wise gating: Bx = B_out * x_proj_out
        Bx_total = B * S * M
        Bx = torch.empty_like(B_out)  # (B, S, M)
        TritonElementwiseMulKernel[(triton.cdiv(Bx_total, 1024),)](
            B_out.reshape(-1), x_proj_out.reshape(-1), Bx.reshape(-1),
            Bx_total, BLOCK=1024
        )

        # 3) Grouped causal conv with padding=3 on the left (K=4). Compute Bx_padded via TritonPadAndGroupedCausalConvKernel
        # Note: We will recompute conv_out directly using Triton without calling torch.conv1d.
        conv_out = torch.empty((B, M, S), dtype=torch.float32, device=x.device)
        TritonPadAndGroupedCausalConvKernel[(B, M, triton.cdiv(S, 64))](
            Bx, conv_weight, conv_bias, conv_out,
            B, S, H, M,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=64
        )

        # 4) Output gating: y = C_out * conv_out
        y_total = B * S * M
        y = torch.empty_like(conv_out, device=x.device, dtype=torch.float32).transpose(1, 2).contiguous()  # (B, S, M) but conv_out is (B, M, S); gate elementwise in (B, S, M) view
        # To avoid confusion with transpose, compute y as a temporary (B, S, M): elementwise multiply in linear memory
        # We can compute it directly in Triton:
        y = torch.empty((B, S, M), dtype=torch.float32, device=x.device)
        TritonElementwiseMulKernel[(triton.cdiv(B*S*M, 1024),)](
            C_out.reshape(-1), conv_out.transpose(1, 2).reshape(-1), y.reshape(-1),
            B * S * M, BLOCK=1024
        )

        # 5) Final output projection via Triton GEMM + bias
        out = torch.empty((B, S, M), dtype=torch.float32, device=x.device)
        TritonGemmBiasKernel[(B*S, triton.cdiv(M, 64))](
            y, out_proj_weight, out_proj_bias, out,
            B, S, M,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_M=64, BLOCK_K=64
        )

        return out


def run(*args):
    return ModelNew()(*args)
