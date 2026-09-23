import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel: linear projection out[B, S, M] = x @ W[:M, :].T + bias[:M]
# x: (B, S, H), W: (M, H), bias: (M)
@triton.jit
def TritonLinearProjectionKernel(
    x_ptr,            # *T, shape [B, S, H]
    W_ptr,            # *T, shape [M, H]
    bias_ptr,         # *T, shape [M]
    out_ptr,          # *T, shape [B, S, M]
    B, S, H, M,       # int32
    stride_x_b, stride_x_s, stride_x_h,    # strides for x (B, S, H)
    stride_w_m, stride_w_h,                # strides for W (M, H)
    stride_o_b, stride_o_s, stride_o_m,    # strides for out (B, S, M)
    BLOCK_M: tl.constexpr,                 # tile along M
    BLOCK_H: tl.constexpr                  # tile along H
):
    # Grid: (B*S, ceil(M/BLOCK_M))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    b = pid0 // S
    s = pid0 % S

    m_start = pid1 * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # Accumulator for out[b, s, m_offsets]
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Reduce over H
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Load x[b, s, h_offsets]
        x_ptr_row = x_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h
        x_vals = tl.load(x_ptr_row, mask=mask_h, other=0.0)

        # Load W[m_offsets, h_offsets]
        w_ptrs = W_ptr + m_offsets[:, None] * stride_w_m + h_offsets[None, :] * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0)

        # acc += sum_h(x_vals * w_vals[h, :]) for each m in m_offsets
        # We sum across H dimension
        # x_vals: [BLOCK_H], w_vals: [BLOCK_M, BLOCK_H] -> broadcast multiply -> [BLOCK_M, BLOCK_H]
        # sum across BLOCK_H
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += bias_vals

    # Store out[b, s, m_offsets]
    out_ptr_row = out_ptr + b * stride_o_b + s * stride_o_s + m_offsets * stride_o_m
    tl.store(out_ptr_row, acc, mask=mask_m)


# Triton kernel: element-wise gate, out[b, s, h] = b[b, s, h] * x_proj[b, s, h]
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, Out_ptr,
    B, S, H,
    stride_b_b, stride_b_s, stride_b_h,
    stride_x_b, stride_x_s, stride_x_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b
    s_start = pid_s * BLOCK_S
    h_start = pid_h * BLOCK_H

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # Iterate over tiles in S and H
    for si in range(0, BLOCK_S):
        s = s_start + si
        if s >= S:
            break
        for hi in range(0, BLOCK_H):
            h = h_start + hi
            if h >= H:
                break
            b_ptr = B_ptr + b * stride_b_b + s * stride_b_s + h * stride_b_h
            x_ptr = X_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h
            b_val = tl.load(b_ptr)
            x_val = tl.load(x_ptr)
            out_ptr = Out_ptr + b * stride_o_b + s * stride_o_s + h * stride_o_h
            tl.store(out_ptr, b_val * x_val)


# Triton kernel: left-pad along S by PAD, in_ptr: [B, H, S], out_ptr: [B, H, S+PAD], write out[:, :, PAD:] = in
@triton.jit
def TritonPadLeftKernel(
    in_ptr,           # *T, shape [B, H, S]
    out_ptr,          # *T, shape [B, H, S+PAD]
    B, H, S, PAD,     # int32
    stride_in_b, stride_in_h, stride_in_s,
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    S_out = S + PAD
    s_start = pid_s * BLOCK_S

    # Write zeros to first PAD columns
    for i in range(0, PAD):
        tl.store(out_ptr + b * stride_out_b + h * stride_out_h + i * stride_out_s, 0.0)

    # Copy in[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s = s_start + i
        if s < S:
            in_val = tl.load(in_ptr + b * stride_in_b + h * stride_in_h + s * stride_in_s)
            tl.store(out_ptr + b * stride_out_b + h * stride_out_h + (s + PAD) * stride_out_s, in_val)


# Triton kernel: grouped causal 1D conv (K=4, groups=H)
# Inputs:
#   Bx_pad: [B, H, S+3]  (grouped over H, sequence length S+3)
#   conv_weight: [H, 1, 4]
#   conv_bias: [H]
# Output:
#   out: [B, H, S]
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr,        # *T, shape [B, H, S+3]
    weight_ptr,        # *T, shape [H, 1, 4]
    bias_ptr,          # *T, shape [H]
    out_ptr,           # *T, shape [B, H, S]
    B, H, S, PAD,      # int32, PAD=3
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k, stride_w_c,  # weight strides (H, 4, 1); here c=0 so we ignore stride_w_c
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    S_in = S + PAD
    s_start = pid_s * BLOCK_S

    for si in range(0, BLOCK_S):
        s = s_start + si
        if s >= S:
            break
        # Accumulate over K=4
        acc = 0.0
        for k in range(0, 4):
            idx = s + k
            if idx < S_in:
                bx_ptr = Bx_pad_ptr + b * stride_bx_b + h * stride_bx_h + idx * stride_bx_s
                bx_val = tl.load(bx_ptr)
                w_ptr = weight_ptr + h * stride_w_h + 0 * stride_w_c + k * stride_w_k
                w_val = tl.load(w_ptr)
                acc += bx_val * w_val
        # Add bias
        bias_val = tl.load(bias_ptr + h)
        acc += bias_val
        out_ptr_pos = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s
        tl.store(out_ptr_pos, acc)


# Triton kernel: element-wise out[b, s, h] = a[b, s, h] * b[b, s, h]
@triton.jit
def TritonMulGateKernel(
    A_ptr, B_ptr, Out_ptr,
    B, S, H,
    stride_a_b, stride_a_s, stride_a_h,
    stride_b_b, stride_b_s, stride_b_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b
    s_start = pid_s * BLOCK_S
    h_start = pid_h * BLOCK_H

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_s = s_offsets < S
    mask_h = h_offsets < H

    for si in range(0, BLOCK_S):
        s = s_start + si
        if s >= S:
            break
        for hi in range(0, BLOCK_H):
            h = h_start + hi
            if h >= H:
                break
            a_ptr = A_ptr + b * stride_a_b + s * stride_a_s + h * stride_a_h
            b_ptr = B_ptr + b * stride_b_b + s * stride_b_s + h * stride_b_h
            a_val = tl.load(a_ptr)
            b_val = tl.load(b_ptr)
            out_ptr = Out_ptr + b * stride_o_b + s * stride_o_s + h * stride_o_h
            tl.store(out_ptr, a_val * b_val)


# Triton kernel: final projection out[b, s, h] = sum_h in[b, s, h] * W[h, h] + bias[h]
# Note: This is a simple GEMM-like kernel over H dimension for each (b, s). It mirrors F.linear with W_T = (H, H) and A = (B, S, H).
# However, since y shape is (B, S, H) and out_proj_weight is (H, H), bias (H), this kernel computes:
# out[b, s, h] = sum_{h_in=0..H-1} y[b, s, h_in] * out_proj_weight[h_in, h] + out_proj_bias[h]
# We'll use BLOCK_H=64 for H tiles.
@triton.jit
def TritonFinalProjectionKernel(
    A_ptr,            # *T, shape [B, S, H] (input y)
    W_ptr,            # *T, shape [H, H] (out_proj_weight)
    bias_ptr,         # *T, shape [H]
    Out_ptr,          # *T, shape [B, S, H]
    B, S, H,
    stride_a_b, stride_a_s, stride_a_h,
    stride_w_h, stride_w_out_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b
    s = pid_s
    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Accumulator for output vector [BLOCK_H]
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Reduce over H_in
    for h_in in range(0, H):
        a_ptr = A_ptr + b * stride_a_b + s * stride_a_s + h_in * stride_a_h
        a_val = tl.load(a_ptr)
        # Load W[h_in, h_offsets]
        w_ptrs = W_ptr + h_in * stride_w_h + h_offsets * stride_w_out_h
        w_vals = tl.load(w_ptrs, mask=mask_h, other=0.0)
        # Multiply and sum over BLOCK_H
        acc += tl.sum(w_vals * a_val, axis=0)

    # Add bias
    bias_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bias_vals

    # Store result
    out_ptrs = Out_ptr + b * stride_out_b + s * stride_out_s + h_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_h)


# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        Perform the same computation as the original run function, but using Triton kernels for compute.
        All tensors must be CUDA tensors. We preserve dtype.
        """
        assert x.is_cuda, "Input tensor x must be on CUDA device."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda, "in_proj_weight and in_proj_bias must be CUDA tensors."
        assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be CUDA tensors."
        assert out_proj_weight.is_cuda and out_proj_bias.is_cuda, "out_proj_weight and out_proj_bias must be CUDA tensors."

        # Ensure contiguous tensors
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape
        M = H  # hidden size

        # 1) Triple linear projection via Triton: B, C, x_proj
        # Compute B
        B_out = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonLinearProjectionKernel[(B * S, triton.cdiv(M, 64))](
            x, in_proj_weight[:M, :], in_proj_bias[:M], B_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[:M, :].stride(0), in_proj_weight[:M, :].stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_M=64, BLOCK_H=64
        )

        # Compute C
        C_out = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonLinearProjectionKernel[(B * S, triton.cdiv(M, 64))](
            x, in_proj_weight[M:2 * M, :], in_proj_bias[M:2 * M], C_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[M:2 * M, :].stride(0), in_proj_weight[M:2 * M, :].stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_M=64, BLOCK_H=64
        )

        # Compute x_proj
        x_proj_out = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonLinearProjectionKernel[(B * S, triton.cdiv(M, 64))](
            x, in_proj_weight[2 * M:3 * M, :], in_proj_bias[2 * M:3 * M], x_proj_out,
            B, S, H, M,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[2 * M:3 * M, :].stride(0), in_proj_weight[2 * M:3 * M, :].stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            BLOCK_M=64, BLOCK_H=64
        )

        # 2) Element-wise gating: Bx = B * x_proj (Triton)
        Bx = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(M, 64))](
            B_out, x_proj_out, Bx,
            B, S, M,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128, BLOCK_H=64
        )

        # 3) Left-pad for causal conv: Bx_pad[B, H, S+3]
        S_in = S + 3
        Bx_pad = torch.empty((B, M, S_in), dtype=x.dtype, device=x.device)
        TritonPadLeftKernel[(B, M, triton.cdiv(S_in, 128))](
            Bx, Bx_pad,
            B, M, S, 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=128
        )

        # 4) Grouped causal conv in Triton: conv_out[B, H, S]
        conv_out = torch.empty((B, M, S), dtype=x.dtype, device=x.device)
        TritonGroupedCausalConvKernel[(B, M, triton.cdiv(S, 128))](
            Bx_pad, conv_weight, conv_bias, conv_out,
            B, M, S, 3,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128
        )

        # 5) Output gating via Triton: y = C * conv_out
        y = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        # Transpose y for kernel to [B, S, M] (already in desired layout)
        TritonMulGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(M, 64))](
            C_out, conv_out, y,
            B, S, M,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.transpose(1, 2).contiguous().stride(0), conv_out.transpose(1, 2).contiguous().stride(1), conv_out.transpose(1, 2).contiguous().stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128, BLOCK_H=64
        )

        # 6) Final projection via Triton: out = linear(y, out_proj_weight, out_proj_bias)
        # Here we use TritonFinalProjectionKernel. Note: W is (H, H), A is (B, S, H), bias is (H).
        # We launch grid over (B, S, ceil(H/64)).
        out = torch.empty((B, S, M), dtype=x.dtype, device=x.device)
        TritonFinalProjectionKernel[(B, S, triton.cdiv(M, 64))](
            y, out_proj_weight, out_proj_bias, out,
            B, S, M,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=64
        )

        return out


def run(*args):
    return ModelNew()(*args)
