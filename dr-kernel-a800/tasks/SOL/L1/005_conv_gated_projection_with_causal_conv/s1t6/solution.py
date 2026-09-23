import torch
import triton
import triton.language as tl

# Triton kernel: Linear projection out[b, s, m] = sum_h x[b, s, h] * W[m, h] + bias[m]
# x: [B, S, H], W: [M, H], bias: [M] -> out: [B, S, M]
@triton.jit
def TritonLinearProjectionKernel(
    x_ptr,          # *const T, [B, S, H]
    W_ptr,          # *const T, [M, H]
    bias_ptr,       # *const T, [M]
    out_ptr,        # *T, [B, S, M]
    B, S, H, M,     # int32
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_o_b, stride_o_s, stride_o_m,
    BLOCK_S: tl.constexpr,  # tile size over S
    BLOCK_M: tl.constexpr,  # tile size over M (channels)
    BLOCK_H: tl.constexpr   # tile size over H for accumulation
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    s = pid_s
    m_start = pid_m * BLOCK_M

    # Compute s offsets for this tile
    s_offsets = s + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Accumulator for out[b, s_offsets, m_start:m_start+BLOCK_M]
    acc = tl.zeros([BLOCK_S, BLOCK_M], dtype=tl.float32)  # accumulate in fp32 for stability

    # Loop over H in chunks
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Load x row vector: shape [BLOCK_S] for s_offsets
        x_row_ptrs = x_ptr + b * stride_x_b + s_offsets[:, None] * stride_x_s + h_offsets[None, :] * stride_x_h
        x_row = tl.load(x_row_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

        # Load W block: shape [BLOCK_M, BLOCK_H]
        w_ptrs = W_ptr + (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_w_m + h_offsets[None, :] * stride_w_h
        w_block = tl.load(w_ptrs, mask=(m_start + tl.arange(0, BLOCK_M))[:, None] < M, other=0.0).to(tl.float32)

        # Accumulate: acc += sum over H-chunk of (x_row * w_block.T)
        # x_row: [BLOCK_S, 1], w_block.T: [BLOCK_H, BLOCK_M] -> reduce over BLOCK_H to [BLOCK_S, BLOCK_M]
        # We need to broadcast x_row along BLOCK_H so it becomes [BLOCK_S, BLOCK_H], then dot with w_block.T
        # Alternatively, do outer multiply and sum: sum over h-axis (axis=1)
        # Outer multiply: [BLOCK_S, BLOCK_H] * [BLOCK_H, BLOCK_M] -> [BLOCK_S, BLOCK_M]
        # Compute per h: x_row[:, None, :] * w_block[None, :, :] then sum over h
        # Here: x_row expanded as x_row[:, None, :] and w_block as w_block[None, :, :]
        # But Triton supports tl.dot for 2D, so we can form x_row_expanded and w_block.T and use tl.dot
        # Build expanded terms:
        # x_row_expanded: [BLOCK_S, BLOCK_H] by repeating x_row across h dim
        # We can't directly expand, so use tl.dot(x_row[:, None, :], w_block.T[None, :, :]) -> not available
        # Instead, compute acc += sum_h (x_row[:, h] * w_block[h, :]) by iterating h in the chunk:
        # This is okay for BLOCK_H not too large (e.g., 64).
        for hi in range(0, BLOCK_H):
            h_idx = h_start + hi
            mask_h_i = h_idx < H
            x_vec = tl.load(x_ptr + b * stride_x_b + s_offsets * stride_x_s + h_idx * stride_x_h, mask=mask_s, other=0.0).to(tl.float32)  # [BLOCK_S]
            w_vec = tl.load(W_ptr + (m_start + tl.arange(0, BLOCK_M)) * stride_w_m + h_idx * stride_w_h, mask=(m_start + tl.arange(0, BLOCK_M)) < M, other=0.0).to(tl.float32)  # [BLOCK_M]
            # acc += x_vec[:, None] * w_vec[None, :] (broadcast), with mask_h_i guarding contribution
            acc += (x_vec[:, None] * w_vec[None, :]) * mask_h_i

    # Add bias: bias[m] across M
    bias_vals = tl.load(bias_ptr + (m_start + tl.arange(0, BLOCK_M)), mask=(m_start + tl.arange(0, BLOCK_M)) < M, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # Store out
    out_ptrs = out_ptr + b * stride_o_b + s_offsets[:, None] * stride_o_s + (m_start + tl.arange(0, BLOCK_M))[None, :] * stride_o_m
    store_mask = mask_s[:, None] & ((m_start + tl.arange(0, BLOCK_M))[None, :] < M)
    tl.store(out_ptrs, acc, mask=store_mask)

# Triton kernel: element-wise gating Bx = B * x_proj over (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr,          # *const T, [B, S, H]
    P_ptr,          # *const T, [B, S, H]
    out_ptr,        # *T, [B, S, H]
    B, S, H,        # int32
    stride_b_b, stride_b_s, stride_b_h,
    stride_p_b, stride_p_s, stride_p_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr
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

    # Load tiles
    B_tile = tl.load(
        B_ptr + b * stride_b_b + s_offsets[:, None] * stride_b_s + h_offsets[None, :] * stride_b_h,
        mask=mask_s[:, None] & mask_h[None, :], other=0.0
    ).to(tl.float32)

    P_tile = tl.load(
        P_ptr + b * stride_p_b + s_offsets[:, None] * stride_p_s + h_offsets[None, :] * stride_p_h,
        mask=mask_s[:, None] & mask_h[None, :], other=0.0
    ).to(tl.float32)

    # Compute Bx = B * P
    out_tile = B_tile * P_tile

    # Store
    out_ptrs = out_ptr + b * stride_o_b + s_offsets[:, None] * stride_o_s + h_offsets[None, :] * stride_o_h
    tl.store(out_ptrs, out_tile, mask=mask_s[:, None] & mask_h[None, :])

# Triton kernel: left-pad along sequence dimension by PAD (here PAD=3) for a [B, H, S] tensor
@triton.jit
def TritonPadLeftKernel(
    in_ptr,         # *const T, [B, H, S]
    out_ptr,        # *T, [B, H, S + PAD]
    B, H, S, PAD,   # int32
    stride_i_b, stride_i_h, stride_i_s,
    stride_o_b, stride_o_h, stride_o_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h

    s_out_start = pid_s * BLOCK_S
    S_out = S + PAD

    # Store zeros for first PAD columns
    for i in range(0, PAD):
        out_ptr_pos = out_ptr + b * stride_o_b + h * stride_o_h + i * stride_o_s
        tl.store(out_ptr_pos, 0.0)

    # Copy from in to out starting at PAD
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            in_ptr_pos = in_ptr + b * stride_i_b + h * stride_i_h + s_in * stride_i_s
            val = tl.load(in_ptr_pos)
            out_ptr_pos = out_ptr + b * stride_o_b + h * stride_o_h + (s_in + PAD) * stride_o_s
            tl.store(out_ptr_pos, val)

# Triton kernel: grouped causal 1D conv (K=4), groups=H
# Input: Bx_padded [B, H, S+3], Weight [H, 1, 4], Bias [H]
# Output: conv_out [B, H, S]
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr,     # *const T, [B, H, S+PAD] (PAD=3)
    weight_ptr,     # *const T, [H, 1, 4] (stride for (h, k))
    bias_ptr,       # *const T, [H]
    out_ptr,        # *T, [B, H, S]
    B, S, H,        # int32
    PAD,            # int32, usually 3
    stride_b_b, stride_b_h, stride_b_s,   # strides for Bx_pad
    stride_w_h, stride_w_k,               # strides for weight (h, k)
    stride_o_b, stride_o_h, stride_o_s,   # strides for out (B, H, S)
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    s_start = pid_s * BLOCK_S

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # K=4 causal convolution
    for k in range(0, 4):
        t_in = s_offsets + k
        valid = (t_in < (S + PAD)) & mask_s
        bx_ptr = Bx_pad_ptr + b * stride_b_b + h * stride_b_h + t_in * stride_b_s
        bx_vals = tl.load(bx_ptr, mask=valid, other=0.0).to(tl.float32)

        w_ptr = weight_ptr + h * stride_w_h + k * stride_w_k  # weight[h, 0, k]
        w_val = tl.load(w_ptr).to(tl.float32)

        acc += bx_vals * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + h).to(tl.float32)
    acc += bias_val

    # Store conv_out[b, h, s_offsets]
    out_ptrs = out_ptr + b * stride_o_b + h * stride_o_h + s_offsets * stride_o_s
    tl.store(out_ptrs, acc, mask=mask_s)

# Triton kernel: final projection out[b, s, h] = sum_h y[b, s, h] * W[h, h] + bias[h]
# y: [B, S, H], out_proj_weight: [H, H] -> we pass W as (H, H)^T => (H, H)
@triton.jit
def TritonFinalProjectionKernel(
    y_ptr,          # *const T, [B, S, H]
    W_ptr,          # *const T, [H, H] (note: we pass out_proj_weight.T)
    bias_ptr,       # *const T, [H]
    out_ptr,        # *T, [B, S, H]
    B, S, H,        # int32
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_h, stride_w_k,  # W is [H, H], k stride along H
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr
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

    # Load y tile [BLOCK_S, BLOCK_H]
    y_ptrs = y_ptr + b * stride_y_b + s_offsets[:, None] * stride_y_s + h_offsets[None, :] * stride_y_h
    y_tile = tl.load(y_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    # Load W tile [BLOCK_H, BLOCK_H]
    W_ptrs = W_ptr + h_offsets[:, None] * stride_w_h + h_offsets[None, :] * stride_w_k
    W_tile = tl.load(W_ptrs, mask=mask_h[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    # Compute out = y_tile @ W_tile (dot-reduce along H)
    # acc shape [BLOCK_S, BLOCK_H]
    acc = tl.zeros([BLOCK_S, BLOCK_H], dtype=tl.float32)
    for i in range(0, BLOCK_H):
        # For each column i of W_tile, take dot with rows of y_tile across H dimension
        # y_tile[:, i] * W_tile[i, :] -> sum over i axis
        # Implement via per-row accumulation:
        w_vec = W_tile[i, :]  # [BLOCK_H]
        y_vec = y_tile[:, i]  # [BLOCK_S]
        # Multiply outer and reduce: we need sum over H dimension. Do explicit sum over BLOCK_H:
        # acc[:, i] = sum_j y_tile[:, j] * W_tile[j, i]
        # Unrolled loop for simplicity:
        for j in range(0, BLOCK_H):
            acc[:, i] += y_tile[:, j] * W_tile[j, i]

    # Add bias
    bias_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # Store output
    out_ptrs = out_ptr + b * stride_o_b + s_offsets[:, None] * stride_o_s + h_offsets[None, :] * stride_o_h
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])

# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-Only fused computation equivalent to the original:
        1) Triple linear projection via Triton (B, C, x_proj)
        2) Element-wise gating Bx = B * x_proj via Triton
        3) Left-pad for causal conv via Triton
        4) Grouped causal 1D conv via Triton (K=4, groups=H)
        5) Final projection via Triton
        """
        # Ensure CUDA and contiguous
        assert x.is_cuda, "Input tensor x must be on CUDA device."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda, "in_proj_weight and in_proj_bias must be CUDA tensors."
        assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be CUDA tensors."
        assert out_proj_weight.is_cuda and out_proj_bias.is_cuda, "out_proj_weight and out_proj_bias must be CUDA tensors."

        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape

        # 1) Triple linear projection (B, C, x_proj)
        # We'll allocate outputs in the same dtype as input x.
        B_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        C_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        x_proj_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)

        # Launch TritonLinearProjectionKernel three times (for each of the three projections).
        # BLOCK sizes: S tiles 128, H tiles 64, H accumulation tiles 64
        TritonLinearProjectionKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, in_proj_weight[:H, :], in_proj_bias[:H], B_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[:H, :].stride(0), in_proj_weight[:H, :].stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=128, BLOCK_M=64, BLOCK_H=64
        )

        TritonLinearProjectionKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, in_proj_weight[H:2*H, :], in_proj_bias[H:2*H], C_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[H:2*H, :].stride(0), in_proj_weight[H:2*H, :].stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=128, BLOCK_M=64, BLOCK_H=64
        )

        TritonLinearProjectionKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, in_proj_weight[2*H:3*H, :], in_proj_bias[2*H:3*H], x_proj_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight[2*H:3*H, :].stride(0), in_proj_weight[2*H:3*H, :].stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            BLOCK_S=128, BLOCK_M=64, BLOCK_H=64
        )

        # 2) Element-wise gating: Bx = B * x_proj via Triton
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            B_out, x_proj_out, Bx,
            B, S, H,
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=128, BLOCK_H=64
        )

        # 3) Left-pad Bx by 3 for causal conv
        Bx_padded = torch.empty((B, H, S + 3), dtype=x.dtype, device=x.device)
        TritonPadLeftKernel[(B, H, triton.cdiv(S, 128))](
            Bx, Bx_padded,
            B, H, S, 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            BLOCK_S=128
        )

        # 4) Grouped causal 1D conv (K=4), groups=H via Triton
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        TritonGroupedCausalConvKernel[(B, H, triton.cdiv(S, 128))](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, S, H, 3,  # PAD=3
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(2),  # stride along h, k
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128, BLOCK_H=64
        )

        # 5) Output gating: y = C * conv_out
        y = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            C_out, conv_out, y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_S=128, BLOCK_H=64
        )

        # 6) Final output projection via Triton
        # We need W = out_proj_weight.T (H, H). Pass as (H, H)^T -> (H, H)
        out_proj_weight_T = out_proj_weight.transpose(0, 1).contiguous()
        output = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonFinalProjectionKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            y, out_proj_weight_T, out_proj_bias, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight_T.stride(0), out_proj_weight_T.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_S=128, BLOCK_H=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
