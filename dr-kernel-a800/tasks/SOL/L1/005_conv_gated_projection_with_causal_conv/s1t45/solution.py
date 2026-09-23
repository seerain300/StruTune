import torch
import triton
import triton.language as tl


# Triton kernel: F.linear-like per M in {H, H, H}
# Computes out[B, S, M] where out[b, s, m] = sum_i x[b, s, i] * W[m, i] + bias[m]
@triton.jit
def TritonLinearKernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    Bsz, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # grid: (Bsz, ceil_div(S, BLOCK_S), ceil_div(M, BLOCK_H))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduce over H (input feature dimension)
    for h_idx in range(0, H):
        x_ptrs = x_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + h_idx * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)  # [BLOCK_S, 1]
        w_ptrs = w_ptr + m_offsets[None, :] * stride_w_m + h_idx * stride_w_h      # [1, BLOCK_M]
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)   # [1, BLOCK_M]
        # broadcast multiply: [BLOCK_S, 1] * [1, BLOCK_M] -> [BLOCK_S, BLOCK_M]
        acc += x_vals * w_vals

    # add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)  # [BLOCK_M]
    acc += bias_vals[None, :]  # broadcast over S

    # store
    out_ptrs = out_ptr + pid_b * stride_o_b + s_offsets[:, None] * stride_o_s + m_offsets[None, :] * stride_o_h
    store_mask = mask_s[:, None] & mask_m[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


# Triton kernel: element-wise gating Bx = B * x_proj over (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    b_ptrs = B_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: left-pad along sequence by PAD for Bx, producing Bx_pad[B, H, S + PAD]
# Inputs: Bx [B, H, S], output: out_pad [B, H, S + PAD], PAD=3
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sp = tl.program_id(2)  # tiling over S_out = S + PAD

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_out_start = pid_sp * BLOCK_S

    # write zeros at the first PAD columns
    for i in range(0, PAD):
        out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s
        tl.store(out_ptr_pos, 0.0)

    # copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


# Triton kernel: grouped causal 1D convolution with groups=H, kernel_size=4
# Input: Bx_pad of shape (B, H, S+PAD), weight (H, 1, 4), bias (H)
# Output: conv_out of shape (B, H, S)
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD, K,  # K=4
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k,  # conv_weight strides for (H, 1, 4): w_h, w_k
    stride_o_b, stride_o_h, stride_o_s,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_start = pid_s * BLOCK_S
    S_out = S + PAD  # length of padded input along S

    # Accumulator for conv_out[b, h, s] across s in this tile
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Load conv bias for this h
    bias_val = tl.load(conv_bias_ptr + pid_h)

    # For each k in {0..K-1}, accumulate sum_{s in tile} Bx_pad[b, h, s + k] * conv_weight[h, 0, k]
    for k_idx in range(0, K):
        w_val = tl.load(conv_weight_ptr + pid_h * stride_w_h + k_idx * stride_w_k)
        # Loop over the tile of s
        for i in range(0, BLOCK_S):
            s_out = s_start + i
            s_in = s_out - PAD  # original s in input
            # validity: s_in must be in [0, S_out) and within tile bounds
            valid = (s_in >= 0) & (s_in < S_out)
            val = tl.load(Bx_pad_ptr + pid_b * stride_bx_b + pid_h * stride_bx_h + s_in * stride_bx_s, mask=valid, other=0.0)
            acc[i] += val * w_val

    # add bias
    acc += bias_val

    # store
    out_ptrs = out_ptr + pid_b * stride_o_b + pid_h * stride_o_h + (s_start + tl.arange(0, BLOCK_S)) * stride_o_s
    store_mask = (s_start + tl.arange(0, BLOCK_S)) < S
    tl.store(out_ptrs, acc, mask=store_mask)


# Triton kernel: final linear projection over H dimension
# out[B, S, H] = y[B, S, :H] @ out_proj_weight[:H, :]^T + out_proj_bias[:H]
@triton.jit
def TritonLinearFinalKernel(
    y_ptr, w_ptr, bias_ptr, out_ptr,
    Bsz, S, M, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_m, stride_w_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduce over H (feature dim of y)
    for h_idx in range(0, H):
        y_ptrs = y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + h_idx * stride_y_h  # [BLOCK_S, 1]
        y_vals = tl.load(y_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)
        w_ptrs = w_ptr + m_offsets[None, :] * stride_w_m + h_idx * stride_w_h                      # [1, BLOCK_M]
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)
        acc += y_vals * w_vals

    # add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)  # [BLOCK_M]
    acc += bias_vals[None, :]

    # store
    out_ptrs = out_ptr + pid_b * stride_o_b + s_offsets[:, None] * stride_o_s + m_offsets[None, :] * stride_o_h
    store_mask = mask_s[:, None] & mask_m[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # These are dummy placeholders; the run function will pass actual tensors.
        # We keep them here only to match the signature expected by the evaluator.

    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-based fused implementation that performs:
          1) Three in-projection linear ops: (B, S, H) -> (B, S, H)
          2) Element-wise gating: Bx = B * x_proj
          3) Grouped causal conv with kernel_size=4, groups=H: (B, H, S)
          4) Output gating: y = C * conv_out
          5) Final out projection: (B, S, H)
        """
        B, S, H = x.shape
        assert H == self.hidden_size

        # 1) Three linear projections using TritonLinearKernel
        # Prepare outputs
        B_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        C_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        x_proj_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)

        # Launch TritonLinearKernel three times with different W and bias slices
        grid_linear = (B, triton.cdiv(S, 128), triton.cdiv(H, 128))  # 128x128 tiles for H and S

        # B = x @ in_proj_weight[:H, :].T + in_proj_bias[:H]
        TritonLinearKernel[grid_linear](
            x, in_proj_weight[:H, :], in_proj_bias[:H], B_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            128, 128,
        )

        # C = x @ in_proj_weight[H:2H, :].T + in_proj_bias[H:2H]
        TritonLinearKernel[grid_linear](
            x, in_proj_weight[H:2 * H, :], in_proj_bias[H:2 * H], C_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            128, 128,
        )

        # x_proj = x @ in_proj_weight[2H:3H, :].T + in_proj_bias[2H:3H]
        TritonLinearKernel[grid_linear](
            x, in_proj_weight[2 * H:3 * H, :], in_proj_bias[2 * H:3 * H], x_proj_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            x_proj_out.stride(0), x_proj_out.stride(1), x_proj_out.stride(2),
            128, 128,
        )

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 128))](
            B_out, x_proj_out, Bx,
            B, S, H,
            128, 128,
        )

        # 3) Pad Bx along S by PAD=3 for causal conv
        Bx_pad = torch.empty((B, H, S + 3), dtype=x.dtype, device=x.device)
        TritonPadLeftKernel[(B, H, triton.cdiv(S + 3, 128))](
            Bx, Bx_pad,
            B, H, S, 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            128,
        )

        # 4) Grouped causal 1D convolution with groups=H, kernel_size=4
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        # Strides for conv_weight: (H, 1, 4)
        stride_w_h, stride_w_k = conv_weight.stride(0), conv_weight.stride(2)  # conv_weight is (H, 1, 4)
        TritonGroupedCausalConvKernel[(B, H, triton.cdiv(S, 128))](
            Bx_pad, conv_weight, conv_bias, conv_out,
            B, H, S, 3, 4,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            stride_w_h, stride_w_k,
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            128,
        )

        # 5) Output gating: y = C * conv_out (elementwise), shape (B, S, H)
        y = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        # To ensure elementwise multiplication, we need C: (B, S, H) and conv_out: (B, H, S).
        # We transpose conv_out to (B, S, H) for broadcast multiply. This matches PyTorch's broadcasting behavior.
        conv_out_T = conv_out.transpose(1, 2).contiguous()  # (B, S, H)
        TritonGateKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 128))](
            C_out, conv_out_T, y,
            B, S, H,
            128, 128,
        )

        # 6) Final projection: F.linear(y, out_proj_weight, out_proj_bias) -> (B, S, H)
        out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearFinalKernel[(B, triton.cdiv(S, 128), triton.cdiv(H, 128))](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            128, 128,
        )

        return out


def run(*args):
    return ModelNew()(*args)
