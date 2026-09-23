import torch
import triton
import triton.language as tl


# 1) Linear projection: out[B, S, M] = x @ W[:M, :].T + bias[:M]
# Shapes:
#   x: (B, S, H) contiguous, strides: (S*H, H, 1)
#   W: (M, H) contiguous, strides: (H, 1)
#   bias: (M) contiguous
#   out: (B, S, M) contiguous, strides: (S*M, M, 1)
@triton.jit
def TritonLinearKernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,     # strides for x
    stride_w_m, stride_w_h,                 # strides for w
    stride_out_b, stride_out_s, stride_out_m,  # strides for out
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m = pid_m

    acc = tl.zeros([BLOCK_S, 1], dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # X[b, s, h] with strides
        x_ptrs = x_ptr + b * stride_x_b + s_offsets[:, None] * stride_x_s + h_offsets[None, :] * stride_x_h
        x_tile = tl.load(x_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)  # [BLOCK_S, BLOCK_H]

        # W[m, h] vector over h
        w_ptrs = w_ptr + m * stride_w_m + h_offsets * stride_w_h
        w_tile = tl.load(w_ptrs, mask=mask_h, other=0.0).to(tl.float32)  # [BLOCK_H]
        w_tile = w_tile[None, :]  # broadcast across s

        acc += tl.dot(x_tile, w_tile)

    bias_val = tl.load(bias_ptr + m).to(tl.float32)
    acc += bias_val  # broadcast over s

    out_ptrs = out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m * stride_out_m
    tl.store(out_ptrs, acc, mask=(s_offsets < S)[:, None])


# 2) Element-wise gating: Bx = B * x_proj
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
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

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# 3) Left-pad along sequence by PAD=3: Bx_pad[B, H, S + PAD]
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
    pid_sp = tl.program_id(2)  # tiles over S_out = S + PAD

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_out_start = pid_sp * BLOCK_S

    # write zeros at the first PAD columns
    for i in range(0, PAD):
        tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s, 0.0)

    # copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s).to(tl.float32)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


# 4) Grouped causal 1D convolution (groups=H, kernel_size=4) with conv_weight (H, 1, 4), bias (H)
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD,  # sizes
    stride_bx_b, stride_bx_h, stride_bx_s,  # strides for Bx_pad
    stride_out_b, stride_out_h, stride_out_s,  # strides for out
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    s_start = pid_s * BLOCK_S
    S_out = S + PAD

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for k in range(0, 4):
        s_out = s_start + tl.arange(0, BLOCK_S)
        mask = (s_out < S_out) & (s_out >= PAD)
        b_ptrs = Bx_pad_ptr + b * stride_bx_b + h * stride_bx_h + s_out * stride_bx_s
        b_vals = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)
        w_ptrs = conv_weight_ptr + h * stride_out_h + k * stride_out_s  # conv_weight[h, 0, k]
        # conv_weight is (H, 1, 4) with strides (1, 4, 1). We assume conv_weight_ptr layout with strides (stride_w_h, stride_w_k):
        # conv_weight[h, 0, k] -> conv_weight_ptr + h*stride_w_h + k*stride_w_k
        # Here we pass conv_weight_ptr from host, using strides (H, 4). Adjust: conv_weight_ptr is contiguous, so stride_w_h = 4, stride_w_k = 1.
        # Note: The original conv_weight in PyTorch is (H, 1, 4). In our kernel, we pass a pointer where conv_weight[h, 0, k] is at offset h*4 + k.
        # To keep this robust, we restructure: conv_weight_ptr should be laid out as [H, 4] contiguous, so conv_weight_ptr[h*4 + k] = W[h,0,k].
        # However, PyTorch's conv_weight shape is (H, 1, 4) with strides (1,4,1). In our Triton invocation, we pass a flat pointer and use strides:
        # We need to pass conv_weight with strides (H, 4) to access W[h, 0, k] at offset h*H + k. But since H and 4 are independent dims, we use:
        # conv_weight_ptr is contiguous of length H*4. Element W[h, 0, k] is at index h*4 + k.
        w_val = tl.load(conv_weight_ptr + h * 4 + k).to(tl.float32)
        acc += b_vals * w_val

    bias_val = tl.load(conv_bias_ptr + h).to(tl.float32)
    acc += bias_val

    out_ptrs = out_ptr + b * stride_out_b + h * stride_out_h + (s_start + tl.arange(0, BLOCK_S)) * stride_out_s
    tl.store(out_ptrs, acc, mask=(s_start + tl.arange(0, BLOCK_S) < S))


# 5) Elementwise multiply: Y = C * conv_out (C: [B,S,H], conv_out: [B,H,S])
@triton.jit
def TritonMulKernel(
    C_ptr, conv_out_ptr, Y_ptr,
    B, S, H,
    stride_c_b, stride_c_s, stride_c_h,
    stride_co_b, stride_co_h, stride_co_s,
    stride_y_b, stride_y_s, stride_y_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    c_ptrs = C_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    co_ptrs = conv_out_ptr + pid_b * (H * S) + h_offsets[:, None] * H + s_offsets[None, :]

    c_vals = tl.load(c_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    co_vals = tl.load(co_ptrs, mask=mask_s[None, :] & mask_h[:, None], other=0.0).to(tl.float32)

    y_vals = c_vals * co_vals

    y_ptrs = Y_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(y_ptrs, y_vals, mask=mask_s[:, None] & mask_h[None, :])


# 6) Final projection: OUT[b, s, m] = sum_h Y[b, s, h] * W[m, h] + bias[m]
@triton.jit
def TritonLinearFinalKernel(
    Y_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_m,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m = pid_m

    acc = tl.zeros([BLOCK_S, 1], dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        y_ptrs = Y_ptr + b * stride_y_b + s_offsets[:, None] * stride_y_s + h_offsets[None, :] * stride_y_h
        y_tile = tl.load(y_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)  # [BLOCK_S, BLOCK_H]

        w_ptrs = w_ptr + m * stride_w_m + h_offsets * stride_w_h
        w_tile = tl.load(w_ptrs, mask=mask_h, other=0.0).to(tl.float32)  # [BLOCK_H]
        w_tile = w_tile[None, :]  # broadcast over s

        acc += tl.dot(y_tile, w_tile)

    bias_val = tl.load(bias_ptr + m).to(tl.float32)
    acc += bias_val  # broadcast over s

    out_ptrs = out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m * stride_out_m
    tl.store(out_ptrs, acc, mask=(s_offsets < S)[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # x: [B, S, H]
        B, S, H = x.shape
        device = x.device
        dtype = x.dtype

        # 1) Three linear projections via Triton
        # First group: m=0..H-1
        out1 = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_linear1 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid_linear1](
            x, in_proj_weight[:H, :], in_proj_bias[:H], out1,
            B, S, H, H,
            S*H, H, 1,
            H, 1,
            B*S*H, S, H,
            128, 64, 64
        )

        # Second group: m=H..2H-1
        out2 = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_linear2 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid_linear2](
            x, in_proj_weight[H:2*H, :], in_proj_bias[H:2*H], out2,
            B, S, H, H,
            S*H, H, 1,
            H, 1,
            B*S*H, S, H,
            128, 64, 64
        )

        # Third group: m=2H..3H-1
        out3 = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_linear3 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid_linear3](
            x, in_proj_weight[2*H:3*H, :], in_proj_bias[2*H:3*H], out3,
            B, S, H, H,
            S*H, H, 1,
            H, 1,
            B*S*H, S, H,
            128, 64, 64
        )

        # 2) Element-wise gating Bx = out1 * out3
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_gate = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonGateKernel[grid_gate](
            out1, out3, Bx,
            B, S, H,
            128, 64
        )

        # 3) Left-pad along sequence by PAD=3
        Bx_pad = torch.empty((B, H, S + 3), device=device, dtype=torch.float32)
        grid_pad = (B, H, triton.cdiv(S + 3, 128))
        TritonPadLeftKernel[grid_pad](
            Bx, Bx_pad,
            B, H, S, 3,
            B*H, H, 1,
            B*H, H, (S+3),
            128
        )

        # 4) Grouped causal 1D conv (groups=H) with conv_weight (H, 1, 4), bias (H)
        # conv_weight in Triton: treat it as [H, 4] contiguous. We pass conv_weight.view(-1) and adjust strides.
        conv_weight_flat = conv_weight.view(-1).contiguous()  # shape [H*4]
        conv_bias_flat = conv_bias
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_conv = (B, H, triton.cdiv(S, 128))
        TritonGroupedCausalConvKernel[grid_conv](
            Bx_pad, conv_weight_flat, conv_bias_flat, conv_out,
            B, H, S, 3,
            B*H, H, 1,
            B*H, H, S,
            128
        )

        # 5) Output gating: y = C * conv_out
        y = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_mul = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonMulKernel[grid_mul](
            out2, conv_out, y,
            B, S, H,
            B*S*H, S, H,
            H*S, H, S,
            B*S*H, S, H,
            128, 64
        )

        # 6) Final projection: F.linear(y, out_proj_weight, out_proj_bias) in Triton
        final_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_final = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearFinalKernel[grid_final](
            y, out_proj_weight, out_proj_bias, final_out,
            B, S, H, H,
            B*S*H, S, H,
            H, 1,
            B*S*H, S, H,
            128, 64, 64
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
