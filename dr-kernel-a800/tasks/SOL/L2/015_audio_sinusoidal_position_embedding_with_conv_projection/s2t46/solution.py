import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton 2D conv kernel: stride=2, padding=1, 3x3
# Inputs:
#   X: [B, C_in, IH, IW], contiguous
#   W: [OC, C_in, 3, 3], contiguous
#   BIAS: [OC], contiguous
# Output:
#   Y: [B, OC, OH, OW], contiguous
@triton.jit
def conv2d_stride2_3x3_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, IH, IW, OC, OH, OW,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_c, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros([1], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C_in):
        # Loop over 3x3 kernel taps with padding=1 and stride=2
        for kh in range(0, 3):
            ih = 2 * oh + kh - 1
            valid_h = (ih >= 0) & (ih < IH)
            for kw in range(0, 3):
                iw = 2 * ow + kw - 1
                valid_w = (iw >= 0) & (iw < IW)
                in_bounds = valid_h & valid_w
                # Load input scalar with mask
                x_ptr_elem = X_ptr + b * x_stride_b + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr_elem, mask=in_bounds, other=0.0)
                # Load weight scalar
                w_ptr_elem = W_ptr + oc * w_stride_oc + ci * w_stride_c + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr_elem)
                # Accumulate
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + oc)
    acc += bias_val

    # Store output
    y_ptr_elem = Y_ptr + b * y_stride_b + oc * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr_elem, acc)


# Triton GELU kernel (tanh approximation) on 1D flattened tensor
# y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/pi) * (x[i] + 0.044715*x[i]^3)))
@triton.jit
def gelu_1d_kernel(
    X_ptr, Y_ptr, N, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton kernel for final linear projection and positional embedding
# Input X_flat: [B * T * N], W: [M=1024, N], pos: [T * M], Output Y_flat: [B * T * M]
# For each (b, t), compute Y[b, t, m] = sum_n X[b, t, n] * W[m, n], then scale by 32.0 and add pos[t, :].
@triton.jit
def linear_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T, N, M, scale: tl.float32,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    base_x = b * (T * N)
    base_y = b * (T * M)
    # Loop over m in tiles
    for m0 in range(0, M, 128):
        m_offsets = m0 + tl.arange(0, 128)
        mask_m = m_offsets < M
        acc = tl.zeros([128], dtype=tl.float32)
        # Loop over n in tiles
        for n0 in range(0, N, 256):
            n_offsets = n0 + tl.arange(0, 256)
            mask_n = n_offsets < N
            x_vals = tl.load(X_ptr + base_x + t * N + n_offsets, mask=mask_n, other=0.0)  # [256]
            w_vals = tl.load(W_ptr + m_offsets[:, None] * N + n_offsets[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)  # [128, 256]
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        # scale and add pos
        acc = acc * scale
        pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
        acc = acc + pos_vec
        # store
        tl.store(Y_ptr + base_y + t * M + m_offsets, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), bfloat16 or float16
        conv weights: (OC, C_in, 3, 3), bfloat16/float16
        biases: (OC), bfloat16/float16
        conv_out_weight: (d_model=1024, N=3840), bfloat16/float16
        positional_embedding: (1500, 1024), bfloat16/float16
        embed_scale: float
        Returns: (B, T_out, d_model) where T_out is output time dimension after 3 convs.
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        assert input_features.is_cuda and conv2d1_weight.is_cuda and conv_out_weight.is_cuda, "Tensors must be on CUDA"

        # Stage 1: Conv2d (1 -> 384 channels), stride=2, padding=1, 3x3
        B, _, IH, IW = input_features.shape  # B, 1, 80, T
        C_in1 = 1
        C_out1 = 384
        T1 = IW  # spatial width becomes time after conv
        OH1 = (IH - 1) // 2 + 1  # 40
        OW1 = (T1 + 1) // 2      # depends on T

        x1 = torch.empty((B, C_out1, OH1, OW1), device=input_features.device, dtype=torch.float32)
        grid1 = (B, C_out1, OH1, OW1)
        conv2d_stride2_3x3_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C_in1, IH, IW, C_out1, OH1, OW1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # GELU after conv1
        N1 = x1.numel()
        x1_flat = x1.view(-1).contiguous()
        y1_flat = torch.empty_like(x1_flat, dtype=torch.float32)
        BLOCK = 1024
        grid_g1 = (triton.cdiv(N1, BLOCK),)
        gelu_1d_kernel[grid_g1](x1_flat, y1_flat, N1, BLOCK)
        x1 = y1_flat.view_as(x1)

        # Stage 2: Conv2d (384 -> 384), stride=2, padding=1, 3x3
        C_in2 = C_out1
        C_out2 = 384
        T2 = OW1
        OH2 = (OH1 - 1) // 2 + 1  # 20
        OW2 = (T2 + 1) // 2       # after second conv

        x2 = torch.empty((B, C_out2, OH2, OW2), device=input_features.device, dtype=torch.float32)

        w2_strides = conv2d2_weight.stride()

        grid2 = (B, C_out2, OH2, OW2)
        conv2d_stride2_3x3_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_in2, OH1, T2, C_out2, OH2, OW2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            w2_strides[0], w2_strides[1], w2_strides[2], w2_strides[3],
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # GELU after conv2
        N2 = x2.numel()
        x2_flat = x2.view(-1).contiguous()
        y2_flat = torch.empty_like(x2_flat, dtype=torch.float32)
        grid_g2 = (triton.cdiv(N2, BLOCK),)
        gelu_1d_kernel[grid_g2](x2_flat, y2_flat, N2, BLOCK)
        x2 = y2_flat.view_as(x2)

        # Stage 3: Conv2d (384 -> 384), stride=2, padding=1, 3x3
        C_in3 = C_out2
        C_out3 = 384
        T3 = OW2
        OH3 = (OH2 - 1) // 2 + 1  # 10
        OW3 = (T3 + 1) // 2       # final spatial width (time_after_conv)

        x3 = torch.empty((B, C_out3, OH3, OW3), device=input_features.device, dtype=torch.float32)

        w3_strides = conv2d3_weight.stride()

        grid3 = (B, C_out3, OH3, OW3)
        conv2d_stride2_3x3_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_in3, OH2, T3, C_out3, OH3, OW3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w3_strides[0], w3_strides[1], w3_strides[2], w3_strides[3],
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # GELU after conv3
        N3 = x3.numel()
        x3_flat = x3.view(-1).contiguous()
        y3_flat = torch.empty_like(x3_flat, dtype=torch.float32)
        grid_g3 = (triton.cdiv(N3, BLOCK),)
        gelu_1d_kernel[grid_g3](y3_flat, x3_flat, N3, BLOCK)  # write result back into y3_flat buffer
        # Note: we'll compute final output in a separate kernel; keep x3_flat for linear

        # Final linear projection and positional embedding
        # Reshape to (B, T, N) where N = C_out3 * 10 = 3840
        N = C_out3 * 10  # 3840
        # Prepare X_flat: (B*T*N)
        B_T = B * OW3  # B * time_after_conv
        X_flat = torch.empty(B_T * N, device=input_features.device, dtype=torch.float32)
        # Fill X_flat using x3.view(B, T, -1) flattening. We need x3_flat already computed:
        # We have y3_flat; we can use x3.view(B, -1) directly, but since x3 is fp32 and we applied gelu,
        # we just reuse x3 as source for projection. However, gelu was applied to x3 in fp32; to keep
        # it simple and correct, we recompute from x3 without gelu for projection step:
        # Since gelu was done separately, the latest x3_flat already holds post-conv + GELU.
        # We'll use x3_flat to form X_flat:
        x3_flat = x3.view(B, -1).contiguous()  # (B, T*N)
        X_flat = x3_flat.view(-1).contiguous()  # (B*T*N)

        M = conv_out_weight.shape[0]  # 1024
        # Output buffer (B*T*M)
        Y_flat = torch.empty(B_T * M, device=input_features.device, dtype=torch.float32)

        # Launch linear+pos kernel: we need T as OW3
        T = OW3
        grid_lin = (B, T)
        linear_pos_kernel[grid_lin](
            X_flat, conv_out_weight, positional_embedding, Y_flat,
            B, T, N, M, float(embed_scale),
        )

        # Reshape to (B, T, M)
        output = Y_flat.view(B, T, M)
        return output


def run(*args):
    return ModelNew()(*args)
