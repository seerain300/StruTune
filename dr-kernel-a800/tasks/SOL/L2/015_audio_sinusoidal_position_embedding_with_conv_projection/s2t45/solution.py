import math
import torch
import torch.nn.functional as F

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    # 2D conv: stride=2, padding=1, 3x3. Grid = (B, C_out, OH, OW)
    @triton.jit
    def conv2d_stride2_3x3_kernel(
        x_ptr,         # *const float, input [B, C_in, IH, IW]
        w_ptr,         # *const float, weight [C_out, C_in, 3, 3]
        bias_ptr,      # *const float, bias [C_out]
        y_ptr,         # *float, output [B, C_out, OH, OW]
        B, C_out, C_in, IH, IW, OH, OW,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
        y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    ):
        b = tl.program_id(0)
        co = tl.program_id(1)
        oh = tl.program_id(2)
        ow = tl.program_id(3)

        acc = tl.zeros((), dtype=tl.float32)

        # Loop over input channels and 3x3 taps
        for ci in range(0, C_in):
            for kh in range(0, 3):
                ih = 2 * oh + kh - 1  # stride=2, padding=1
                valid_h = (ih >= 0) & (ih < IH)
                for kw in range(0, 3):
                    iw = 2 * ow + kw - 1
                    valid_w = (iw >= 0) & (iw < IW)
                    valid = valid_h & valid_w
                    if valid:
                        x_off = b * x_stride_b + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                        x_val = tl.load(x_ptr + x_off)
                        w_off = co * w_stride_co + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                        w_val = tl.load(w_ptr + w_off)
                        acc += x_val * w_val

        # Add bias
        bias_val = tl.load(bias_ptr + co)
        acc = acc + bias_val

        # Store output
        y_off = b * y_stride_b + co * y_stride_c + oh * y_stride_h + ow * y_stride_w
        tl.store(y_ptr + y_off, acc)


    # GELU (tanh approximation) over 1D flattened tensor
    @triton.jit
    def gelu_1d_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # tanh approximation constants
        c = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        inner = c * (x + 0.044715 * x3)
        t = tl.tanh(inner)
        y = 0.5 * x * (1.0 + t)
        tl.store(y_ptr + offsets, y, mask=mask)


    # Final linear projection + positional embedding
    # X: [B, T, N], W: [M, N], pos: [T, M]
    # Y[b, t, m] = sum_n X[b, t, n] * W[m, n], scale, add pos[t, m]
    @triton.jit
    def linear_project_pos_kernel(
        X_ptr,     # *const float, shape [B, T, N]
        W_ptr,     # *const float, shape [M, N]
        pos_ptr,   # *const float, shape [T, M]
        Y_ptr,     # *float, shape [B, T, M]
        B, T, N, M, scale: tl.float32,
        X_stride_b, X_stride_t, X_stride_n,
        W_stride_m, W_stride_n,
        Y_stride_b, Y_stride_t, Y_stride_m,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1)
        # Tile over m
        for m0 in range(0, M, 128):
            m_offsets = m0 + tl.arange(0, 128)
            mask_m = m_offsets < M
            acc = tl.zeros([128], dtype=tl.float32)
            # Tile over n
            for n0 in range(0, N, 256):
                n_offsets = n0 + tl.arange(0, 256)
                mask_n = n_offsets < N
                x_ptrs = X_ptr + b * X_stride_b + t * X_stride_t + n_offsets * X_stride_n
                x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)  # [256]
                w_ptrs = W_ptr + m_offsets[:, None] * W_stride_m + n_offsets[None, :] * W_stride_n
                w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)  # [128, 256]
                acc += tl.sum(w_vals * x_vals[None, :], axis=1)
            acc = acc * scale
            pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
            acc = acc + pos_vec
            y_ptrs = Y_ptr + b * Y_stride_b + t * Y_stride_t + m_offsets * Y_stride_m
            tl.store(y_ptrs, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T_in), dtype can be bfloat16; we compute in fp32
        conv weights/bias: bfloat16, (OC, C_in, 3,3), (OC)
        conv_out_weight: (1024, N) where N=384*10=3840
        positional_embedding: (1500, 1024)
        embed_scale: float, sqrt(1024) = 32.0
        Output: (B, T_out, 1024) where T_out = (T_in+1)//2 after 1st conv, then //2 twice.
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        B = input_features.shape[0]
        IH = input_features.shape[2]
        IW = input_features.shape[3]

        # Stage 1: Conv2d (1 -> 384) stride=2, padding=1, 3x3
        C_in1 = 1
        C_out1 = 384
        T_in = IW  # original time length
        OH1 = (IH - 1) // 2 + 1  # = 40
        OW1 = (T_in + 1) // 2    # time dimension after first conv

        x1 = torch.empty((B, C_out1, OH1, OW1), device=input_features.device, dtype=torch.float32)

        x1_strides = input_features.stride()
        w1_strides = conv2d1_weight.stride()
        y1_strides = x1.stride()

        grid1 = (B, C_out1, OH1, OW1)
        conv2d_stride2_3x3_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C_out1, C_in1, IH, IW, OH1, OW1,
            x1_strides[0], x1_strides[1], x1_strides[2], x1_strides[3],
            w1_strides[0], w1_strides[1], w1_strides[2], w1_strides[3],
            y1_strides[0], y1_strides[1], y1_strides[2], y1_strides[3],
        )

        # GELU after conv1
        N1 = x1.numel()
        x1_flat = x1.view(-1).contiguous()
        y1_flat = torch.empty_like(x1_flat, dtype=torch.float32)
        BLOCK = 1024
        grid_g1 = (triton.cdiv(N1, BLOCK),)
        gelu_1d_kernel[grid_g1](x1_flat, y1_flat, N1, BLOCK)
        x1 = y1_flat.view_as(x1)

        # Stage 2: Conv2d (384 -> 384) stride=2, padding=1, 3x3
        C_in2 = C_out1
        C_out2 = 384
        T2 = OW1  # time dimension coming from conv1 output (spatial width)
        OH2 = (OH1 - 1) // 2 + 1  # = 20
        OW2 = (T2 + 1) // 2       # after second conv

        x2 = torch.empty((B, C_out2, OH2, OW2), device=input_features.device, dtype=torch.float32)

        y2_strides = x2.stride()
        w2_strides = conv2d2_weight.stride()

        grid2 = (B, C_out2, OH2, OW2)
        conv2d_stride2_3x3_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_out2, C_in2, OH1, T2, OH2, OW2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            w2_strides[0], w2_strides[1], w2_strides[2], w2_strides[3],
            y2_strides[0], y2_strides[1], y2_strides[2], y2_strides[3],
        )

        # GELU after conv2
        N2 = x2.numel()
        x2_flat = x2.view(-1).contiguous()
        y2_flat = torch.empty_like(x2_flat, dtype=torch.float32)
        grid_g2 = (triton.cdiv(N2, BLOCK),)
        gelu_1d_kernel[grid_g2](x2_flat, y2_flat, N2, BLOCK)
        x2 = y2_flat.view_as(x2)

        # Stage 3: Conv2d (384 -> 384) stride=2, padding=1, 3x3
        C_in3 = C_out2
        C_out3 = 384
        T3 = OW2  # time dimension coming from conv2 output (spatial width)
        OH3 = (OH2 - 1) // 2 + 1  # = 10
        OW3 = (T3 + 1) // 2       # final spatial width (time_after_conv)

        x3 = torch.empty((B, C_out3, OH3, OW3), device=input_features.device, dtype=torch.float32)

        y3_strides = x3.stride()
        w3_strides = conv2d3_weight.stride()

        grid3 = (B, C_out3, OH3, OW3)
        conv2d_stride2_3x3_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_out3, C_in3, OH2, T3, OH3, OW3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w3_strides[0], w3_strides[1], w3_strides[2], w3_strides[3],
            y3_strides[0], y3_strides[1], y3_strides[2], y3_strides[3],
        )

        # GELU after conv3
        N3 = x3.numel()
        x3_flat = x3.view(-1).contiguous()


def run(*args):
    return ModelNew()(*args)
