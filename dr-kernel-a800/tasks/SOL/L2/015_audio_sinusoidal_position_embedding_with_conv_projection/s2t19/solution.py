import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: 2D conv stride=2, padding=1, 3x3, arbitrary C_in, OC
# X: (B, C_in, IH, IW) float32, W: (OC, C_in, 3, 3) float32, BIAS: (OC) float32, Y: (B, OC, OH, OW) float32
@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, IH, IW, OC,
    OH, OW,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)
    acc = tl.zeros((), dtype=tl.float32)

    for ic in range(0, C_in):
        for kh in range(0, 3):
            ih = 2 * oh + kh - 1  # stride=2, padding=1
            for kw in range(0, 3):
                iw = 2 * ow + kw - 1
                in_bounds = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                # Compute input offset: ((b * C_in + ic) * IH + ih) * IW + iw
                x_idx = ((b * C_in + ic) * IH + ih) * IW + iw
                x_val = tl.load(X_ptr + x_idx, mask=in_bounds, other=0.0)
                # Load weight scalar w[oc, ic, kh, kw]
                w_idx = oc * (C_in * 9) + ic * 9 + kh * 3 + kw
                w_val = tl.load(W_ptr + w_idx)
                acc += x_val * w_val

    # Add bias
    b_idx = oc
    bias_val = tl.load(BIAS_ptr + b_idx)
    acc += bias_val

    # Store to Y[b, oc, oh, ow]
    y_idx = ((b * OC + oc) * OH + oh) * OW + ow
    tl.store(Y_ptr + y_idx, acc)


# Triton GELU (tanh approximation) over 1D flattened tensor
@triton.jit
def gelu_kernel_1d(X_ptr, Y_ptr, N, scale):
    # Each program handles a chunk of size 1024
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(Y_ptr + offsets, y, mask=mask)


# Triton kernel: linear projection + positional embedding
# X: (B, T_after, N) float32, W: (M=1024, N) float32, POS: (T_after, M) float32, Y: (B, T_after, M) float32
@triton.jit
def linear_pos_kernel(
    X_ptr, W_ptr, POS_ptr, Y_ptr,
    B, T_after, N, M, scale
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    m = tl.program_id(2)
    # Accumulate across N in chunks
    acc = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, N, 256):
        n_offsets = n0 + tl.arange(0, 256)
        mask_n = n_offsets < N
        x_vals = tl.load(X_ptr + b * (T_after * N) + t * N + n_offsets, mask=mask_n, other=0.0)  # [256]
        w_ptrs = W_ptr + m * N + n_offsets
        w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)  # [256]
        acc += tl.sum(w_vals * x_vals, axis=0)
    acc = acc * scale
    pos_vals = tl.load(POS_ptr + t * M + m)
    acc = acc + pos_vals
    tl.store(Y_ptr + b * (T_after * M) + t * M + m, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        x: (B, 1, 80, T) bfloat16 tensor
        conv2d* weights/bias: provided as in get_inputs, bfloat16
        conv_out_weight: (1024, 3840) bfloat16
        positional_embedding: (1500, 1024) bfloat16
        embed_scale: float (32.0)
        Returns: (B, time_after_conv, 1024) float32
        """
        assert TRITON_AVAILABLE, "Triton not available"
        # Cast inputs to float32 for kernel compute
        x_f32 = x.to(torch.float32)
        w1 = conv2d1_weight.to(torch.float32)
        b1 = conv2d1_bias.to(torch.float32)
        w2 = conv2d2_weight.to(torch.float32)
        b2 = conv2d2_bias.to(torch.float32)
        w3 = conv2d3_weight.to(torch.float32)
        b3 = conv2d3_bias.to(torch.float32)
        W = conv_out_weight.to(torch.float32)          # (1024, 3840)
        pos = positional_embedding.to(torch.float32)   # (1500, 1024)

        # conv1: (1,80,T) -> (B,384,40,OW1)
        B = x_f32.shape[0]
        IH = 80
        IW = x_f32.shape[-1]
        OC1 = w1.shape[0]
        y1 = torch.empty((B, OC1, (IH - 1) // 2 + 1, (IW - 1) // 2 + 1), dtype=torch.float32)
        grid1 = (B, OC1, (IH - 1) // 2 + 1, (IW - 1) // 2 + 1)
        conv2d_stride2_kernel[grid1](
            x_f32, w1, b1, y1,
            B, 1, IH, IW, OC1, (IH - 1) // 2 + 1, (IW - 1) // 2 + 1
        )

        # GELU conv1
        y1_flat = y1.reshape(-1)
        y1_gelu = torch.empty_like(y1_flat)
        N1 = y1_flat.numel()
        gelu_kernel_1d[(N1 + 1023) // 1024,](y1_flat, y1_gelu, N1, 1.0)
        y1_gelu = y1_gelu.reshape(B, OC1, (IH - 1) // 2 + 1, (IW - 1) // 2 + 1)

        # conv2: (384,40,OW1) -> (B,384,20,OW2)
        y2 = torch.empty((B, OC1, ( (IH - 1) // 2 + 1 ) - 1 // 2 + 1, ( (IW - 1) // 2 + 1 ) - 1 // 2 + 1), dtype=torch.float32)
        # The above line was a mistake; simplify by computing OW2 correctly:
        OW1 = (IW - 1) // 2 + 1
        y2_OH = ( (IH - 1) // 2 + 1 ) // 2
        y2_OW = (OW1 - 1) // 2 + 1
        y2 = torch.empty((B, OC1, y2_OH, y2_OW), dtype=torch.float32)
        grid2 = (B, OC1, y2_OH, y2_OW)
        conv2d_stride2_kernel[grid2](
            y1_gelu, w2, b2, y2,
            B, OC1, (IH - 1) // 2 + 1, OW1, OC1, y2_OH, y2_OW
        )

        # GELU conv2
        y2_flat = y2.reshape(-1)
        y2_gelu = torch.empty_like(y2_flat)
        N2 = y2_flat.numel()
        gelu_kernel_1d[(N2 + 1023) // 1024,](y2_flat, y2_gelu, N2, 1.0)
        y2_gelu = y2_gelu.reshape(B, OC1, y2_OH, y2_OW)

        # conv3: (384,20,OW2) -> (B,384,10,OW3)
        y3_OH = y2_OH // 2  # 10
        y3_OW = (y2_OW - 1) // 2 + 1  # depends on T
        y3 = torch.empty((B, OC1, y3_OH, y3_OW), dtype=torch.float32)
        grid3 = (B, OC1, y3_OH, y3_OW)
        conv2d_stride2_kernel[grid3](
            y2_gelu, w3, b3, y3,
            B, OC1, y2_OH, y2_OW, OC1, y3_OH, y3_OW
        )

        # Reshape conv3 to (B, T_after, 384*10) where T_after = y3_OW and 384*10=3840
        T_after = y3_OW
        y3_reshaped = y3.reshape(B, T_after, 384 * 10)

        # Final linear projection + positional embedding
        # X_lin: (B, T_after, 3840)
        # W: (1024, 3840)
        # pos: (T_after, 1024)
        X_lin = y3_reshaped.to(torch.float32)
        M = W.shape[0]  # 1024
        Y = torch.empty((B, T_after, M), dtype=torch.float32)
        pos_sub = pos[:T_after, :]
        grid_fp = (B, T_after, M)
        linear_pos_kernel[grid_fp](
            X_lin, W, pos_sub, Y,
            B, T_after, 384 * 10, M, embed_scale
        )
        return Y


def run(*args):
    return ModelNew()(*args)
