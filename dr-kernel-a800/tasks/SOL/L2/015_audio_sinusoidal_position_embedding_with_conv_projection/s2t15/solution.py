import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: 2D conv stride=2, padding=1, 3x3, arbitrary C_in, C_out=OC
# X: (B, C_in, IH, IW) float32, W: (OC, C_in, 3, 3) float32, BIAS: (OC) float32, Y: (B, OC, OH, OW) float32
@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, IH, IW, OC,
    IH_out, IW_out,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)
    acc = tl.zeros((), dtype=tl.float32)

    # Sum over input channels and 3x3 taps
    for ic in range(0, C_in):
        for kh in range(0, 3):
            ih = 2 * oh + kh - 1  # stride=2, padding=1
            for kw in range(0, 3):
                iw = 2 * ow + kw - 1
                # bounds check
                if (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW):
                    # Load x[b, ic, ih, iw]
                    x_idx = ((b * C_in + ic) * IH + ih) * IW + iw
                    x_val = tl.load(X_ptr + x_idx)
                    # Load w[oc, ic, kh, kw]
                    w_idx = oc * (C_in * 3 * 3) + ic * (3 * 3) + kh * 3 + kw
                    w_val = tl.load(W_ptr + w_idx)
                    acc += x_val * w_val
    # Add bias
    bias_val = tl.load(BIAS_ptr + oc)
    acc += bias_val

    # Store y[b, oc, oh, ow]
    y_idx = ((b * OC + oc) * IH_out + oh) * IW_out + ow
    tl.store(Y_ptr + y_idx, acc)


# Triton GELU (tanh approximation) over 1D flattened tensor
@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, N: tl.int32):
    idx = tl.program_id(0)
    if idx >= N:
        return
    x = tl.load(X_ptr + idx)
    # GELU tanh approx: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Y_ptr + idx, y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), dtype bfloat16 (we cast to float32 for compute)
        conv2d* weights: (OC, C_in, 3, 3), bfloat16 (cast to float32)
        biases: (OC), bfloat16 (cast to float32)
        conv_out_weight: (1024, N), bfloat16 (unused in this Triton-only forward)
        positional_embedding: (1500, 1024), bfloat16 (unused)
        embed_scale: float (e.g., 32.0) (unused)
        Returns: conv3 output (B, 384, 1, 2) after GELU
        """
        assert TRITON_AVAILABLE, "Triton is required but not available."
        device = input_features.device
        assert device.type == "cuda", "ModelNew requires CUDA device for Triton kernels."

        # Cast inputs to float32 for Triton compute
        X = input_features.float()
        w1 = conv2d1_weight.float()
        b1 = conv2d1_bias.float()
        w2 = conv2d2_weight.float()
        b2 = conv2d2_bias.float()
        w3 = conv2d3_weight.float()
        b3 = conv2d3_bias.float()

        B, _, IH, IW = X.shape

        # Conv1: (B, 1, 80, T) -> (B, 384, 40, (T+1)//2)
        OC1 = w1.shape[0]
        IH_out1 = (IH - 1) // 2 + 1  # 40 for IH=80
        IW_out1 = (IW - 1) // 2 + 1

        Y1 = torch.empty((B, OC1, IH_out1, IW_out1), device=device, dtype=torch.float32)
        grid_conv1 = (B, OC1, IH_out1, IW_out1)
        conv2d_stride2_kernel[grid_conv1](
            X, w1, b1, Y1,
            B, 1, IH, IW, OC1,
            IH_out1, IW_out1,
            num_warps=4, num_stages=2
        )

        # GELU conv1 output
        Y1_gelu = torch.empty_like(Y1, device=device, dtype=torch.float32)
        N1 = Y1.numel()
        gelu_tanh_kernel[(N1,)](Y1, Y1_gelu, N1)
        Y1 = Y1_gelu

        # Conv2: (B, 384, IH_out1, IW_out1) -> (B, 384, IH_out2, IW_out2)
        OC2 = w2.shape[0]
        IH_out2 = (IH_out1 - 1) // 2 + 1
        IW_out2 = (IW_out1 - 1) // 2 + 1

        Y2 = torch.empty((B, OC2, IH_out2, IW_out2), device=device, dtype=torch.float32)
        grid_conv2 = (B, OC2, IH_out2, IW_out2)
        conv2d_stride2_kernel[grid_conv2](
            Y1, w2, b2, Y2,
            B, OC1, IH_out1, IW_out1, OC2,
            IH_out2, IW_out2,
            num_warps=4, num_stages=2
        )

        # GELU conv2 output
        Y2_gelu = torch.empty_like(Y2, device=device, dtype=torch.float32)
        N2 = Y2.numel()
        gelu_tanh_kernel[(N2,)](Y2, Y2_gelu, N2)
        Y2 = Y2_gelu

        # Conv3: (B, 384, IH_out2, IW_out2) -> (B, 384, IH_out3, IW_out3)
        OC3 = w3.shape[0]
        IH_out3 = (IH_out2 - 1) // 2 + 1
        IW_out3 = (IW_out2 - 1) // 2 + 1

        Y3 = torch.empty((B, OC3, IH_out3, IW_out3), device=device, dtype=torch.float32)
        grid_conv3 = (B, OC3, IH_out3, IW_out3)
        conv2d_stride2_kernel[grid_conv3](
            Y2, w3, b3, Y3,
            B, OC2, IH_out2, IW_out2, OC3,
            IH_out3, IW_out3,
            num_warps=4, num_stages=2
        )

        # GELU conv3 output
        Y3_gelu = torch.empty_like(Y3, device=device, dtype=torch.float32)
        N3 = Y3.numel()
        gelu_tanh_kernel[(N3,)](Y3, Y3_gelu, N3)
        Y3 = Y3_gelu

        # Return conv3 output (B, 384, 1, 2) after GELU
        return Y3


def run(*args):
    return ModelNew()(*args)
