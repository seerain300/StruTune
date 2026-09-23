import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr,            # *fp32 (or bf16), input tensor: (B, Ci, H, W)
    w_ptr,            # *fp32 (or bf16), weight tensor: (Co, Ci, Kh, Kw)
    bias_ptr,         # *fp32 (or bf16), bias tensor: (Co,)
    y_ptr,            # *fp32 (or bf16), output tensor: (B, Co, Ho, Wo)
    B, Ci, H, W, Co, Kh, Kw, Ho, Wo,
    x_sN, x_sC, x_sH, x_sW,
    w_sCo, w_sCi, w_sKh, w_sKw,
    y_sN, y_sC, y_sH, y_sW,
):
    # Grid: (B, Co, Ho, Wo)
    b = tl.program_id(0)
    co = tl.program_id(1)
    ho = tl.program_id(2)
    wo = tl.program_id(3)

    # Accumulator for output
    acc = 0.0
    # Loop over input channels and kernel elements
    for ci in range(0, Ci):
        for kh in range(0, Kh):
            hi = ho + kh - 1  # stride=2 conv, padding=1
            in_bounds_h = (hi >= 0) & (hi < H)
            for kw in range(0, Kw):
                ti = wo + kw - 1
                in_bounds_w = (ti >= 0) & (ti < W)
                in_bounds = in_bounds_h & in_bounds_w
                # Compute input offset
                x_offset = b * x_sN + ci * x_sC + hi * x_sH + ti * x_sW
                # Load input with mask; if out of bounds, load 0
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                # Load weight
                w_offset = co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val
    # Add bias
    bias_val = tl.load(bias_ptr + co)
    acc += bias_val

    # Store to output
    y_offset = b * y_sN + co * y_sC + ho * y_sH + wo * y_sW
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def linear_proj_kernel(
    x_ptr,    # *fp32 (or bf16), input: (B, T, K)
    w_ptr,    # *fp32 (or bf16), weight: (D, K)
    out_ptr,  # *fp32 (or bf16), output: (B, T, D)
    B, T, K, D,
    x_sN, x_sT, x_sK,
    w_sD, w_sK,
    out_sN, out_sT, out_sD,
):
    # Grid: (B, T, D)
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)

    acc = 0.0
    # Reduction over K
    for k in range(0, K):
        x_offset = b * x_sN + t * x_sT + k * x_sK
        x_val = tl.load(x_ptr + x_offset)
        w_offset = d * w_sD + k * w_sK
        w_val = tl.load(w_ptr + w_offset)
        acc += x_val * w_val

    out_offset = b * out_sN + t * out_sT + d * out_sD
    tl.store(out_ptr + out_offset, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args are: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        assert len(args) == 9, "ModelNew.forward expects 9 inputs"
        (
            input_features,
            conv2d1_weight,
            conv2d1_bias,
            conv2d2_weight,
            conv2d2_bias,
            conv2d3_weight,
            conv2d3_bias,
            conv_out_weight,   # shape: (d_model=1024, in_features=3840)
            positional_embedding,  # shape: (max_source_positions=1500, d_model=1024), dtype bfloat16
            embed_scale,
        ) = args

        # Ensure dtype is bfloat16 and tensors are contiguous
        # (original code uses bfloat16; keep dtype)
        input_features = input_features.contiguous()
        conv2d1_weight = conv2d1_weight.contiguous()
        conv2d1_bias = conv2d1_bias.contiguous()
        conv2d2_weight = conv2d2_weight.contiguous()
        conv2d2_bias = conv2d2_bias.contiguous()
        conv2d3_weight = conv2d3_weight.contiguous()
        conv2d3_bias = conv2d3_bias.contiguous()
        conv_out_weight = conv_out_weight.contiguous()
        # positional_embedding is fine as-is; we will use only the first T rows.

        # Shapes
        B, Ci, H, W = input_features.shape  # input_features: (B, 1, 80, time_dim)
        Co1, Ci1, Kh, Kw = conv2d1_weight.shape  # (384, 1, 3, 3)
        Ho1 = (H + 2*1 - Kh) // 2 + 1
        Wo1 = (W + 2*1 - Kw) // 2 + 1
        Co2, Ci2, Kh2, Kw2 = conv2d2_weight.shape  # (384, 384, 3, 3)
        Ho2 = (Ho1 + 2*1 - Kh2) // 2 + 1
        Wo2 = (Wo1 + 2*1 - Kw2) // 2 + 1
        Co3, Ci3, Kh3, Kw3 = conv2d3_weight.shape  # (384, 384, 3, 3)
        Ho3 = (Ho2 + 2*1 - Kh3) // 2 + 1
        Wo3 = (Wo2 + 2*1 - Kw3) // 2 + 1

        # Allocate outputs for each conv
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=input_features.device, dtype=input_features.dtype)
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=input_features.device, dtype=input_features.dtype)
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=input_features.device, dtype=input_features.dtype)

        # Launch Triton conv kernels (stride=2, padding=1)
        # conv1
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh, Kw, Ho1, Wo1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            num_warps=4,
        )
        # GELU activation (PyTorch) - must use default approximate='none' to match F.gelu behavior in original
        x1 = F.gelu(x1)

        # conv2
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, Kh2, Kw2, Ho2, Wo2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            num_warps=4,
        )
        x2 = F.gelu(x2)

        # conv3
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, Kh3, Kw3, Ho3, Wo3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            num_warps=4,
        )
        x3 = F.gelu(x3)

        # Reshape: (B, channels, freq, time) -> (B, time, channels*freq)
        b, c, f, t = x3.shape  # t is time_after_conv
        x3 = x3.permute(0, 3, 1, 2).contiguous()  # (B, t, c*f)
        K = c * f  # 384 * 10 = 3840
        D = conv_out_weight.shape[0]  # 1024

        # Linear projection to d_model using Triton
        out = torch.empty((B, t, D), device=input_features.device, dtype=input_features.dtype)
        grid_linear = (B, t, D)
        linear_proj_kernel[grid_linear](
            x3, conv_out_weight, out,
            B, t, K, D,
            x3.stride(0), x3.stride(1), x3.stride(2),
            conv_out_weight.stride(0), conv_out_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=4,
        )

        # Scale embeddings
        out = out * embed_scale  # embed_scale is a python float; Triton will treat it as scalar

        # Add positional embeddings (same for all batches; broadcast add)
        # out: (B, t, D), positional_embedding: (max_source_positions=1500, D)
        # We only need first t rows: (t, D). Broadcast add across B.
        pos = positional_embedding[:t, :].to(out.dtype)
        out = out + pos  # broadcast along batch dimension

        return out


def run(*args):
    return ModelNew()(*args)
