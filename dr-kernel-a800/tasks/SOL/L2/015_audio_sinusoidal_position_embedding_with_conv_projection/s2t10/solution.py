import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: 2D conv, stride=2, padding=1, 3x3, input C_in, output C_out
# Input: x[B, C_in, IH, IW], weight w[C_out, C_in, 3, 3], bias b[C_out]
# Output: y[B, C_out, OH, OW], with OH = (IH - 1)//2 + 1, OW = (IW + 1)//2
@triton.jit
def conv2d_stride2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C_in, IH, IW, C_out, OH, OW,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_ic, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # stride=2, padding=1 -> ih = 2*oh + kh - 1, iw = 2*ow + kw - 1
    for kh in range(3):
        ih = 2 * oh + kh - 1
        valid_h = (ih >= 0) & (ih < IH)
        for kw in range(3):
            iw = 2 * ow + kw - 1
            valid_w = (iw >= 0) & (iw < IW)
            valid = valid_h & valid_w
            if valid:
                # Only one input channel (C_in=1), so loop runs once
                for ic in range(C_in):
                    x_off = b * x_stride_b + ic * x_stride_c + ih * x_stride_h + iw * x_stride_w
                    x_val = tl.load(x_ptr + x_off)
                    w_off = oc * w_stride_oc + ic * w_stride_ic + kh * w_stride_kh + kw * w_stride_kw
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + oc)
    acc += b_val

    # store output
    y_off = b * y_stride_b + oc * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_off, acc)


# Triton GELU kernel (tanh approximation) over 1D flattened tensors
@triton.jit
def gelu_kernel_1d(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = x + 0.044715 * x3
    gelu = 0.5 * x * (1.0 + tl.tanh(c * inner))
    tl.store(y_ptr + offsets, gelu, mask=mask)


# Triton kernel: final linear projection + scale + add positional embedding
# X: [B, T, N] row-wise, W: [M, N] (M=1024), pos_emb: [T, M]
@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T, N, M, scale,
    x_stride_b, x_stride_t, x_stride_n,
    w_stride_m, w_stride_n,
    y_stride_b, y_stride_t, y_stride_m,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # for each output channel m
    for m in range(0, M):
        acc = tl.zeros((), dtype=tl.float32)
        # accumulate over N
        for n0 in range(0, N, 16):
            offs = n0 + tl.arange(0, 16)
            mask = offs < N
            x_offs = b * x_stride_b + t * x_stride_t + offs * x_stride_n
            x_vec = tl.load(X_ptr + x_offs, mask=mask, other=0.0)
            w_offs = m * w_stride_m + offs * w_stride_n
            w_vec = tl.load(W_ptr + w_offs, mask=mask, other=0.0)
            acc += tl.sum(x_vec * w_vec, axis=0)
        acc = acc * scale
        # add positional embedding for this t and m
        pos_off = t * M + m
        pos_val = tl.load(pos_ptr + pos_off)
        acc += pos_val
        # store Y[b, t, m]
        y_off = b * y_stride_b + t * y_stride_t + m * y_stride_m
        tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        assert len(args) == 9, "Expected 9 tensors: input_features..positional_embedding, embed_scale"
        input_features = args[0]
        conv1_w = args[1]  # [OC, C_in, 3, 3] = [384, 1, 3, 3]
        conv1_b = args[2]  # [384]
        conv2_w = args[3]  # [384, 384, 3, 3]
        conv2_b = args[4]  # [384]
        conv3_w = args[5]  # [384, 384, 3, 3]
        conv3_b = args[6]  # [384]
        conv_out_w = args[7]  # [d_model, N] = [1024, 3840]
        pos_emb = args[8]  # [max_source_positions, d_model] = [1500, 1024]
        embed_scale = args[9]  # float

        B, C_in, IH, IW = input_features.shape  # (B, 1, 80, time_dim)
        OC = 384

        # Convert inputs to fp32 for kernels
        x = input_features.to(torch.float32).contiguous()

        # Compute output spatial sizes for stride=2, padding=1
        OH1 = (IH - 1) // 2 + 1  # 40
        OW1 = (IW + 1) // 2      # depends on time_dim

        # Stage 1: conv1 (C_in=1) -> (B, 384, 40, OW1)
        y1 = torch.empty((B, OC, OH1, OW1), dtype=torch.float32, device=input_features.device)
        grid1 = (B, OC, OH1, OW1)
        conv2d_stride2_kernel[grid1](
            x, conv1_w.to(torch.float32), conv1_b.to(torch.float32), y1,
            B, C_in, IH, IW, OC, OH1, OW1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        )
        # GELU after conv1
        y1_g = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_kernel_1d[(N1 + 1024 - 1) // 1024](y1, y1_g, N1, BLOCK=1024)

        # Stage 2: conv2 (C_in=OC, 384) -> (B, 384, 20, OW2)
        OH2 = (OH1 - 1) // 2 + 1  # 20
        OW2 = (OW1 + 1) // 2      # half of OW1
        y2 = torch.empty((B, OC, OH2, OW2), dtype=torch.float32, device=input_features.device)
        grid2 = (B, OC, OH2, OW2)
        conv2d_stride2_kernel[grid2](
            y1_g, conv2_w.to(torch.float32), conv2_b.to(torch.float32), y2,
            B, OC, OH1, OW1, OC, OH2, OW2,
            y1_g.stride(0), y1_g.stride(1), y1_g.stride(2), y1_g.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        )
        # GELU after conv2
        y2_g = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_kernel_1d[(N2 + 1024 - 1) // 1024](y2, y2_g, N2, BLOCK=1024)

        # Stage 3: conv3 (C_in=OC, 384) -> (B, 384, 10, OW3)
        OH3 = (OH2 - 1) // 2 + 1  # 10
        OW3 = (OW2 + 1) // 2      # final time_after_conv from workload
        y3 = torch.empty((B, OC, OH3, OW3), dtype=torch.float32, device=input_features.device)
        grid3 = (B, OC, OH3, OW3)
        conv2d_stride2_kernel[grid3](
            y2_g, conv3_w.to(torch.float32), conv3_b.to(torch.float32), y3,
            B, OC, OH2, OW2, OC, OH3, OW3,
            y2_g.stride(0), y2_g.stride(1), y2_g.stride(2), y2_g.stride(3),
            conv3_w.stride(0), conv3_w.stride(1), conv3_w.stride(2), conv3_w.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
        )
        # GELU after conv3
        y3_g = torch.empty_like(y3)
        N3 = y3.numel()
        gelu_kernel_1d[(N3 + 1024 - 1) // 1024](y3, y3_g, N3, BLOCK=1024)

        # Reshape to (B, T, N) where N = OC * final_H = 384 * 10 = 3840
        B3, C_out, FH, TW = y3_g.shape  # FH=10 (given by workload), TW = time_after_conv
        N = C_out * FH  # 3840
        X_reshaped = y3_g.reshape(B3, TW, N).contiguous()  # (B, TW, 3840)

        # Final linear projection + scale + positional embedding
        M = conv_out_w.shape[0]  # 1024
        pos_emb_fp32 = pos_emb.to(torch.float32).contiguous()
        Y = torch.empty((B3, TW, M), dtype=torch.float32, device=input_features.device)

        grid_final = (B3, TW)
        linear_project_pos_kernel[grid_final](
            X_reshaped, conv_out_w.to(torch.float32), pos_emb_fp32,
            Y,
            B3, TW, N, M, embed_scale,
            X_reshaped.stride(0), X_reshaped.stride(1), X_reshaped.stride(2),
            conv_out_w.stride(0), conv_out_w.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
        )

        return Y


def run(*args):
    return ModelNew()(*args)
