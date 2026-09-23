import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def linear_project_kernel(
    x_ptr,                # *const T: input x of shape [B, T, K]
    w_ptr,                # *const T: weight of shape [N, K]
    y_ptr,                # *mut T: output y of shape [B, T, N]
    B: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_x_b, stride_x_t, stride_x_k,
    stride_w_d, stride_w_k,
    stride_y_b, stride_y_t, stride_y_d,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    if pid_b >= B or pid_t >= T or pid_d >= N:
        return

    # base pointers
    x_base = x_ptr + pid_b * stride_x_b + pid_t * stride_x_t
    y_out = y_ptr + pid_b * stride_y_b + pid_t * stride_y_t + pid_d * stride_y_d

    # dot product over K
    acc = 0.0
    for k in range(0, K):
        x_val = tl.load(x_base + k * stride_x_k)
        w_val = tl.load(w_ptr + pid_d * stride_w_d + k * stride_w_k)
        acc += x_val * w_val

    tl.store(y_out, acc)


@triton.jit
def add_pos_embed_kernel(
    y_ptr,                # *const T: y of shape [B, T, N]
    pos_ptr,              # *const T: positional_embedding of shape [T, N]
    out_ptr,              # *mut T: output [B, T, N]
    B: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    stride_y_b, stride_y_t, stride_y_d,
    stride_pos_t, stride_pos_d,
    stride_out_b, stride_out_t, stride_out_d,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    if pid_b >= B or pid_t >= T or pid_d >= N:
        return

    y_in = y_ptr + pid_b * stride_y_b + pid_t * stride_y_t + pid_d * stride_y_d
    pos_val = tl.load(pos_ptr + pid_t * stride_pos_t + pid_d * stride_pos_d)
    out_out = out_ptr + pid_b * stride_out_b + pid_t * stride_out_t + pid_d * stride_out_d

    val = tl.load(y_in) + pos_val
    tl.store(out_out, val)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args order: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        input_features = args[0]
        conv2d1_weight, conv2d1_bias = args[1], args[2]
        conv2d2_weight, conv2d2_bias = args[3], args[4]
        conv2d3_weight, conv2d3_bias = args[5], args[6]
        conv_out_weight = args[7]  # [N, K], N=d_model=1024, K=conv_out_dim (3840 in helper)
        positional_embedding = args[8]  # [max_source_positions, d_model]
        embed_scale = args[9]  # float

        # Conv1
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x, approximate='tanh')
        # Conv2
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x, approximate='tanh')
        # Conv3
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x, approximate='tanh')

        # Reshape: (B, C3, H3, W3) -> (B, W3, C3*H3)
        x = x.permute(0, 3, 1, 2).contiguous()
        B = x.shape[0]
        T = x.shape[1]  # time_after_conv (dynamic per workload)
        K = x.shape[2]  # features per time step

        x = x.view(B, T, K)

        # Linear projection in Triton: y[b, t, d] = sum_k x[b, t, k] * W[d, k]
        N = conv_out_weight.shape[0]
        y = torch.empty((B, T, N), dtype=x.dtype, device=x.device)

        stride_x_b, stride_x_t, stride_x_k = x.stride()
        stride_w_d, stride_w_k = conv_out_weight.stride()
        stride_y_b, stride_y_t, stride_y_d = y.stride()

        grid = (B, T, N)
        linear_project_kernel[grid](
            x, conv_out_weight, y,
            B, T, K, N,
            stride_x_b, stride_x_t, stride_x_k,
            stride_w_d, stride_w_k,
            stride_y_b, stride_y_t, stride_y_d,
            num_warps=4,
            num_stages=2,
        )

        # Scale by embed_scale
        y = y * embed_scale

        # Add positional embedding (broadcast across batch)
        pos = positional_embedding[:T, :].contiguous()  # [T, N]
        out = torch.empty_like(y)

        stride_pos_t, stride_pos_d = pos.stride()
        stride_out_b, stride_out_t, stride_out_d = out.stride()

        add_pos_embed_kernel[grid](
            y, pos, out,
            B, T, N,
            stride_y_b, stride_y_t, stride_y_d,
            stride_pos_t, stride_pos_d,
            stride_out_b, stride_out_t, stride_out_d,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
