import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: conv2d NCHW, 3x3, stride=2, padding=1, with GELU (tanh approx), no bias.
# x: [B, C_in, H, W], w: [C_out, C_in, 3, 3], y: [B, C_out, H_out, W_out]
@triton.jit
def conv3x3_stride2_gelu_nchw(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    # Grid: (B, H_out, W_out)
    b = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)

    # Accumulator for output channels
    # We'll compute all output channels for this (b, oh, ow).
    # Initialize a vector y_vec of length C_out with zeros (in float32).
    # Note: Triton doesn't allow Python lists in @jit; we'll compute each co scalar in a loop.
    # So we instead store each co scalar immediately after computing.

    # Loop over output channels
    for co in range(0, C_out):
        y_val = tl.zeros((), dtype=tl.float32)
        # Loop over input channels and 3x3 kernel
        for ic in range(0, C_in):
            for kh in range(0, 3):
                ih = oh * 2 - 1 + kh  # stride=2, padding=1 => ih = 2*oh - 1 + kh
                for kw in range(0, 3):
                    iw = ow * 2 - 1 + kw  # iw = 2*ow - 1 + kw
                    # Validity check for input index
                    in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                    # Compute input offset for this (b, ic, ih, iw)
                    x_off = b * x_stride_b + ic * x_stride_c + ih * x_stride_h + iw * x_stride_w
                    # Load input value; if out-of-bounds, use 0.0
                    x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                    # Load corresponding weight w[co, ic, kh, kw]
                    w_off = co * w_stride_co + ic * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                    w_val = tl.load(w_ptr + w_off)
                    # Accumulate
                    y_val += x_val * w_val

        # Add bias (optional bias not provided; if needed, uncomment and add bias load)
        # bias_val = tl.load(bias_ptr + co)
        # y_val += bias_val

        # Apply GELU (tanh approximation)
        # c = sqrt(2/pi) = 0.7978845608028654
        c = 0.7978845608028654
        x3 = y_val * y_val * y_val
        tanh_arg = c * (y_val + 0.044715 * x3)
        tanh_val = tl.math.tanh(tanh_arg)
        y_val = 0.5 * y_val * (1.0 + tanh_val)

        # Store to output y[b, co, oh, ow]
        y_off = b * y_stride_b + co * y_stride_c + oh * y_stride_h + ow * y_stride_w
        tl.store(y_ptr + y_off, y_val)


# Triton kernel: batched GEMV for y[b, t, d] = sum_k x[b, t, k] * W[d, k], no bias.
# x: [B, T, K] row-major, W: [N, K], y: [B, T, N]
@triton.jit
def linear_gemv_kernel(
    x_ptr, w_ptr, y_ptr,
    B, T, K, N,
    x_stride_b, x_stride_t, x_stride_k,
    y_stride_b, y_stride_t, y_stride_n,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Accumulator for N outputs
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # vector
        k_mask = k_offsets < K
        # Load x[b, t, k_offsets] vector
        x_off = b * (T * K) + t * K + k_offsets  # linear indexing
        x_vec = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)  # [BLOCK_K], dtype inferred
        # Accumulate over N: acc[d] += sum_k x_vec[k] * W[d, k]
        # We need to load W[d, k_offsets] for all d and multiply.
        # Triton supports broadcasting; we'll loop d and use masked vector ops.
        for d in range(0, N):
            w_vec = tl.load(w_ptr + d * K + k_offsets, mask=k_mask, other=0.0)
            acc[d] += tl.sum(x_vec * w_vec, axis=0)

    # Store y[b, t, d] = acc[d]
    for d in range(0, N):
        y_off = b * (T * N) + t * N + d
        tl.store(y_ptr + y_off, acc[d])


# Triton kernel: add positional embedding to y[b, t, d]
# y: [B, T, N], pos_emb: [T, N]
@triton.jit
def add_pos_emb_kernel(
    y_ptr, pos_ptr,
    B, T, N,
    y_stride_b, y_stride_t, y_stride_n,
    pos_stride_t, pos_stride_n,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Load pos_emb[t, :] vector and add to y[b, t, :]
    for d in range(0, N):
        y_off = b * (T * N) + t * N + d
        pos_off = t * pos_stride_t + d * pos_stride_n
        val = tl.load(y_ptr + y_off) + tl.load(pos_ptr + pos_off)
        tl.store(y_ptr + y_off, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.d_model = 1024
        self.embed_scale = math.sqrt(self.d_model)

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding):
        """
        input_features: [B, 1, 80, T], bfloat16, contiguous
        convN_weight: [C_out, C_in, 3, 3], bfloat16, contiguous
        convN_bias: [C_out], bfloat16, contiguous
        conv_out_weight: [d_model, K_conv_out], bfloat16, contiguous
        positional_embedding: [max_source_positions, d_model], bfloat16, contiguous
        Returns: y [B, T_out, d_model], bfloat16
        """

        # Ensure tensors are on CUDA and contiguous
        assert input_features.is_cuda, "All inputs must be CUDA tensors"
        assert conv2d1_weight.is_cuda and conv2d2_weight.is_cuda and conv2d3_weight.is_cuda
        assert conv_out_weight.is_cuda and positional_embedding.is_cuda

        B, _, H, T = input_features.shape
        C_in = 1
        # First conv: 3x3, stride=2, padding=1
        H_out1 = (H + 2 - 3) // 2 + 1
        W_out1 = (T + 2 - 3) // 2 + 1
        # Allocate output for conv1
        x = torch.empty((B, 384, H_out1, W_out1), dtype=input_features.dtype, device=input_features.device)
        # Launch Triton conv kernel for conv1
        conv3x3_stride2_gelu_nchw[ (B, H_out1, W_out1) ](
            input_features, conv2d1_weight, x,
            B, C_in, H, T, 384, H_out1, W_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        )
        # GELU done inside kernel above; no need to call F.gelu here

        # Second conv: in_channels=384 -> out_channels=384
        H_out2 = (H_out1 + 2 - 3) // 2 + 1
        W_out2 = (W_out1 + 2 - 3) // 2 + 1
        y = torch.empty((B, 384, H_out2, W_out2), dtype=input_features.dtype, device=input_features.device)
        conv3x3_stride2_gelu_nchw[ (B, H_out2, W_out2) ](
            x, conv2d2_weight, y,
            B, 384, H_out1, W_out1, 384, H_out2, W_out2,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        )

        # Third conv: in_channels=384 -> out_channels=384
        H_out3 = (H_out2 + 2 - 3) // 2 + 1
        W_out3 = (W_out2 + 2 - 3) // 2 + 1
        z = torch.empty((B, 384, H_out3, W_out3), dtype=input_features.dtype, device=input_features.device)
        conv3x3_stride2_gelu_nchw[ (B, H_out3, W_out3) ](
            y, conv2d3_weight, z,
            B, 384, H_out2, W_out2, 384, H_out3, W_out3,
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        )

        # Reshape to [B, T_final, K] where T_final=W_out3, K=C_out*H_out3*W_out3
        # In our fixed setup, K should be 384*H_out3*W_out3. However, we need general handling.
        # Compute K dynamically.
        C_out3 = 384
        K = C_out3 * H_out3 * W_out3
        B2, T_final = z.shape[0], z.shape[3]
        # Permute to [B, W, C*H], then [B, T_final, K]
        z_perm = z.permute(0, 3, 1, 2).contiguous()  # [B, W_out3, 384, H_out3]
        x_flat = z_perm.view(B, T_final, K).contiguous()  # [B, T_final, K]

        # Linear projection to d_model (no bias)
        N = self.d_model  # 1024
        # Ensure conv_out_weight is [N, K]
        assert conv_out_weight.shape[0] == N, f"conv_out_weight first dim must be {N}, got {conv_out_weight.shape[0]}"
        assert conv_out_weight.shape[1] == K, f"conv_out_weight second dim must be {K}, got {conv_out_weight.shape[1]}"
        # Allocate y_out [B, T_final, N]
        y_out = torch.empty((B, T_final, N), dtype=input_features.dtype, device=input_features.device)
        # Launch Triton GEMV kernel
        # x_flat: [B, T_final, K], conv_out_weight: [N, K], y_out: [B, T_final, N]
        BLOCK_K = 128
        linear_gemv_kernel[(B, T_final)](
            x_flat, conv_out_weight,
            y_out,
            B, T_final, K, N,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            y_out.stride(0), y_out.stride(1), y_out.stride(2),
            BLOCK_K=BLOCK_K,
        )

        # Scale by embed_scale
        # Multiply in-place
        y_out = y_out * self.embed_scale

        # Add positional embedding: shape [max_source_positions, d_model] provided, we need slice [:seq_len, :]
        # seq_len is time_after_conv in helper; here it's T_final. But positional_embedding provided is [max_source_positions, d_model].
        # We assume max_source_positions >= T_final; otherwise, we can only use what's available. The original code uses up to d_model dimension.
        # We slice positional_embedding to [T_final, d_model] and add.
        # Cast to y_out dtype and device; ensure contiguous.
        pos_emb = positional_embedding[:T_final, :].to(dtype=y_out.dtype, device=y_out.device).contiguous()
        # Launch add_pos_emb kernel
        add_pos_emb_kernel[(B, T_final)](
            y_out, pos_emb,
            B, T_final, N,
            y_out.stride(0), y_out.stride(1), y_out.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
