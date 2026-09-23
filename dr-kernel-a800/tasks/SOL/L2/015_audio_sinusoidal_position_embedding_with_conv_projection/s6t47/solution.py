import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: GELU (tanh approximation) on NCHW tensor
# x_ptr -> input, out_ptr -> output, shapes: [B, C, H, W]
@triton.jit
def gelu_nchw_kernel(
    x_ptr, out_ptr,
    B, C, H, W,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    x_idx = b * x_stride_b + c * x_stride_c + h * x_stride_h + w * x_stride_w
    out_idx = b * out_stride_b + c * out_stride_c + h * out_stride_h + w * out_stride_w
    x_val = tl.load(x_ptr + x_idx)
    # tanh-based GELU: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    tanh_arg = c0 * (x_val + 0.044715 * x3)
    y = 0.5 * x_val * (1.0 + tl.math.tanh(tanh_arg))
    tl.store(out_ptr + out_idx, y)


# Triton kernel: Linear projection y[b, t, d] = sum_k x[b, t, k] * W[d, k]
# x: [B, T, K], W: [N, K] (note: W is transposed conv_out_weight so W[d, k] = original W^T[k, d])
# out: [B, T, N]
@triton.jit
def linear_btk_kn_kernel(
    x_ptr, W_ptr, out_ptr,
    B, T, K, N,
    x_stride_b, x_stride_t, x_stride_k,
    W_stride_n, W_stride_k,  # W is [N, K]
    out_stride_b, out_stride_t, out_stride_n,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)
    acc = tl.zeros((), dtype=tl.float32)
    # loop over K in chunks for better performance
    for k_start in range(0, K, 64):
        k_offsets = k_start + tl.arange(0, 64)
        k_mask = k_offsets < K
        # load x[b, t, k_offsets]
        x_off = b * x_stride_b + t * x_stride_t + k_offsets * x_stride_k
        x_vec = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)
        # load W[d, k_offsets] = W_ptr[d * W_stride_n + k_offsets * W_stride_k]
        W_off = d * W_stride_n + k_offsets * W_stride_k
        W_vec = tl.load(W_ptr + W_off, mask=k_mask, other=0.0)
        # accumulate dot product
        acc += tl.sum(x_vec * W_vec, axis=0)
    # store result
    out_off = b * out_stride_b + t * out_stride_t + d * out_stride_n
    tl.store(out_ptr + out_off, acc)


# Triton kernel: add scaled positional embedding to y[b, t, :]
# y: [B, T, N], pos: [T, N], scale: float, out_ptr can point to y itself (in-place)
@triton.jit
def add_pos_emb_btk_kernel(
    y_ptr, pos_ptr, scale,
    B, T, N,
    y_stride_b, y_stride_t, y_stride_n,
    pos_stride_t, pos_stride_n,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n = tl.program_id(2)
    y_off = b * y_stride_b + t * y_stride_t + n * y_stride_n
    pos_off = t * pos_stride_t + n * pos_stride_n
    y_val = tl.load(y_ptr + y_off)
    pos_val = tl.load(pos_ptr + pos_off)
    y_new = y_val + pos_val * scale
    tl.store(y_ptr + y_off, y_new)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        input_features = args[0]            # [B, 1, 80, T]
        conv2d1_weight = args[1]            # [C_out1, C_in, 3, 3] = [384, 1, 3, 3]
        conv2d1_bias = args[2]              # [C_out1]
        conv2d2_weight = args[3]            # [384, 384, 3, 3]
        conv2d2_bias = args[4]              # [384]
        conv2d3_weight = args[5]            # [384, 384, 3, 3]
        conv2d3_bias = args[6]              # [384]
        conv_out_weight = args[7]           # [d_model, conv_out_dim] = [1024, conv_out_dim] (conv_out_dim may be 3840)
        positional_embedding = args[8]      # [max_source_positions, d_model]
        embed_scale = float(args[9])        # float

        # Stage 1: Conv2d (1 -> 384) + GELU (PyTorch conv, Triton GELU)
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = x.contiguous()
        B, C1, H1, W1 = x.shape
        x_gelu = torch.empty_like(x, dtype=torch.float32, device=x.device)
        grid1 = (B, C1, H1, W1)
        gelu_nchw_kernel[grid1](
            x, x_gelu,
            B, C1, H1, W1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            x_gelu.stride(0), x_gelu.stride(1), x_gelu.stride(2), x_gelu.stride(3),
        )

        # Stage 2: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = x.contiguous()
        B, C2, H2, W2 = x.shape
        x_gelu2 = torch.empty_like(x, dtype=torch.float32, device=x.device)
        grid2 = (B, C2, H2, W2)
        gelu_nchw_kernel[grid2](
            x, x_gelu2,
            B, C2, H2, W2,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            x_gelu2.stride(0), x_gelu2.stride(1), x_gelu2.stride(2), x_gelu2.stride(3),
        )

        # Stage 3: Conv2d (384 -> 384) + GELU
        x = F.conv2d(x_gelu2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = x.contiguous()
        B, C3, H3, W3 = x.shape
        x_gelu3 = torch.empty_like(x, dtype=torch.float32, device=x.device)
        grid3 = (B, C3, H3, W3)
        gelu_nchw_kernel[grid3](
            x, x_gelu3,
            B, C3, H3, W3,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            x_gelu3.stride(0), x_gelu3.stride(1), x_gelu3.stride(2), x_gelu3.stride(3),
        )

        # Reshape: (batch, channels, H, W) -> (batch, W, channels*H)
        b, c, f, t = x_gelu3.shape
        B = b
        T = t
        K = c * f  # features after conv3: C_out3 * H_out3 * W_out3

        # Flatten to [B, T, K]
        x_flat = x_gelu3.permute(0, 3, 1, 2).contiguous().view(B, T, K)

        # Prepare conv_out_weight for Triton linear: W as [N, K] where N=d_model=1024
        # conv_out_weight is [d_model, conv_out_dim]; conv_out_dim in helper may be 3840, but original logic uses full K.
        N = 1024
        # Ensure float32 for stable accumulation
        # W[d, k] = conv_out_weight[k, d] where k in [0..K-1], d in [0..N-1]
        # We pass W as [N, K] so kernel can read W[d, k]
        W = conv_out_weight.t().contiguous().to(torch.float32)  # [K, N]

        # Allocate output for linear: [B, T, N]
        out_linear = torch.empty((B, T, N), dtype=torch.float32, device=x.device)

        # Launch Triton linear kernel: y[b, t, d] = sum_k x[b, t, k] * W[d, k]
        grid_linear = (B, T, N)
        linear_btk_kn_kernel[grid_linear](
            x_flat, W, out_linear,
            B, T, K, N,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            W.stride(0), W.stride(1),
            out_linear.stride(0), out_linear.stride(1), out_linear.stride(2),
        )

        # Scale embeddings
        y_scaled = out_linear * embed_scale  # broadcast scalar

        # Add positional embeddings [T, d_model] -> [T, N], matching N=1024
        pos_embed = positional_embedding[:T, :].to(torch.float32)  # [T, N]

        # Launch Triton add-positional-embedding kernel
        grid_emb = (B, T, N)
        add_pos_emb_btk_kernel[grid_emb](
            y_scaled, pos_embed, embed_scale,
            B, T, N,
            y_scaled.stride(0), y_scaled.stride(1), y_scaled.stride(2),
            pos_embed.stride(0), pos_embed.stride(1),
        )

        # Return result in bfloat16 to match original expectations
        return y_scaled.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
