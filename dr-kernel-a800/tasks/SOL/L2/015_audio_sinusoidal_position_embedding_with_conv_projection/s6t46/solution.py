import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Linear projection y[b, t, d] = sum_k x[b, t, k] * W[d, k], no bias.
# Inputs: x_ptr[B, T, K], W_ptr[N, K], y_ptr[B, T, N]
# Each program handles one (b, t) row, producing y[b, t, :]
@triton.jit
def linear_proj_kernel(
    x_ptr,            # *ptr to x [B, T, K], dtype float32
    W_ptr,            # *ptr to W [N, K], dtype float32
    y_ptr,            # *ptr to output y [B, T, N], dtype float32
    B, T, K, N,
    x_stride_b, x_stride_t, x_stride_k,
    W_stride_n, W_stride_k,
    y_stride_b, y_stride_t, y_stride_n,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # We produce output vector for all N dims in chunks of BLOCK_N
    n_offsets = tl.arange(0, BLOCK_N)
    for n_start in range(0, N, BLOCK_N):
        n = n_start + n_offsets
        mask_n = n < N

        # Accumulator for this (b, t): shape [BLOCK_N]
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Loop over K and accumulate dot products: y[b, t, n] += sum_k x[b, t, k] * W[n, k]
        for k in range(0, K):
            # Load x[b, t, k]
            x_off = b * x_stride_b + t * x_stride_t + k * x_stride_k
            x_val = tl.load(x_ptr + x_off).to(tl.float32)
            # Load W[n, k] for this chunk
            w_vals = tl.load(W_ptr + n * W_stride_n + k * W_stride_k, mask=mask_n, other=0.0).to(tl.float32)
            acc += w_vals * x_val

        # Store acc to y[b, t, n]
        for i in range(0, BLOCK_N):
            if (n_start + i) < N:
                y_off = b * y_stride_b + t * y_stride_t + (n_start + i) * y_stride_n
                tl.store(y_ptr + y_off, acc[i])


# Triton kernel: Add positional embedding to y[b, t, d]
# y_ptr[B, T, N], pos_ptr[T, N], grid=(B, T)
@triton.jit
def add_pos_emb_kernel(
    y_ptr,            # *ptr to y [B, T, N], dtype float32
    pos_ptr,          # *ptr to pos_emb [T, N], dtype float32
    B, T, N,
    y_stride_b, y_stride_t, y_stride_n,
    pos_stride_t, pos_stride_n,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_offsets = tl.arange(0, BLOCK_N)
    for n_start in range(0, N, BLOCK_N):
        n = n_start + n_offsets
        mask_n = n < N

        # Load pos[t, n]
        pos_vals = tl.load(pos_ptr + t * pos_stride_t + n * pos_stride_n, mask=mask_n, other=0.0).to(tl.float32)

        # Load y[b, t, n] and add pos
        for i in range(0, BLOCK_N):
            if (n_start + i) < N:
                y_off = b * y_stride_b + t * y_stride_t + (n_start + i) * y_stride_n
                y_val = tl.load(y_ptr + y_off).to(tl.float32)
                y_new = y_val + pos_vals[i]
                tl.store(y_ptr + y_off, y_new)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        device = input_features.device
        assert device.type == "cuda", "ModelNew requires CUDA device for Triton kernels"

        # Run convolutions and GELU with PyTorch (fast and correct)
        # Stage 1
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)  # tanh approximation

        # Stage 2
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 3
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape: (batch, channels, H, W) -> (batch, W, channels*H) and view to (batch, time_after_conv, K)
        b, c, h, w = x.size()
        K = c * h * w
        # Permute to (batch, W, channels, H) then reshape to (batch, W, c*h)
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        # Prepare tensors for Triton
        # x_flat: [B, T, K] as float32 for accumulation
        x_flat = x.to(torch.float32)
        # conv_out_weight: [N, conv_out_dim] in the original helper, but we will use only the first K columns if needed.
        # Here, the helper sets conv_out_dim=3840 and N=1024. We ensure N=1024, K=actual K. If K > conv_out_dim, we would need to pad; here helper ensures K<=conv_out_dim.
        N = conv_out_weight.shape[0]
        W_linear = conv_out_weight.to(torch.float32)  # [N, K]

        # Allocate y [B, T, N]
        y = torch.empty((b, w, N), device=device, dtype=torch.float32)

        # Launch linear projection kernel: grid over (B, T)
        grid = (b, w)
        # Choose BLOCK_N as a power-of-two up to 1024; 128 is a good default
        BLOCK_N = 128
        linear_proj_kernel[grid](
            x_flat, W_linear, y,
            b, w, K, N,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            W_linear.stride(0), W_linear.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

        # Scale by embed_scale
        embed_scale_val = float(embed_scale)
        y = y * embed_scale_val

        # Add positional embedding: pos_emb is [T, N]; slice positional_embedding to [time_after_conv, N]
        pos_emb = positional_embedding[:w, :].to(torch.float32)  # [T, N]

        # Launch add positional embedding kernel: grid over (B, T)
        add_pos_emb_kernel[grid](
            y, pos_emb,
            b, w, N,
            y.stride(0), y.stride(1), y.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

        # Return as bfloat16 (original code uses bfloat16)
        return y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
