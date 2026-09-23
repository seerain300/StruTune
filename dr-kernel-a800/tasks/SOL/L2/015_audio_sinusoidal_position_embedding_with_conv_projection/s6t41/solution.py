import math
import torch
import triton
import triton.language as tl


# Triton kernel: 2D Convolution (NCHW) with 3x3, stride=2, padding=1
# Input: x[B, C_in, H, W], weight[C_out, C_in, 3, 3], bias[C_out]
# Output: out[B, C_out, H_out, W_out]
@triton.jit
def conv2d_3x3_stride2_gelu_nchw_kernel(
    x_ptr, w_ptr, out_ptr, bias_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_HO: tl.constexpr, BLOCK_WO: tl.constexpr,
):
    # Grid: (B, C_out, H_out_tiles, W_out_tiles)
    b = tl.program_id(0)
    co = tl.program_id(1)
    tile_h = tl.program_id(2)
    tile_w = tl.program_id(3)

    # Compute output tile indices
    ho_start = tile_h * BLOCK_HO
    wo_start = tile_w * BLOCK_WO

    ho_offsets = ho_start + tl.arange(0, BLOCK_HO)
    wo_offsets = wo_start + tl.arange(0, BLOCK_WO)

    # Masks for output bounds
    ho_mask = ho_offsets < H_out
    wo_mask = wo_offsets < W_out

    # Initialize accumulation tile
    y_tile = tl.zeros((BLOCK_HO, BLOCK_WO), dtype=tl.bfloat16)

    # Loop over input channels
    for ic in range(0, C_in):
        # Loop over 3x3 neighborhood
        for kh in range(0, 3):
            for kw in range(0, 3):
                # Map output to input indices with padding
                ih = ho_offsets * 2 + (1 - kh)  # 2*ho + (1 - kh)
                iw = wo_offsets * 2 + (1 - kw)  # 2*wo + (1 - kw)
                # Compute validity masks
                valid_h = (ih >= 0) & (ih < H)
                valid_w = (iw >= 0) & (iw < W)
                # Combined mask for loading x
                load_mask = (ho_mask[:, None] & wo_mask[None, :]) & (valid_h[:, None] & valid_w[None, :])

                # Build offsets for x[b, ic, ih, iw]
                x_off = (b * x_stride_b
                         + ic * x_stride_c
                         + ih[:, None] * x_stride_h
                         + iw[None, :] * x_stride_w)
                x_vec = tl.load(x_ptr + x_off, mask=load_mask, other=0.0)

                # Load corresponding weight scalar w[co, ic, kh, kw]
                w_off = co * w_stride_co + ic * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_off)

                # Accumulate outer product into y_tile
                # y_tile += w_val * x_vec (broadcast across tile)
                # Ensure w_val is cast to bfloat16 for arithmetic
                y_tile += w_val.to(tl.bfloat16) * x_vec

    # Add bias
    b_val = tl.load(bias_ptr + co).to(tl.bfloat16)
    y_tile += b_val  # broadcast bias across tile

    # Apply GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = y_tile * y_tile * y_tile
    tanh_arg = c * (y_tile + 0.044715 * x3)
    tanh_val = tl.math.tanh(tanh_arg)
    y_tile = 0.5 * y_tile * (1.0 + tanh_val)

    # Store results
    out_off = (b * out_stride_b
               + co * out_stride_c
               + ho_offsets[:, None] * out_stride_h
               + wo_offsets[None, :] * out_stride_w)
    store_mask = (ho_mask[:, None] & wo_mask[None, :])
    tl.store(out_ptr + out_off, y_tile, mask=store_mask)


# Triton kernel: batched linear projection per (b, t) to produce N outputs
# Input: X[B, T, K] (flattened features), W[N, K] (conv_out_weight), Output Y[B, T, N]
@triton.jit
def linear_projection_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, T, K, N,
    x_stride_b, x_stride_t, x_stride_k,
    w_stride_n, w_stride_k,
    y_stride_b, y_stride_t, y_stride_n,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)

    # For each output feature d, compute dot product over K
    for d in range(0, N):
        acc = tl.zeros((), dtype=tl.bfloat16)
        # Loop over K in chunks
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < K

            # Load x[b, t, k_idx]
            x_off = b * x_stride_b + t * x_stride_t + k_idx * x_stride_k
            x_vec = tl.load(X_ptr + x_off, mask=k_mask, other=0.0)

            # Load W[d, k_idx]
            w_off = d * w_stride_n + k_idx * w_stride_k
            w_vec = tl.load(W_ptr + w_off, mask=k_mask, other=0.0)

            # Accumulate dot product for this d
            # acc += sum(w_vec * x_vec) over valid k
            # Manually reduce: since k_mask is 1D, we can compute masked sum
            # Create a scalar sum by multiplying each element and summing
            prod = w_vec * x_vec
            # Zero out invalid elements
            prod = tl.where(k_mask, prod, 0.0)
            # Reduce across BLOCK_K: sum(prod)
            # Triton does not have tl.sum, but we can fold into scalar
            # Note: tl.sum requires tensor along an axis; here we reduce manually
            # Sum via loop over BLOCK_K (small)
            for kk in range(0, BLOCK_K):
                acc += prod[kk]

        # Store Y[b, t, d] = acc
        y_off = b * y_stride_b + t * y_stride_t + d * y_stride_n
        tl.store(Y_ptr + y_off, acc)


# Triton kernel: add scaled positional embedding to Y[b, t, :]
# Inputs: Y[B, T, N], pos_emb[T, N], scale (float)
# Operation: Y += pos_emb * scale
@triton.jit
def add_scaled_pos_embed_kernel(
    Y_ptr, pos_ptr, scale,
    B, T, N,
    y_stride_b, y_stride_t, y_stride_n,
    pos_stride_t, pos_stride_n,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, T, N)
    b = tl.program_id(0)
    t = tl.program_id(1)
    n = tl.program_id(2)

    # Load pos_emb[t, n]
    pos_off = t * pos_stride_t + n * pos_stride_n
    pos_val = tl.load(pos_ptr + pos_off)
    scaled = pos_val * scale

    # Load Y[b, t, n] and add
    y_off = b * y_stride_b + t * y_stride_t + n * y_stride_n
    y_val = tl.load(Y_ptr + y_off)
    y_new = y_val + scaled

    # Store back
    tl.store(Y_ptr + y_off, y_new)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Unpack args into the same order as the original run()
        # Expected order:
        # input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        (
            input_features,
            conv2d1_weight, conv2d1_bias,
            conv2d2_weight, conv2d2_bias,
            conv2d3_weight, conv2d3_bias,
            conv_out_weight,
            positional_embedding,
            embed_scale,
        ) = args

        # Ensure tensors are on the same device and dtype
        device = input_features.device
        dtype = input_features.dtype  # bfloat16

        # Conv1: [B, 1, 80, time_dim] -> [B, 384, 40, H1]
        B = input_features.shape[0]
        C_in = 1
        H = input_features.shape[2]
        W = input_features.shape[3]
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H + 2*1 - 3) // 2 + 1  # stride=2, padding=1
        W_out1 = (W + 2*1 - 3) // 2 + 1

        x1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=dtype, device=device)
        # Launch Triton conv + GELU
        grid1 = (B, C_out1, triton.cdiv(H_out1, 8), triton.cdiv(W_out1, 8))
        conv2d_3x3_stride2_gelu_nchw_kernel[grid1](
            input_features, conv2d1_weight, x1, conv2d1_bias,
            B, C_in, H, W, C_out1, H_out1, W_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            BLOCK_HO=8, BLOCK_WO=8, num_warps=4
        )

        # Conv2: [B, 384, 40, H1] -> [B, 384, 20, H2]
        C_in2 = C_out1
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H_out1 + 2*1 - 3) // 2 + 1
        W_out2 = (W_out1 + 2*1 - 3) // 2 + 1

        x2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=dtype, device=device)
        grid2 = (B, C_out2, triton.cdiv(H_out2, 8), triton.cdiv(W_out2, 8))
        conv2d_3x3_stride2_gelu_nchw_kernel[grid2](
            x1, conv2d2_weight, x2, conv2d2_bias,
            B, C_in2, H_out1, W_out1, C_out2, H_out2, W_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            BLOCK_HO=8, BLOCK_WO=8, num_warps=4
        )

        # Conv3: [B, 384, 20, H2] -> [B, 384, 10, H3]
        C_in3 = C_out2
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = (H_out2 + 2*1 - 3) // 2 + 1
        W_out3 = (W_out2 + 2*1 - 3) // 2 + 1

        x3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=dtype, device=device)
        grid3 = (B, C_out3, triton.cdiv(H_out3, 8), triton.cdiv(W_out3, 8))
        conv2d_3x3_stride2_gelu_nchw_kernel[grid3](
            x2, conv2d3_weight, x3, conv2d3_bias,
            B, C_in3, H_out2, W_out2, C_out3, H_out3, W_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            BLOCK_HO=8, BLOCK_WO=8, num_warps=4
        )

        # Reshape: (B, C_out3, H_out3, W_out3) -> (B, W_out3, C_out3*H_out3)
        x3_perm = x3.permute(0, 3, 1, 2).contiguous()  # [B, W_out3, C_out3, H_out3]
        T = x3_perm.shape[1]
        K = x3_perm.shape[2] * x3_perm.shape[3]
        X_flat = x3_perm.view(B, T, K)  # [B, T, K]

        # Linear projection: Y[B, T, N] where N=d_model=1024, W=conv_out_weight [N, K]
        N = 1024  # d_model
        # Ensure conv_out_weight is contiguous and dtype matches
        W_flat = conv_out_weight.to(dtype)  # [N, K]
        Y = torch.empty((B, T, N), dtype=dtype, device=device)

        # Launch Triton linear projection kernel
        grid_lin = (B, T)
        linear_projection_kernel[grid_lin](
            X_flat, W_flat, Y,
            B, T, K, N,
            X_flat.stride(0), X_flat.stride(1), X_flat.stride(2),
            W_flat.stride(0), W_flat.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_K=1024, num_warps=4
        )

        # Scale by embed_scale
        Y_scaled = Y * embed_scale

        # Add scaled positional embedding: pos_emb [max_positions, N], slice to T
        pos_emb = positional_embedding.to(dtype)
        # Slice pos_emb to first T rows
        pos_emb = pos_emb[:T, :].contiguous()

        # Launch Triton kernel to add scaled pos_emb
        grid_pos = (B, T, N)
        add_scaled_pos_embed_kernel[grid_pos](
            Y_scaled, pos_emb, embed_scale,
            B, T, N,
            Y_scaled.stride(0), Y_scaled.stride(1), Y_scaled.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
            BLOCK_N=1, num_warps=1
        )

        return Y_scaled


def run(*args):
    return ModelNew()(*args)
