import math
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride2_gelu_nchw_inplace(
    x_ptr,           # *f32, input: [B, C_in, H, W]
    w_ptr,           # *f32, weight: [C_out, C_in, 3, 3]
    b_ptr,           # *f32, bias: [C_out]
    out_ptr,         # *f32, output: [B, C_out, H_out, W_out]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,  # padding = 1
):
    # Grid: (B, H_out, W_out, C_out)
    b_id = tl.program_id(0)
    ho = tl.program_id(1)
    wo = tl.program_id(2)
    co = tl.program_id(3)

    # Accumulator for output scalar
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # Compute input coordinates with padding
                hi = ho + kh - pad_h
                wi = wo + kw - pad_w
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                # If out-of-bounds, treat as zero
                x_val = tl.load(x_ptr + b_id * (C_in * H * W) + ci * (H * W) + hi * W + wi, mask=in_bounds, other=0.0)
                # Load corresponding weight for (co, ci, kh, kw)
                w_val = tl.load(w_ptr + co * (C_in * 9) + ci * 9 + kh * 3 + kw)
                acc += x_val * w_val

    # Add bias
    b_val = tl.load(b_ptr + co)
    acc += b_val

    # Apply GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + c1 * x3)))

    # Store output
    tl.store(out_ptr + b_id * (C_out * H_out * W_out) + co * (H_out * W_out) + ho * W_out + wo, gelu)


@triton.jit
def linear_gemm_no_bias_tanh_kernel(
    X_ptr,           # *f32, input: [M, K], M=B*T, K=features after conv
    W_ptr,           # *f32, weight: [N, K], N=d_model (e.g., 1024), K=conv_out.features
    Y_ptr,           # *f32, output: [M, N]
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        off_k = k0 + tl.arange(0, BLOCK_K)

        # Load X tile: shape [BLOCK_M, BLOCK_K]
        X_tile = tl.load(
            X_ptr + off_m[:, None] * K + off_k[None, :],
            mask=(off_m[:, None] < M) & (off_k[None, :] < K),
            other=0.0,
        )
        # Load W tile: shape [BLOCK_K, BLOCK_N]
        W_tile = tl.load(
            W_ptr + off_k[:, None] * N + off_n[None, :],
            mask=(off_k[:, None] < K) & (off_n[None, :] < N),
            other=0.0,
        )

        # Accumulate: acc += sum_k (X_tile[:, k] * W_tile[k, :])
        # Manual outer-product accumulation for robustness
        # Note: K_tile may be larger than BLOCK_K due to masked loads, but we use the actual BLOCK_K extent.
        for kk in range(0, BLOCK_K):
            x_vec = X_tile[:, kk]           # [BLOCK_M]
            w_vec = W_tile[kk, :]           # [BLOCK_N]
            acc += x_vec[:, None] * w_vec[None, :]

    # GELU (tanh approximation) per output element: y = gelu(acc)
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    acc3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + c1 * acc3)))

    # Store results
    tl.store(
        Y_ptr + off_m[:, None] * N + off_n[None, :],
        gelu,
        mask=(off_m[:, None] < M) & (off_n[None, :] < N),
    )


@triton.jit
def add_pos_embed_kernel(
    Y_ptr,           # *f32, output [B*T, N]
    pos_ptr,         # *f32, positional embedding [N]
    scale,           # f32, scaling factor
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load Y tile
    Y_tile = tl.load(Y_ptr + off_m[:, None] * N + off_n[None, :], mask=(off_m[:, None] < M) & (off_n[None, :] < N), other=0.0)
    # Load pos embedding along N
    pos_tile = tl.load(pos_ptr + off_n, mask=off_n < N, other=0.0)  # shape [BLOCK_N]
    # Scale and add
    Y_tile = Y_tile + pos_tile[None, :] * scale
    # Store
    tl.store(Y_ptr + off_m[:, None] * N + off_n[None, :], Y_tile, mask=(off_m[:, None] < M) & (off_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features,             # [B, 1, H, W], bfloat16
        conv2d1_weight, conv2d1_bias,   # conv1: [384,1,3,3], [384]
        conv2d2_weight, conv2d2_bias,   # conv2: [384,384,3,3], [384]
        conv2d3_weight, conv3_bias,     # conv3: [384,384,3,3], [384]
        conv_out_weight,                 # [N, K] where N=d_model=1024, K=features after last conv (e.g., 384*H_out3*W_out3)
        positional_embedding,            # [max_source_positions, N], bfloat16
        embed_scale: float,              # float, e.g., sqrt(1024)=32.0
    ):
        # Convert inputs to float32 for Triton computations
        B = input_features.shape[0]
        H = input_features.shape[2]
        W = input_features.shape[3]

        # Conv1: in_channels=1 -> out_channels=384
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H + 2 * 1 - 3) // 2 + 1
        W_out1 = (W + 2 * 1 - 3) // 2 + 1
        x1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.float32, device=input_features.device)

        grid1 = (B, H_out1, W_out1, C_out1)
        conv3x3_stride2_gelu_nchw_inplace[grid1](
            input_features.float(), conv2d1_weight.float(), conv2d1_bias.float(), x1,
            B, 1, H, W, C_out1, H_out1, W_out1, 1, 1, 2, 1
        )

        # Conv2: in_channels=C_out1 -> out_channels=C_out2
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H_out1 + 2 * 1 - 3) // 2 + 1
        W_out2 = (W_out1 + 2 * 1 - 3) // 2 + 1

        x2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.float32, device=input_features.device)
        grid2 = (B, H_out2, W_out2, C_out2)
        conv3x3_stride2_gelu_nchw_inplace[grid2](
            x1, conv2d2_weight.float(), conv2d2_bias.float(), x2,
            B, C_out1, H_out1, W_out1, C_out2, H_out2, W_out2, 1, 1, 2, 1
        )

        # Conv3: in_channels=C_out2 ->


def run(*args):
    return ModelNew()(*args)
