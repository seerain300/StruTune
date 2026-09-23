import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride2_nchw_inplace(
    x_ptr,          # *f32, input tensor [B, C_in, H, W]
    w_ptr,          # *f32, weight tensor [C_out, C_in, 3, 3]
    b_ptr,          # *f32, bias tensor [C_out]
    out_ptr,        # *f32, output tensor [B, C_out, H_out, W_out]
    B, C_in, H, W, C_out, H_out, W_out,
    stride_h, stride_w, pad_h, pad_w,
    IN_C: tl.constexpr,  # input channels (compile-time for loops)
):
    # Program IDs: we launch grid as (B, H_out, W_out, C_out)
    b = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    co = tl.program_id(3)

    # Accumulator for the output channel co (scalar in f32)
    acc = 0.0

    # Loop over input channels (compile-time bound IN_C), masked by actual C_in
    for ic in range(0, IN_C):
        ic_valid = ic < C_in
        # 3x3 neighborhood, compute input indices
        for kh in range(0, 3):
            ih = oh + kh - pad_h
            ih_valid = (ih >= 0) & (ih < H)
            for kw in range(0, 3):
                iw = ow + kw - pad_w
                iw_valid = (iw >= 0) & (iw < W)
                # mask for this location
                valid = ic_valid & ih_valid & iw_valid
                if valid:
                    # Compute input offset: ((b * C_in + ic) * H + ih) * W + iw
                    x_off = ((b * C_in + ic) * H + ih) * W + iw
                    # Load x value
                    x_val = tl.load(x_ptr + x_off)
                    # Load weight for (co, ic, kh, kw): layout [C_out, C_in, 3, 3]
                    w_off = co * (C_in * 9) + ic * 9 + kh * 3 + kw
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    # Add bias
    b_val = tl.load(b_ptr + co)
    acc += b_val

    # GELU (tanh approximation)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    tanh_arg = c * (acc + 0.044715 * x3)
    tanh_val = tl.math.tanh(tanh_arg)
    acc = 0.5 * acc * (1.0 + tanh_val)

    # Store to out[b, co, oh, ow]
    out_off = ((b * C_out + co) * H_out + oh) * W_out + ow
    tl.store(out_ptr + out_off, acc)


@triton.jit
def linear_gemm_no_bias_tanh_kernel(
    X_ptr,           # *f32, input [B, T, K]
    W_ptr,           # *f32, weight [N, K]
    Y_ptr,           # *f32, output [B, T, N]
    B, T, K, N,
    BLOCK_N: tl.constexpr,  # block size over N
    BLOCK_K: tl.constexpr,  # block size over K
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_start = tl.program_id(2) * BLOCK_N

    # Accumulator for this (b, t) across N block
    y_vec = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        # Accumulate contributions for each d in the block
        for di in range(0, BLOCK_N):
            d = n_start + di
            # Guard: only compute valid d < N
            if d < N:
                acc = 0.0
                # Dot product over K chunk
                for kk in range(0, BLOCK_K):
                    k = k_start + kk
                    if k < K:
                        # Load X[b, t, k]
                        x_off = (b * T + t) * K + k
                        x_val = tl.load(X_ptr + x_off)
                        # Load W[d, k]
                        w_off = d * K + k
                        w_val = tl.load(W_ptr + w_off)
                        acc += x_val * w_val
                y_vec[di] = acc

    # Store y_vec to Y[b, t, n_start:n_start+BLOCK_N]
    for di in range(0, BLOCK_N):
        d = n_start + di
        if d < N:
            y_off = (b * T * N + t * N + d)
            tl.store(Y_ptr + y_off, y_vec[di])


@triton.jit
def add_pos_embed_kernel(
    Y_ptr,           # *f32, output [B, T, N]
    Pos_ptr,         # *f32, positional embedding [N, T_pos]
    B, T, N, T_pos,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n = tl.program_id(2)
    # Load pos[n, t] and add to Y[b, t, n]
    pos_off = n * T_pos + t
    pos_val = tl.load(Pos_ptr + pos_off)
    y_off = (b * T * N + t * N + n)
    y_val = tl.load(Y_ptr + y_off)
    y_val += pos_val
    tl.store(Y_ptr + y_off, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features,             # [B, 1, 80, time_dim], dtype not fixed (will cast to f32)
        conv2d1_weight, conv2d1_bias,   # conv1: [384,1,3,3], [384]
        conv2d2_weight, conv2d2_bias,   # conv2: [384,384,3,3], [384]
        conv2d3_weight, conv3_bias,     # conv3: [384,384,3,3], [384]
        conv_out_weight,                 # [d_model, conv_out_dim] = [1024, conv_out_dim]; we will use only first d_model rows
        positional_embedding,            # [max_source_positions, d_model], bfloat16 (we will cast to f32 for compute)
        embed_scale: float,              # e.g., 32.0 (sqrt(1024))
    ):
        # Cast to float32 for Triton compute
        device = input_features.device
        B, Cin, H, W = input_features.shape  # Cin = 1
        # Conv1: in_channels=Cin -> out_channels=384
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H + 2 * 1 - 3) // 2 + 1
        W_out1 = (W + 2 * 1 - 3) // 2 + 1
        x1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.float32, device=device)

        # Launch conv1 kernel: grid = (B, H_out1, W_out1, C_out1)
        grid1 = (B, H_out1, W_out1, C_out1)
        conv3x3_stride2_nchw_inplace[grid1](
            input_features.float(), conv2d1_weight.float(), conv2d1_bias.float(), x1,
            B, Cin, H, W, C_out1, H_out1, W_out1, 2, 2, 1, 1,
            IN_C=1,
            num_warps=4, num_stages=2
        )

        # Conv2: in_channels=C_out1 -> out_channels=C_out2
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H_out1 + 2 * 1 - 3) // 2 + 1
        W_out2 = (W_out1 + 2 * 1 - 3) // 2 + 1

        x2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.float32, device=device)
        grid2 = (B, H_out2, W_out2, C_out2)
        conv3x3_stride2_nchw_inplace[grid2](
            x1, conv2d2_weight.float(), conv2d2_bias.float(), x2,
            B, C_out1, H_out1, W_out1, C_out2, H_out2, W_out2, 2, 2, 1, 1,
            IN_C=C_out1,  # 384
            num_warps=4, num_stages=2
        )

        # Conv3: in_channels=C_out2 -> out_channels=C_out3
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = (H_out2 + 2 * 1 - 3) // 2 + 1
        W_out3 = (W_out2 + 2 * 1 - 3) // 2 + 1

        x3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.float32, device=device)
        grid3 = (B, H_out3, W_out3, C_out3)
        conv3x3_stride2_nchw_inplace[grid3](
            x2, conv2d3_weight.float(), conv3_bias.float(), x3,
            B, C_out2, H_out2, W_out2, C_out3, H_out3, W_out3, 2, 2, 1, 1,
            IN_C=C_out2,  # 384
            num_warps=4, num_stages=2
        )

        # Reshape: (B, C_out3, H_out3, W_out3) -> (B, W_out3, C_out3*H_out3) then (B, T, K)
        # We need T = W_out3 and K = C_out3 * H_out3 * W_out3. The original code sets T = time_after_conv, but we'll follow W_out3 for consistency with reshape in helper. If time_after_conv differs, the original helper sets conv_out_dim accordingly; here we will use W_out3 as T.
        T = W_out3
        K = C_out3 * H_out3 * W_out3

        # Flatten x3 to [B, T, K] by permute to [B, W_out3, H_out3, C_out3] and view
        x3_perm = x3.permute(0, 3, 2, 1).contiguous()  # [B, C_out3, H_out3, W_out3]
        x3_flat = x3_perm.view(B, T, K).contiguous()

        # Prepare conv_out_weight for linear projection: [N, K], where N=d_model=1024
        # In the provided helper, conv_out_dim=3840; but here we use the actual K.
        # We can slice conv_out_weight to [N, min(K, conv_out_dim)] to be safe. If K > conv_out_dim, we use the first conv_out_dim features.
        # Since helper sets conv_out_dim=3840, and in our case K is much larger, we use the first conv_out_dim features. For correctness, we must ensure conv_out_dim matches K; given helper code, it does. So we slice to N=1024, K_linear=K.
        N = 1024
        conv_out_weight_triton = conv_out_weight.float()  # [N, K_linear]
        # Ensure second dim is at least N; if not, raise. In the helper, it's [1024, 3840] for the test cases.
        if conv_out_weight_triton.shape[0] != N:
            raise RuntimeError(f"conv_out_weight first dim must be N=1024, got {conv_out_weight_triton.shape[0]}")
        if conv_out_weight_triton.shape[1] < K:
            raise RuntimeError(f"conv_out_weight second dim must be >= K={K}, got {conv_out_weight_triton.shape[1]}")
        # Y = X @ W  (no bias); we will implement this in Triton.

        # Allocate output Y [B, T, N]
        Y = torch.empty((B, T, N), dtype=torch.float32, device=device)

        # Launch linear kernel: grid = (B, T, ceil_div(N, BLOCK_N))
        BLOCK_N = 128
        BLOCK_K = 256
        grid_linear = (B, T, (N + BLOCK_N - 1) // BLOCK_N)
        linear_gemm_no_bias_tanh_kernel[grid_linear](
            x3_flat, conv_out_weight_triton, Y,
            B, T, K, N,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale
        Y = Y * embed_scale

        # Add positional embedding: pos is [max_source_positions, N], we add first T rows. Since T == time_after_conv from helper, T == W_out3.
        # Cast pos to float32 for compute
        pos = positional_embedding.float()  # [T, N], we need [N, T] for kernel
        pos = pos.permute(1, 0).contiguous()  # [N, T]

        # Launch add positional embedding kernel: grid = (B, T, N)
        grid_pos = (B, T, N)
        add_pos_embed_kernel[grid_pos](
            Y, pos,
            B, T, N, T,
            num_warps=4, num_stages=2
        )

        # Return Y (already scaled and with pos added)
        return Y


def run(*args):
    return ModelNew()(*args)
