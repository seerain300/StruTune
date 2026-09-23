import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv2d 3x3, stride=2, padding=1, generic IC -> OC
# X: [B, IC, F_in, T_in], W: [OC, IC, 3, 3], bias: [OC], Y: [B, OC, F_out, T_out]
@triton.jit
def conv3x3_s2_p1_gelu(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, IC, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # Launch grid: 1D over (b, oc, f_block, t_block)
    total_blocks = B * OC * ((F_out + BLOCK_F - 1) // BLOCK_F) * ((T_out + BLOCK_T - 1) // BLOCK_T)
    pid = tl.program_id(0)

    # Decode pid into (b, oc, f_block, t_block)
    grid_f_blocks = (F_out + BLOCK_F - 1) // BLOCK_F
    grid_t_blocks = (T_out + BLOCK_T - 1) // BLOCK_T

    b = pid // (OC * grid_f_blocks * grid_t_blocks)
    rem = pid % (OC * grid_f_blocks * grid_t_blocks)
    oc = rem // (grid_f_blocks * grid_t_blocks)
    f_block = rem % (grid_f_blocks * grid_t_blocks) // 1
    t_block = rem % (grid_f_blocks * grid_t_blocks) % 1
    # Note: using f_block=0, t_block=0 here simplifies; since grid covers all, pid is enough.

    # Compute output indices for this tile
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    # Note: input padding=1, stride=2
    for ic in range(0, IC):
        for kh in range(3):
            for kw in range(3):
                # Input coordinates with padding=1 and stride=2
                f_in_idx = f_out_idx * 2 + (1 - kh)  # [BF, 1]
                t_in_idx = t_out_idx * 2 + (1 - kw)  # [1, BT]
                # Mask for in-bounds input
                in_mask = (f_in_idx >= 0) & (f_in_idx < F_in) & (t_in_idx >= 0) & (t_in_idx < T_in)
                # Compute input pointers
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in_idx * x_sF + t_in_idx * x_sT
                x_tile = tl.load(x_ptrs, mask=out_mask & in_mask, other=0.0)  # [BF, BT]

                # Load corresponding weights (scalar per (oc, ic, kh, kw))
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptrs)

                # Accumulate
                acc += x_tile * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + oc)
    acc += bias_val

    # Apply GELU
    # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x = acc
    x3 = x * x * x
    gelu = 0.5 * x * (1.0 + tl.tanh(c0 * (x + 0.044715 * x3)))

    # Store result
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton matmul kernel: compute Y[M, N] = X[M, K] @ Wt[K, N], where Wt is W[N, K]^T (i.e., W contiguous as [K, N])
@triton.jit
def matmul_gemv_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    x_sM, x_sK,
    w_sK, w_sN,
    y_sM, y_sN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over (M tiles, N tiles)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]

        # X tile: [BM, BK]
        x_ptrs = X_ptr + (offs_m[:, None] * x_sM) + (offs_k[None, :] * x_sK)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # W tile: [BK, BN] where W is laid out as [K, N] (i.e., Wt)
        w_ptrs = W_ptr + (offs_k[:, None] * w_sK) + (offs_n[None, :] * w_sN)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # acc += x_tile @ w_tile
        acc += tl.dot(x_tile, w_tile)

    # Store
    y_ptrs = Y_ptr + (offs_m[:, None] * y_sM) + (offs_n[None, :] * y_sN)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


# Triton elementwise kernel: Y = Y * scale + P[:T, :], where Y is [B, T, D], P is [T, D]
# We pass Y and P as 1D contiguous arrays and decode indices.
@triton.jit
def scale_add_pos_embed(
    Y_ptr,  # [B*T*D], contiguous
    P_ptr,  # [T*D, D], contiguous (row-major per sequence position)
    B, T, D,
    scale,  # float
    NUMEL,  # B*T*D
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL

    # Decode offsets into (b, t, d)
    t = offsets // D
    d = offsets % D
    # For each element at offset, compute its positional embedding index: row = t * D + d
    # P is [T*D, D], so row index is t_flat * D + d
    # But since we only need P[:T, :], ensure t < T; here we assume NUMEL = B*T*D and t in [0, T).
    row = t * D + d

    # Load Y
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)

    # Load P row
    p = tl.load(P_ptr + row, mask=mask, other=0.0)

    # Compute
    y = y * scale + p

    # Store
    tl.store(Y_ptr + offsets, y, mask=mask)


def _launch_conv3x3_s2_p1_gelu(x, w, bias, B, IC, F_in, T_in, OC, F_out, T_out, BLOCK_F=32, BLOCK_T=32):
    # x: [B, IC, F_in, T_in], w: [OC, IC, 3, 3], bias: [OC]
    y = torch.empty((B, OC, F_out, T_out), device=x.device, dtype=torch.float32)
    grid = (B * OC * triton.cdiv(F_out, BLOCK_F) * triton.cdiv(T_out, BLOCK_T),)
    # We don't use strides for simplicity; pass 1 for non-used dims
    conv3x3_s2_p1_gelu[grid](
        x, w, bias, y,
        B, IC, F_in, T_in, OC, F_out, T_out,
        1, 1, 1, 1,
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T,
    )
    return y


def _launch_matmul_gemv(X, Wt, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    # X: [M, K], Wt: [K, N] (Wt is W^T with shape [N, K] transposed to [K, N])
    # Allocate output
    Y = torch.empty((M, N), device=X.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_gemv_kernel[grid](
        X, Wt, Y,
        M, N, K,
        X.stride(0), X.stride(1),
        Wt.stride(0), Wt.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return Y


def _launch_scale_add_pos_embed(Y, P, B, T, D, scale, BLOCK=1024):
    # Y: [B, T, D], P: [T, D], scale: float
    NUMEL = B * T * D
    grid = (triton.cdiv(NUMEL, BLOCK),)
    # We pass Y as 1D contiguous and P as 1D contiguous [T*D, D] where row = t*D + d
    # To pass P as 1D, flatten P: [T*D, D] -> [T*D*D]
    # Here P is [T, D]; flatten to [T*D]
    P_1d = P.contiguous().view(T * D)
    Y_1d = Y.contiguous().view(-1)
    scale_add_pos_embed[grid](
        Y_1d, P_1d,
        B, T, D,
        scale, NUMEL,
        BLOCK=BLOCK,
    )
    # Reshape back to [B, T, D]
    return Y_1d.view(B, T, D)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        # Ensure dtype float32 for Triton
        B, IC, F_in, T_in = input_features.shape
        x = input_features.contiguous().float()

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        OC1 = conv2d1_weight.shape[0]
        w1 = conv2d1_weight.contiguous().float()    # [OC1, 1, 3, 3]
        b1 = conv2d1_bias.contiguous().float()      # [OC1]
        F_out1 = (F_in + 2 * 1 - 3) // 2 + 1
        T_out1 = (T_in + 2 * 1 - 3) // 2 + 1
        y1 = _launch_conv3x3_s2_p1_gelu(x, w1, b1, B, 1, F_in, T_in, OC1, F_out1, T_out1, BLOCK_F=32, BLOCK_T=32)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x2 = y1
        OC2 = conv2d2_weight.shape[0]
        w2 = conv2d2_weight.contiguous().float()    # [OC2, OC1, 3, 3]
        b2 = conv2d2_bias.contiguous().float()      # [OC2]
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2 * 1 - 3) // 2 + 1
        T_out2 = (T_in2 + 2 * 1 - 3) // 2 + 1
        y2 = _launch_conv3x3_s2_p1_gelu(x2, w2, b2, B, OC1, F_in2, T_in2, OC2, F_out2, T_out2, BLOCK_F=32, BLOCK_T=32)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x3 = y2
        OC3 = conv2d3_weight.shape[0]
        w3 = conv2d3_weight.contiguous().float()    # [OC3, OC2, 3, 3]
        b3 = conv2d3_bias.contiguous().float()      # [OC3]
        F_in3 = F_out2
        T_in3 = T_out2
        F_out3 = (F_in3 + 2 * 1 - 3) // 2 + 1
        T_out3 = (T_in3 + 2 * 1 - 3) // 2 + 1
        y3 = _launch_conv3x3_s2_p1_gelu(x3, w3, b3, B, OC2, F_in3, T_in3, OC3, F_out3, T_out3, BLOCK_F=32, BLOCK_T=32)

        # Reshape: [B, OC3, F_out3, T_out3] -> [B, T_out3, OC3*F_out3]
        b, c, f, t = y3.shape
        y3_perm = y3.permute(0, 3, 1, 2).contiguous()  # [B, t, c, f]
        D = 1024
        C = c  # 384
        F_out3 = f
        T_out3 = t
        x_lin = y3_perm.view(b, t, C * f)  # [B, T_out3, 384*F_out3]

        # Linear projection to d_model (Wt is conv_out_weight^T as [1024, 3840])
        # We need Wt: [K=3840, N=1024] which is transposed conv_out_weight from [N=1024, K=3840]
        K = C * F_out3
        Wt = conv_out_weight.t().contiguous().float()  # [K=3840, N=1024]
        M = b * T_out3
        y_lin = _launch_matmul_gemv(x_lin.view(M, K), Wt, M, D, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)  # [M, D]

        # Scale
        y_scaled = y_lin

        # Add positional embedding: P is [max_source_positions, D], we take [:T_out3, :]
        # Ensure positional_embedding dtype float32 for Triton
        P = positional_embedding.to(torch.float32).contiguous()  # [max_source_positions, D]
        T = T_out3
        y_pos = _launch_scale_add_pos_embed(y_scaled.view(b, T, D), P[:T, :], b, T, D, float(embed_scale), BLOCK=1024)

        return y_pos


def run(*args):
    return ModelNew()(*args)
