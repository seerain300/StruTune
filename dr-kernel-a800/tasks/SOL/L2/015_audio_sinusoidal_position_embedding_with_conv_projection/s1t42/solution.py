import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Input: X(B, Cin, H, T) bfloat16
# Weight: W(Cout, Cin, 3, 3) bfloat16
# Bias: Bias(Cout) bfloat16
# Output: Y(B, Cout, H_out, T_out) bfloat16, with H_out = floor((H - 3)/2 + 1), T_out = floor((T - 3)/2 + 1)
@triton.jit
def conv2d_3x3_stride2_padding1_kernel(
    X_ptr,         # *const bfloat16
    W_ptr,         # *const bfloat16
    BIAS_ptr,      # *const bfloat16
    Y_ptr,         # *bfloat16
    B: tl.int32, Cin: tl.int32, H: tl.int32, T: tl.int32,
    Cout: tl.int32, H_out: tl.int32, T_out: tl.int32,
    BLOCK_C: tl.constexpr,
):
    # Grid dims: (B*H_out, tiles over Cout, T_out)
    pid_bh = tl.program_id(0)   # over B*H_out
    pid_c  = tl.program_id(1)   # tiles over Cout
    pid_t  = tl.program_id(2)   # over T_out

    b = pid_bh // H_out
    oh = pid_bh % H_out

    t_out_idx = pid_t
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # Accumulator for this (b, oh, t_out_idx) and tile of Cout
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for cin in range(0, Cin):
        for kh in range(0, 3):
            ih = oh + kh - 1  # equivalent to oh - kh + 1
            valid_h = (ih >= 0) & (ih < H)
            for kt in range(0, 3):
                it = t_out_idx + kt - 1
                valid_t = (it >= 0) & (it < T)
                # For valid, load X[b, cin, ih, it]; else 0
                x_ptr = X_ptr + b * (Cin * H * T) + cin * (H * T) + ih * T + it
                x_val = tl.load(x_ptr, mask=(valid_h & valid_t), other=0.0).to(tl.float32)

                # Load W[c, cin, kh, kt] for all c in tile
                for c_idx in range(0, BLOCK_C):
                    c = c_offsets[c_idx]
                    mask_c_elem = c < Cout
                    w_ptr = W_ptr + c * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt
                    w_val = tl.load(w_ptr, mask=mask_c_elem, other=0.0).to(tl.float32)
                    acc[c_idx] += x_val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias

    # Apply exact GELU: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Store to Y[b, c_offsets, oh, t_out_idx] as bfloat16
    y_base = Y_ptr + b * (Cout * H_out * T_out) + (c_offsets * (H_out * T_out)) + (oh * T_out) + t_out_idx
    tl.store(y_base, gelu.to(tl.bfloat16), mask=mask_c)


# Triton GELU (exact, erf-based) applied to a tensor Y (flattened)
# Input: Y_ptr, size N, output overwritten
@triton.jit
def gelu_erf_kernel(Y_ptr, N: tl.int32):
    pid = tl.program_id(0)
    if pid < N:
        val = tl.load(Y_ptr + pid, mask=True, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        gelu = 0.5 * val * (1.0 + tl.math.erf(val * inv_sqrt2))
        tl.store(Y_ptr + pid, gelu.to(tl.bfloat16), mask=True)


# Triton GEMM + positional embedding add for final linear projection
# X is (B, t_after_conv, K) flattened as (B*T_after, K). We pass strides accordingly.
# WT is (K, N) = conv_out_weight.T (given conv_out_weight is (N, K) in original, here we use its T to get (K, N)).
# POS is (T_after, N) positional embedding slice, scaled by embed_scale.
# Output Y is (B, T_after, N).
@triton.jit
def gemm_add_pos_kernel(
    X_ptr, WT_ptr, POS_ptr, Y_ptr,
    B: tl.int32, T_after: tl.int32, K: tl.int32, N: tl.int32,
    stride_Xb: tl.int32, stride_Xt: tl.int32, stride_Xk: tl.int32,
    stride_WTk: tl.int32, stride_WTn: tl.int32,
    stride_PosT: tl.int32, stride_PosN: tl.int32,
    embed_scale: tl.float32,
):
    # Grid dims: (B*T_after, tiles over N, tiles over K)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Row index (b, t_after)
    bt = pid_m
    b = bt // T_after
    t = bt % T_after

    # Output channel tile and K tile
    n_offsets = pid_n * 64 + tl.arange(0, 64)
    k_offsets = pid_k * 64 + tl.arange(0, 64)

    mask_n = n_offsets < N

    # Accumulator for this (b, t) row across N tile
    acc = tl.zeros((64,), dtype=tl.float32)

    # Loop over K in chunks
    # We need to load X row for this (b, t), i.e., X[b, t, k_offsets]
    # Note: X is flattened as (B*T_after, K). For a fixed bt, the row is contiguous in K.
    # Compute base pointer for this row: X_ptr + bt * stride_Xb; then advance by k_offsets * stride_Xk.
    # But since we flattened row-wise, stride_Xb is K and stride_Xt is 1 (conceptually), so:
    # For given bt, row base = X_ptr + bt * stride_Xb; then elements are at row_base + k_offsets * stride_Xk
    # However, to be robust, we use X as 2D via stride_Xb=stride across B*T_after rows, stride_Xt=0 (unused), stride_Xk=1 for elements:
    # In practice, when we pass X flattened, stride_Xb = K, stride_Xt = 0, stride_Xk = 1. So row base is X_ptr + bt * K, and k element is + k_offsets.
    # To be precise: we pass X_ptr as flattened (B*T_after*K), so stride_Xb = K, stride_Xt = 0, stride_Xk = 1. Here we don't need X row as we already know bt; we pass X as (B*T_after, K) contiguous.
    # In host, we pass X as (B, T_after, K) contiguous, and compute bt = b*T_after + t. Then X base for row is X_ptr + bt * K.
    # Triton expects strides for 1D; we treat X as flattened (B*T_after*K). So we can compute row base as X_ptr + bt * K (since K = T_after * stride_Xt ?): actually, we pass X as flattened (B*T_after, K) with stride_Xb = K and stride_Xt = 0.
    # To simplify: we pass X as (B*T_after, K) contiguous, so for bt, row_base = X_ptr + bt * K. Then k elements are at + k_offsets.
    # Therefore: we set stride_Xb = K, stride_Xt = 0 (unused), stride_Xk = 1.
    # However, to be robust, we compute bt row base as X_ptr + bt * K by passing X as flattened (B*T_after*K).
    # Implement by using X_ptr + bt * K + k_offsets. But X_ptr points to flattened; so row base is X_ptr + bt * K, then k element = + k_offsets.
    row_base = X_ptr + bt * K
    for k_start in range(0, K, 64):
        k_idx = k_start + k_offsets
        mask_k = k_idx < K
        x_row = tl.load(row_base + k_idx, mask=mask_k, other=0.0).to(tl.float32)  # shape (64,)

        # WT[k, n] where k = k_idx, n = n_offsets
        wt_ptr = WT_ptr + k_idx[:, None] * stride_WTk + n_offsets[None, :] * stride_WTn
        mask_k_n = mask_k[:, None] & mask_n[None, :]
        wt = tl.load(wt_ptr, mask=mask_k_n, other=0.0).to(tl.float32)  # shape (64, 64)
        acc += tl.sum(wt * x_row[:, None], axis=0)

    # Add scaled positional embedding: pos[b, t, n] = POS[t, n] * embed_scale
    pos_ptr = POS_ptr + t * stride_PosT + n_offsets * stride_PosN
    pos_val = tl.load(pos_ptr, mask=mask_n, other=0.0).to(tl.float32) * embed_scale
    acc += pos_val

    # Store to Y[b, t, n_offsets]
    y_ptr = Y_ptr + b * (T_after * N) + t * N + n_offsets
    tl.store(y_ptr, acc.to(tl.bfloat16), mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure CUDA and dtype
        assert input_features.is_cuda, "Input must be on CUDA device"
        assert conv2d1_weight.is_cuda and conv2d1_bias.is_cuda and conv2d2_weight.is_cuda and conv2d2_bias.is_cuda and conv2d3_weight.is_cuda and conv2d3_bias.is_cuda and conv_out_weight.is_cuda and positional_embedding.is_cuda, "All tensors must be on CUDA"

        # Input: (B, 1, 80, T)
        B, Cin, H, T = input_features.shape
        # Conv1: (1, 384, 3, 3) stride=2, padding=1
        Cout1 = conv2d1_weight.shape[0]
        H_out1 = H // 2  # floor((H - 3)/2 + 1) = floor(H/2) since padding
        T_out1 = T // 2
        x1 = torch.empty((B, Cout1, H_out1, T_out1), dtype=torch.bfloat16, device=input_features.device)

        BLOCK_C1 = 64
        grid1 = (B * H_out1, triton.cdiv(Cout1, BLOCK_C1), T_out1)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, Cin, H, T, Cout1, H_out1, T_out1, BLOCK_C=BLOCK_C1
        )

        # GELU1
        x1_flat = x1.reshape(-1)
        N1 = x1_flat.numel()
        gelu_erf_kernel[(N1,)](x1_flat)

        # Conv2: (384, 384, 3, 3)
        Cout2 = conv2d2_weight.shape[0]
        H_out2 = H_out1 // 2
        T_out2 = T_out1 // 2
        x2 = torch.empty((B, Cout2, H_out2, T_out2), dtype=torch.bfloat16, device=input_features.device)

        BLOCK_C2 = 64
        grid2 = (B * H_out2, triton.cdiv(Cout2, BLOCK_C2), T_out2)
        conv2d_3x3_stride2_padding1_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, Cout1, H_out1, T_out1, Cout2, H_out2, T_out2, BLOCK_C=BLOCK_C2
        )

        # GELU2
        x2_flat = x2.reshape(-1)
        N2 = x2_flat.numel()
        gelu_erf_kernel[(N2,)](x2_flat)

        # Conv3: (384, 384, 3, 3)
        Cout3 = conv2d3_weight.shape[0]
        H_out3 = H_out2 // 2
        T_out3 = T_out2 // 2
        x3 = torch.empty((B, Cout3, H_out3, T_out3), dtype=torch.bfloat16, device=input_features.device)

        BLOCK_C3 = 64
        grid3 = (B * H_out3, triton.cdiv(Cout3, BLOCK_C3), T_out3)
        conv2d_3x3_stride2_padding1_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, Cout2, H_out2, T_out2, Cout3, H_out3, T_out3, BLOCK_C=BLOCK_C3
        )

        # GELU3
        x3_flat = x3.reshape(-1)
        N3 = x3_flat.numel()
        gelu_erf_kernel[(N3,)](x3_flat)

        # After conv3: permute to (B, T_after_conv, C*F) with F=40, C=384
        B3, C, H3, T3 = x3.shape  # C=384, H3=20, T3=100
        K = C * 40  # 15360
        x3_perm = x3.permute(0, 3, 1, 2).contiguous().view(B3, T3, K)  # (B, t_after, 384*40)

        # Final GEMM: y = x3_perm @ conv_out_weight.T -> (B, t_after, N_out)
        N_out = conv_out_weight.shape[0]  # 1024
        # Prepare pointers: X as (B*T_after, K), WT = conv_out_weight.T as (K, N_out)
        B_after = B3
        t_after = T3
        X_flat = x3_perm.reshape(B_after * t_after, K).contiguous()  # shape: (B*t_after, K), dtype bfloat16
        WT = conv_out_weight.transpose(0, 1).contiguous()  # (K, N_out), dtype bfloat16

        # POS: positional_embedding is (max_source_positions, d_model). We need slice [:t_after, :N_out].
        # d_model = N_out = 1024. Make sure positional_embedding is on CUDA and dtype bfloat16.
        # Scale by embed_scale = sqrt(d_model)
        # We can construct POS as (t_after, N_out) by slicing positional_embedding[:t_after, :N_out].
        # Note: positional_embedding provided is (1500, 1024), but we only need t_after rows.
        # Create POS tensor (t_after, N_out)
        POS = positional_embedding[:t_after, :N_out].to(torch.bfloat16)

        # Allocate output Y: (B, t_after, N_out)
        Y = torch.empty((B_after, t_after, N_out), dtype=torch.bfloat16, device=X_flat.device)

        # Strides for X_flat: since it's flattened (B*T_after, K), stride along B*T_after is K, along K is 1. But we pass as 1D; Triton needs actual strides.
        # To handle arbitrary strides, we can recompute as 2D: X2D = X_flat.view(B*t_after, K) and pass strides. But Triton expects 1D base and element offsets. Simpler: pass as 1D and rely on contiguous layout.
        # We'll pass X_flat as 1D and compute per (b, t) row via bt index. For that, we set stride_Xb = K and stride_Xt = 0, stride_Xk = 1 conceptually by indexing k_offsets.
        # However, for robustness, we'll pass X_flat as 1D with actual element offsets: pass X_flat directly; Triton will treat it as a 1D array. For loads, we'll compute row base as X_ptr + bt * K and then k offsets.
        # To do that correctly, we need to repackage X_flat as a 2D tensor for proper strides. Simpler: we'll pass X_flat as 1D and rely on the fact that X_flat is contiguous and corresponds to (B*t_after, K) row-major layout by construction.

        # Launch GEMM + add positional embedding
        # grid = (B*t_after, tiles over N, tiles over K)
        grid = (B_after * t_after, triton.cdiv(N_out, 64), triton.cdiv(K, 64))
        gemm_add_pos_kernel[grid](
            X_flat, WT, POS, Y,
            B_after, t_after, K, N_out,
            X_flat.stride(0), 0, 1,  # stride_Xb, stride_Xt, stride_Xk
            WT.stride(0), WT.stride(1),
            POS.stride(0), POS.stride(1),
            embed_scale
        )

        return Y


def run(*args):
    return ModelNew()(*args)
