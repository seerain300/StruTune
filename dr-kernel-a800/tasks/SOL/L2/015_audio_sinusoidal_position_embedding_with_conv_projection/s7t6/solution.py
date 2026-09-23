import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_bias_gelu(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, C_in, H, W,
    C_out, H_out, W_out,
    stride_x_b, stride_x_ci, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_b, stride_y_co, stride_y_h, stride_y_w,
    BLOCK_CO: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    """
    x: [B, C_in, H, W], bfloat16
    w: [C_out, C_in, 3, 3], bfloat16
    bias: [C_out], bfloat16
    y: [B, C_out, H_out, W_out], bfloat16
    stride-2, pad-1 conv, then GELU (exact erf-based).
    """
    # Program IDs
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute tile offsets
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    # Flatten spatial tiles into a single index over HW_out = H_out * W_out
    hw_offset = pid_hw * BLOCK_HW
    hw_total = H_out * W_out
    num_hw_in_tile = tl.minimum(BLOCK_HW, hw_total - hw_offset)
    # Construct 1D hw indices for this tile
    hw_idx = hw_offset + tl.arange(0, num_hw_in_tile)
    # Limit hw_idx to hw_total
    mask_hw = hw_idx < hw_total

    # Map 1D hw_idx to 2D (oh, ow)
    oh = hw_idx // W_out
    ow = hw_idx % W_out

    # Initialize accumulator for output tile
    out_tile = tl.zeros((BLOCK_CO, num_hw_in_tile), dtype=tl.float32)

    # Loop over input channels and kernel elements
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # Compute input coordinates with padding=1 and stride=2
                ih = oh * 2 - 1 + kh
                iw = ow * 2 - 1 + kw
                # Mask for valid input coordinates
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask_hw
                # Compute pointers for x[b, ci, ih, iw]
                x_ptrs = x_ptr + pid_b * stride_x_b + ci * stride_x_ci + ih[:, None] * stride_x_h + iw[None, :] * stride_x_w
                # Load input with mask, zeros otherwise
                x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)  # bfloat16
                # Load weight slice w[co, ci, kh, kw] for all co in tile
                w_ptrs = w_ptr + co_offsets[:, None] * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw
                w_vals = tl.load(w_ptrs, mask=(co_offsets < C_out), other=0.0)  # bfloat16
                # Accumulate: out_tile[co, hw] += sum_ci sum_kh sum_kw x_vals * w_vals
                # Note: x_vals shape (num_hw_in_tile,), w_vals shape (BLOCK_CO,), broadcasting to (BLOCK_CO, num_hw_in_tile)
                out_tile += w_vals[:, None] * x_vals[None, :]

    # Add bias: broadcast bias across spatial tile
    bias_vals = tl.load(bias_ptr + co_offsets, mask=(co_offsets < C_out), other=0.0)  # bfloat16
    out_tile = out_tile + bias_vals[:, None]

    # Apply GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    gelu = 0.5 * out_tile * (1.0 + tl.erf(out_tile * inv_sqrt2))

    # Store results to y[b, co, oh, ow]
    # Build y pointers for the tile: shape (BLOCK_CO, num_hw_in_tile)
    y_ptrs = y_ptr + pid_b * stride_y_b + co_offsets[:, None] * stride_y_co + oh[None, :] * stride_y_h + ow[None, :] * stride_y_w
    store_mask = (co_offsets[:, None] < C_out) & (mask_hw[None, :])
    tl.store(y_ptrs, gelu, mask=store_mask)


def _run_triton_conv(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, embed_scale: float) -> torch.Tensor:
    """
    Helper to run the Triton conv2d stride=2, pad=1, bias, GELU. Returns y in bfloat16.
    x: [B, C_in, H, W], bfloat16
    w: [C_out, C_in, 3, 3], bfloat16
    bias: [C_out], bfloat16
    """
    B, C_in, H, W = x.shape
    C_out = w.shape[0]
    # Compute output sizes: H_out = floor((H + 2*1 - 3 - 1)/2 + 1) = floor((H - 1)/2 + 1)
    H_out = (H + 2 - 3 - 1) // 2 + 1
    W_out = (W + 2 - 3 - 1) // 2 + 1

    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=torch.bfloat16)

    # Strides
    stride_x_b, stride_x_ci, stride_x_h, stride_x_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw = w.stride()
    stride_y_b, stride_y_co, stride_y_h, stride_y_w = y.stride()

    # Tiling parameters: moderate to balance occupancy and register usage
    BLOCK_CO = 32  # number of output channels per tile (C_out is 384; grid over channels)
    BLOCK_HW = 256  # number of spatial positions per tile (H_out*W_out ~ 800; grid over spatial tiles)

    grid = (
        B,
        triton.cdiv(C_out, BLOCK_CO),
        triton.cdiv(H_out * W_out, BLOCK_HW),
    )

    conv2d_stride2_pad1_bias_gelu[grid](
        x, w, bias, y,
        B, C_in, H, W,
        C_out, H_out, W_out,
        stride_x_b, stride_x_ci, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_y_b, stride_y_co, stride_y_h, stride_y_w,
        BLOCK_CO=BLOCK_CO, BLOCK_HW=BLOCK_HW,
        num_warps=4, num_stages=2,
    )
    # Scale by embed_scale (constant 32 in the given setup)
    # Apply in-place scaling to y: y *= embed_scale
    y_flat = y.view(-1)
    N_elems = y_flat.numel()
    grid_scale = (triton.cdiv(N_elems, 1024),)
    scale_elementwise_kernel[grid_scale](y_flat, float(embed_scale), N_elems, num_warps=4, num_stages=2)
    return y


# Triton kernels for elementwise ops (simple placeholders if needed)
@triton.jit
def scale_elementwise_kernel(X_ptr, Y_ptr, N, scale, N_elems: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = x * scale
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def add_pos_emb_kernel(Y_ptr, POS_ptr, Z_ptr, S, N, N_elems: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    # Y is flattened [B*S*N], POS is [S, N] flattened. We need to map offs to (b, s, n) and add POS[s, n].
    # For simplicity, we assume Y_ptr is contiguous and N_elems = S*N*B, though we pass N_elems from host.
    # Recompute N from S*N is not available in kernel, so we rely on host to pass correct N_elems.
    y_val = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    pos_index = offs % N  # second dim (n)
    s_index = offs // N  # first dim (s). With N_elems = S*N, this recovers s; but we need batch index as well.
    # To include batch, we can't recover without 3D layout; since we flatten [B, S, N], we can compute b,s,n via:
    # We instead precompute per-batch in host: launch kernels per batch. For this elementwise, we don't need batch indexing.
    pos_val = tl.load(POS_ptr + s_index * N + pos_index, mask=mask, other=0.0)
    z = y_val + pos_val
    tl.store(Z_ptr + offs, z, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args order matches get_inputs: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[6]
        conv2d2_weight = args[3]
        conv2d2_bias = args[7]
        conv2d3_weight = args[5]
        conv2d3_bias = args[8]
        conv_out_weight = args[2]
        positional_embedding = args[9]
        embed_scale = float(args[10])

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x = _run_triton_conv(input_features, conv2d1_weight, conv2d1_bias, embed_scale)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        # For conv2d, we need x shape [B, 384, H2, W2]; compute H2, W2 as output sizes with stride=2, pad=1
        B, C1, H1, W1 = x.shape
        C2 = conv2d2_weight.shape[0]
        H2 = (H1 + 2 - 3 - 1) // 2 + 1
        W2 = (W1 + 2 - 3 - 1) // 2 + 1
        x = _run_triton_conv(x, conv2d2_weight, conv2d2_bias, embed_scale)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        B2, C3, H3, W3 = x.shape
        C4 = conv2d3_weight.shape[0]
        H4 = (H3 + 2 - 3 - 1) // 2 + 1
        W4 = (W3 + 2 - 3 - 1) // 2 + 1
        x = _run_triton_conv(x, conv2d3_weight, conv2d3_bias, embed_scale)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        # Given the conv pipeline, x has shape [B, 384, 10, T/8]. The original code uses reshape to [B, time_after_conv, 3840].
        # Here, time_after_conv = T/8. channels*freq = 384*10 = 3840.
        B_final, C_final, H_final, W_final = x.shape  # W_final should equal time_after_conv for given axes. We cannot access axes here.
        # We will proceed with permutation to match original: x.permute(0, 3, 1, 2).contiguous().view(B, W_final, C_final*H_final)
        # However, we don't have W_final and H_final labels. The original code uses specific time_after_conv derived from input time_dim.
        # Since we cannot know T/8 here without external input, we instead rely on the fact that forward(*args) passes positional_embedding of size [time_after_conv, 1024], implying time_after_conv is known. We will recompute it via provided positional_embedding tensor shape.
        # From the get_inputs, positional_embedding is [max_source_positions, d_model], but here the positional_embedding provided is of correct size. We can infer time_after_conv from its first dimension.

        # The original code does: x.permute(0, 3, 1, 2).contiguous().view(B, t, c*f)
        # After the last conv, x shape is [B, 384, 10, T/8]. We need to permute to [B, T/8, 384, 10], then flatten channels*freq.
        # But get_inputs returns positional_embedding with correct size, so we can infer S (time_after_conv) from positional_embedding.size(0). Let's denote S as pos_emb_S.
        pos_emb_S = positional_embedding.shape[0]

        # Permute to [B, T/8, C, F] then view(B, S, C*F)
        x_perm = x.permute(0, 3, 1, 2).contiguous()
        # We need to reshape to [B, S, 3840]. The code originally assumes S equals x_perm.shape[1]. In our pipeline, S should be T/8.
        # We cannot compute T/8 from here; however, positional_embedding has shape [pos_emb_S, 1024]. The original code uses pos_emb_S == time_after_conv.
        # Since we cannot infer T/8 from inputs, we will instead construct the output by assuming S = pos_emb_S (which equals time_after_conv). This is consistent with original get_inputs providing pos_emb of correct size.

        B, T_div8, C, F = x_perm.shape
        S = pos_emb_S  # from positional_embedding
        y = x_perm.view(B, S, C * F)

        # Linear projection to d_model=1024 (no bias in original). conv_out_weight shape [1024, 3840]
        # In our args, conv_out_weight is provided as [d_model=1024, conv_out_dim=3840]. We need to apply y @ conv_out_weight.T
        conv_out_weight = args[2]  # [1024, 3840]
        # Triton matmul: y_flat [B*S, 3840], w_t [3840, 1024], output [B*S, 1024]
        y_flat = y.contiguous().view(-1, y.shape[-1])  # [B*S, 3840]
        w_t = conv_out_weight.t().contiguous()        # [3840, 1024]
        B_times_S, K = y_flat.shape
        N = w_t.shape[1]  # 1024

        y_out_flat = torch.empty((B_times_S, N), device=y.device, dtype=torch.bfloat16)

        # Launch Triton GEMM
        BLOCK_N = 128
        BLOCK_K = 64
        grid_mm = (B_times_S, triton.cdiv(N, BLOCK_N))
        linear_matmul_kernel[grid_mm](
            y_flat, w_t, y_out_flat,
            B_times_S, K, N,
            K, w_t.stride(1),  # K stride for w_t
            w_t.stride(0), w_t.stride(1),
            y_out_flat.stride(0), y_out_flat.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        y_out = y_out_flat.view(B, S, N)  # [B, S, 1024]

        # Scale by embed_scale (host embed_scale is sqrt(1024) = 32)
        y_out_flat = y_out.view(-1)
        N_elems = y_out_flat.numel()
        grid_scale = (triton.cdiv(N_elems, 1024),)
        scale_elementwise_kernel[grid_scale](y_out_flat, float(embed_scale), N_elems, num_warps=4, num_stages=2)

        # Add positional embedding [S, N] broadcast over batch
        pos_emb = positional_embedding[:S, :].contiguous().to(torch.bfloat16)  # [S, N]
        N_elems_add = S * N
        grid_add = (triton.cdiv(N_elems_add, 1024),)
        add_pos_emb_kernel[grid_add](y_out_flat, pos_emb.view(-1), y_out_flat, S, N, N_elems_add, num_warps=4, num_stages=2)

        return y_out_flat.view(B, S, N)

# Triton GEMM kernel for y_flat @ w_t
@triton.jit
def linear_matmul_kernel(X_ptr, W_ptr, Y_ptr,
                          M, K, N,
                          stride_x_m, stride_x_k,
                          stride_w_k, stride_w_n,
                          stride_y_m, stride_y_n,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    X: [M, K] row-major
    W: [K, N] row-major (we pass W_t)
    Y: [M, N] row-major
    Compute Y = X @ W.
    """
    pid_m = tl.program_id(0)  # over rows
    pid_n = tl.program_id(1)  # over output columns
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # Accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        mask_x = (pid_m < M) & (k_idx < K)
        x_row_ptrs = X_ptr + pid_m * stride_x_m + k_idx * stride_x_k
        x_vals = tl.load(x_row_ptrs, mask=mask_x, other=0.0)  # bfloat16
        # Load W[k, offs_n] as vector
        w_ptrs = W_ptr + k_idx[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
        w_vals = tl.load(w_ptrs, mask=(offs_n[None, :] < N), other=0.0)  # bfloat16
        # Fused multiply-add into accumulator
        acc += tl.sum(w_vals * x_vals[None, :], axis=0)  # sum over K-block

    # Store result
    y_ptrs = Y_ptr + pid_m * stride_y_m + offs_n * stride_y_n
    mask_y = (pid_m < M) & (offs_n < N)
    tl.store(y_ptrs, acc, mask=mask_y)


def run(*args):
    return ModelNew()(*args)
