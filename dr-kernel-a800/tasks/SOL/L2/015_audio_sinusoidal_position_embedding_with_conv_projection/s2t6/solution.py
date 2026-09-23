import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton Conv2d stage: computes a single output element y[b, oc, oh, ow].
# Input x: (B, 1, IH, IW), weight w: (OC, 1, 3, 3), stride=2, padding=1.
@triton.jit
def conv2d_stage_single_kernel(
    x_ptr,          # *fp32, (B, 1, IH, IW)
    w_ptr,          # *fp32, (OC, 1, 3, 3)
    b_ptr,          # *fp32, (OC,) or None if no bias
    y_ptr,          # *fp32, (B, OC, OH, OW)
    B: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OC: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_oc, y_stride_h, y_stride_w,
    has_bias: tl.constexpr,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Sum over input channel and 3x3 kernel taps (C_in=1 here, but keep generality)
    for ci in range(0, 1):  # C_in is fixed to 1 by get_inputs; loop kept minimal
        for kh in range(0, 3):
            ih = 2 * oh + kh - 1  # stride=2, padding=1
            valid_h = (ih >= 0) & (ih < IH)
            for kw in range(0, 3):
                iw = 2 * ow + kw - 1
                valid_w = (iw >= 0) & (iw < IW)
                valid = valid_h & valid_w

                # Load input x[b, 0, ih, iw]
                x_offset = b * x_stride_b + 0 * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0).to(tl.float32)

                # Load weight w[oc, 0, kh, kw]
                w_offset = oc * w_stride_oc + 0 * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_offset).to(tl.float32)

                acc += x_val * w_val

    # Add bias if present
    if has_bias:
        b_val = tl.load(b_ptr + oc).to(tl.float32)
        acc += b_val

    # Store y[b, oc, oh, ow]
    y_offset = b * y_stride_b + oc * y_stride_oc + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_offset, acc)


# Triton GELU (tanh approximation) over 1D flattened tensor.
@triton.jit
def gelu_kernel_1d(
    in_ptr,      # *fp32, input flattened
    out_ptr,     # *fp32, output flattened
    n_elements: tl.constexpr,
):
    idx = tl.program_id(0)
    if idx >= n_elements:
        return
    x = tl.load(in_ptr + idx).to(tl.float32)
    # tanh approximation for GELU
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    x3 = x * x * x
    t = x + c * x3
    gelu = 0.5 * x * (1.0 + tl.math.tanh(sqrt_2_over_pi * (t)))
    tl.store(out_ptr + idx, gelu)


# Triton linear projection + scaling + positional embedding.
# Input X_flat: (B*T, N), W: (M=1024, N), pos_embed: (MAX_POS, M).
# Output Y_flat: (B*T, M). For each (b, t), compute Y[b*T + t, :] = X[b, t, :] @ W^T * embed_scale + pos_embed[t, :].
@triton.jit
def linear_proj_pos_kernel(
    x_flat_ptr,           # *fp32, flattened (B*T, N)
    w_ptr,                # *fp32, (M=1024, N)
    pos_embed_ptr,        # *fp32, (MAX_POS, M)
    out_flat_ptr,         # *fp32, (B*T, M)
    B: tl.constexpr,      # batch size
    T: tl.constexpr,      # time_after_conv
    N: tl.constexpr,      # channels*freq = 384*40 = 15360
    M: tl.constexpr,      # d_model = 1024
    MAX_POS: tl.constexpr,  # max_source_positions = 1500
    embed_scale: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. B*T-1
    # vectorize over M in tiles
    for m_start in range(0, M, 128):
        # Compute current output vector offset for this (b, t)
        # We'll write contiguous M elements
        # Create m indices
        m_idx = m_start + tl.arange(0, 128)
        mask_m = m_idx < M

        # Accumulator for current tile
        acc = tl.zeros((128,), dtype=tl.float32)

        # Reduce over N in tiles
        for n_start in range(0, N, 1024):
            n_idx = n_start + tl.arange(0, 1024)
            mask_n = n_idx < N

            # Load X_row[b, t, n_idx] where row = pid
            # x_flat_ptr is laid out as row-major over (B*T, N)
            row_base = pid * N
            x_vals = tl.load(x_flat_ptr + row_base + n_idx, mask=mask_n, other=0.0).to(tl.float32)

            # Load W[:, n_idx] as a (M_tile, N_tile) block
            # We need W[m, n] for m in m_idx and n in n_idx.
            # Implement as two-dimensional loads with broadcast:
            w_block = tl.load(w_ptr + (m_idx[:, None] * N + n_idx[None, :]), mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

            # Accumulate: dot(x_vals, w_block) along N dimension
            # x_vals shape: (1024,), w_block shape: (128, 1024)
            # acc += sum over k of x_vals[k] * w_block[:, k]
            for k in range(0, 1024):
                acc += x_vals[k] * w_block[:, k]

        # Scale and add positional embedding for this (b, t)
        # Compute t = pid % T
        t_idx = pid % T
        # Slice pos_embed to the first MAX_POS rows if needed
        pos_row_ptr = pos_embed_ptr + t_idx * M + m_idx
        pos_vals = tl.load(pos_row_ptr, mask=mask_m, other=0.0).to(tl.float32)
        acc = acc * embed_scale + pos_vals

        # Store to out_flat at this row
        out_row_base = pid * M
        out_ptrs = out_flat_ptr + out_row_base + m_idx
        tl.store(out_ptrs, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # We expect: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        # Extract inputs. The original Model.forward signature is (*args), so we take them as-is.
        # We need to reconstitute shapes and launch Triton kernels.
        # Note: We assume args follow the same order as get_inputs and run function.
        # args[0] is input_features (B, 1, 80, time_dim), bf16
        # args[1..6] are weights/biases for convs: (w1, b1, w2, b2, w3, b3)
        # args[7] is conv_out_weight (1024, 384*40)
        # args[8] is positional_embedding (1500, 1024), bf16
        # args[9] is embed_scale (float)

        # Ensure Triton available
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch ops if Triton not available (but evaluation requires Triton-only).
            # However, we will try to launch Triton kernels.
            raise RuntimeError("Triton not available")

        # Parse inputs
        input_features = args[0]  # (B, 1, 80, T_in) bf16
        w1 = args[1]  # (384, 1, 3, 3) bf16
        b1 = args[2]  # (384) bf16
        w2 = args[3]  # (384, 384, 3, 3) bf16
        b2 = args[4]  # (384) bf16
        w3 = args[5]  # (384, 384, 3, 3) bf16
        b3 = args[6]  # (384) bf16
        conv_out_weight = args[7]  # (1024, 384*40) bf16
        positional_embedding = args[8]  # (1500, 1024) bf16
        embed_scale = args[9]  # float

        # Convert all to fp32 for robust Triton computation
        input_features = input_features.to(torch.float32)
        w1 = w1.to(torch.float32)
        b1 = b1.to(torch.float32)
        w2 = w2.to(torch.float32)
        b2 = b2.to(torch.float32)
        w3 = w3.to(torch.float32)
        b3 = b3.to(torch.float32)
        conv_out_weight = conv_out_weight.to(torch.float32)
        positional_embedding = positional_embedding.to(torch.float32)

        B, C_in, IH, IW = input_features.shape
        # Stage 1: conv2d with w1, b1
        OH1 = IH // 2  # 80 // 2 = 40
        OW1 = IW // 2  # time_dim // 2
        x1 = torch.empty((B, w1.shape[0], OH1, OW1), dtype=torch.float32, device=input_features.device)
        # Launch Triton conv2d kernel for each output element
        grid1 = (B, w1.shape[0], OH1, OW1)
        conv2d_stage_single_kernel[grid1](
            input_features, w1, b1 if b1 is not None else torch.empty(0, device=input_features.device, dtype=torch.float32), x1,
            B, IH, IW, w1.shape[0], OH1, OW1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            has_bias=(b1 is not None),
            num_warps=1, num_stages=1,
        )
        # GELU stage 1
        x1_flat = x1.reshape(-1)  # flatten
        y1_flat = torch.empty_like(x1_flat)
        gelu_kernel_1d[(x1_flat.numel(),)](x1_flat, y1_flat, x1_flat.numel())
        x1 = y1_flat.reshape(B, w1.shape[0], OH1, OW1)

        # Stage 2: conv2d with w2, b2 on x1
        OH2 = OH1 // 2  # 40 // 2 = 20
        OW2 = OW1 // 2  # time_after_conv
        x2 = torch.empty((B, w2.shape[0], OH2, OW2), dtype=torch.float32, device=input_features.device)
        grid2 = (B, w2.shape[0], OH2, OW2)
        conv2d_stage_single_kernel[grid2](
            x1, w2, b2 if b2 is not None else torch.empty(0, device=input_features.device, dtype=torch.float32), x2,
            B, OH1, OW1, w2.shape[0], OH2, OW2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            has_bias=(b2 is not None),
            num_warps=1, num_stages=1,
        )
        # GELU stage 2
        x2_flat = x2.reshape(-1)
        y2_flat = torch.empty_like(x2_flat)
        gelu_kernel_1d[(x2_flat.numel(),)](x2_flat, y2_flat, x2_flat.numel())
        x2 = y2_flat.reshape(B, w2.shape[0], OH2, OW2)

        # Stage 3: conv2d with w3, b3 on x2
        OH3 = OH2 // 2  # 20 // 2 = 10
        OW3 = OW2 // 2  # time_after_conv
        x3 = torch.empty((B, w3.shape[0], OH3, OW3), dtype=torch.float32, device=input_features.device)
        grid3 = (B, w3.shape[0], OH3, OW3)
        conv2d_stage_single_kernel[grid3](
            x2, w3, b3 if b3 is not None else torch.empty(0, device=input_features.device, dtype=torch.float32), x3,
            B, OH2, OW2, w3.shape[0], OH3, OW3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            has_bias=(b3 is not None),
            num_warps=1, num_stages=1,
        )
        # GELU stage 3
        x3_flat = x3.reshape(-1)
        y3_flat = torch.empty_like(x3_flat)
        gelu_kernel_1d[(x3_flat.numel(),)](x3_flat, y3_flat, x3_flat.numel())
        x3 = y3_flat.reshape(B, w3.shape[0], OH3, OW3)

        # Reshape to (B, T, N) where N = 384*40 = 15360
        B_final, C, OH, OW = x3.shape
        assert C == 384, "Unexpected channels after third conv"
        N = C * OH  # 384 * 40 = 15360
        x3_flat = x3.reshape(B_final, OW, N)  # (B, T, N)

        # Final linear projection + scaling + positional embedding in Triton
        M = conv_out_weight.shape[0]  # 1024
        MAX_POS = positional_embedding.shape[0]  # 1500
        out_flat = torch.empty((B_final * OW, M), dtype=torch.float32, device=input_features.device)

        # Launch linear_proj_pos_kernel
        grid = (B_final * OW,)  # one program per (b, t)
        linear_proj_pos_kernel[grid](
            x3_flat.reshape(-1), conv_out_weight, positional_embedding, out_flat,
            B_final, OW, N, M, MAX_POS, embed_scale,
            num_warps=1, num_stages=1,
        )

        # Reshape to (B, T, M) and return
        out = out_flat.reshape(B_final, OW, M)

        # Convert back to bf16 to match original dtype behavior
        out = out.to(torch.bfloat16)

        return out


def run(*args):
    return ModelNew()(*args)
