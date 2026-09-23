import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton Conv2d stage kernel: computes a tile of outputs for fixed (b, oc).
# x: (B, C_in, IH, IW), w: (OC, C_in, 3, 3), y: (B, OC, OH, OW), stride=2, padding=1
@triton.jit
def conv2d_stage_kernel(
    x_ptr,          # *fp32, input tensor
    w_ptr,          # *fp32, weights tensor
    b_ptr,          # *fp32, bias tensor or nullptr
    y_ptr,          # *fp32, output tensor
    B: tl.constexpr,
    C_in: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OC: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_oc, y_stride_h, y_stride_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    has_bias: tl.constexpr,
):
    # program ids: tile over (b, oc, oh_tiles, ow_tiles)
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh_tile = tl.program_id(2)
    ow_tile = tl.program_id(3)

    # Compute start indices for this tile
    oh_start = oh_tile * BLOCK_H
    ow_start = ow_tile * BLOCK_W

    # Create index vectors for the tile
    oh_offsets = oh_start + tl.arange(0, BLOCK_H)
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)

    # Valid mask for output coordinates
    oh_mask = oh_offsets < OH
    ow_mask = ow_offsets < OW
    # Make 2D mask for store
    mask_store = oh_mask[:, None] & ow_mask[None, :]

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel taps
    for ci in range(0, C_in):
        for kh in range(0, 3):
            ih = 2 * oh_offsets[:, None] + kh - 1  # shape (BLOCK_H, 1)
            valid_h = (ih >= 0) & (ih < IH)       # shape (BLOCK_H, 1)
            for kw in range(0, 3):
                iw = 2 * ow_offsets[None, :] + kw - 1  # shape (1, BLOCK_W)
                valid_w = (iw >= 0) & (iw < IW)       # shape (1, BLOCK_W)

                # Broadcast to (BLOCK_H, BLOCK_W)
                valid = valid_h & valid_w & mask_store

                # Compute input offsets
                x_offsets = b * x_stride_b + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w  # (BLOCK_H, BLOCK_W)
                x_val = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)  # (BLOCK_H, BLOCK_W)

                # Load weight scalar for (oc, ci, kh, kw)
                w_offset = oc * w_stride_oc + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_offset)  # scalar

                acc += x_val * w_val

    # Add bias if present
    if has_bias:
        b_val = tl.load(b_ptr + oc).to(tl.float32)
        acc += b_val  # broadcast over tile

    # Store results to y
    y_offsets = b * y_stride_b + oc * y_stride_oc + oh_offsets[:, None] * y_stride_h + ow_offsets[None, :] * y_stride_w
    tl.store(y_ptr + y_offsets, acc, mask=mask_store)


# Triton GELU (tanh approximation) over 1D flattened tensor
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
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    x3 = x * x * x
    u = sqrt_2_over_pi * (x + c * x3)
    y = 0.5 * x * (1.0 + tl.tanh(u))
    tl.store(out_ptr + idx, y)


# Triton Linear projection + scale + positional embedding add
# X: (B, T, N), W: (M, N), pos: (M, d_model), output Y: (B, T, M)
@triton.jit
def linear_scale_pos_kernel(
    x_ptr,          # *fp32, input (B, T, N) flattened
    w_ptr,          # *fp32, weights (M, N)
    pos_ptr,        # *fp32, positional embedding (M, d_model)
    y_ptr,          # *fp32, output (B*T, M)
    B, T, N, M,     # runtime ints
    embed_scale: tl.constexpr,  # float32
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B * T:
        return
    # Each program handles one (b, t) row
    b = pid // T
    t = pid % T

    # Loop over M in tiles
    for m0 in range(0, M, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
        mask_m = m_offsets < M

        # Accumulator for this (b, t)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Reduce over N
        for n0 in range(0, N, 128):  # N is 15360; tile of 128
            n_offsets = n0 + tl.arange(0, 128)  # (128,)
            mask_n = n_offsets < N
            # Load X[b, t, n_offsets]: X is contiguous in (B, T, N)
            x_base = (b * T + t) * N
            x_off = x_base + n_offsets
            x_vals = tl.load(x_ptr + x_off, mask=mask_n, other=0.0).to(tl.float32)  # (128,)
            # Load W[m_offsets, n_offsets]: W shape (M, N), row-major
            w_off = m_offsets[:, None] * N + n_offsets[None, :]  # (BLOCK_M, 128)
            w_vals = tl.load(w_ptr + w_off, mask=mask_m[:, None], other=0.0).to(tl.float32)  # (BLOCK_M, 128)
            # Accumulate: acc[m] += sum(w[m, n] * x[n])
            # Multiply (BLOCK_M, 128) with (128,) -> (BLOCK_M,)
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # Scale
        acc *= embed_scale

        # Add positional embedding row pos[m, :] for this (b, t). pos indexed by (m,)
        # Note: t is the time index in the output; pos is indexed by m.
        pos_off = m_offsets  # (BLOCK_M,)
        pos_vals = tl.load(pos_ptr + pos_off, mask=mask_m, other=0.0).to(tl.float32)  # (BLOCK_M,)
        acc += pos_vals

        # Store to output y at row (b*T + t)
        y_row_base = (b * T + t) * M
        y_off = y_row_base + m_offsets
        tl.store(y_ptr + y_off, acc, mask=mask_m)


# Helper to compute OH and OW given IH, IW, kernel, stride, padding
def out_hw(IH, IW, KH=3, KW=3, STRIDE=2, PAD=1):
    OH = (IH + 2 * PAD - KH) // STRIDE + 1
    OW = (IW + 2 * PAD - KW) // STRIDE + 1
    return OH, OW


def cdiv(x, y):
    return (x + y - 1) // y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect arguments as in the original run signature:
        # input_features: (B, 1, 80, T)
        # conv2d1_weight: (384, 1, 3, 3)
        # conv2d1_bias: (384,)
        # conv2d2_weight: (384, 384, 3, 3)
        # conv2d2_bias: (384,)
        # conv2d3_weight: (384, 384, 3, 3)
        # conv2d3_bias: (384,)
        # conv_out_weight: (1024, 384*40) = (1024, 15360)
        # positional_embedding: (1500, 1024)
        # embed_scale: float

        # For Triton kernels, we keep compute in fp32 for robustness.
        # Args are received in the same order as run(*args) from the original code.

        # Stage 1: Conv2d (1 -> 384 channels), stride=2, padding=1
        x = args[0]  # (B, 1, 80, T)
        w1 = args[1]  # (384, 1, 3, 3)
        b1 = args[2]  # (384,)
        y1 = torch.empty((x.shape[0], 384, 20, x.shape[3] // 2), dtype=torch.float32, device=x.device)

        B = x.shape[0]
        C_in1 = w1.shape[1]
        IH = x.shape[2]
        IW = x.shape[3]
        OC1 = w1.shape[0]
        OH = out_hw(IH, IW, KH=3, KW=3, STRIDE=2, PAD=1)[0]
        OW = out_hw(IW, 3, STRIDE=2, PAD=1)[1]  # but we use x.shape[3] // 2 which is correct

        # Strides (contiguous assumed)
        x_stride_b = x.stride(0)
        x_stride_c = x.stride(1)
        x_stride_h = x.stride(2)
        x_stride_w = x.stride(3)

        w_stride_oc = w1.stride(0)
        w_stride_ci = w1.stride(1)
        w_stride_kh = w1.stride(2)
        w_stride_kw = w1.stride(3)

        y_stride_b = y1.stride(0)
        y_stride_oc = y1.stride(1)
        y_stride_h = y1.stride(2)
        y_stride_w = y1.stride(3)

        # Launch conv kernel over tiles
        BLOCK_H = 8
        BLOCK_W = 8
        grid = (B, OC1, cdiv(OH, BLOCK_H), cdiv(OW, BLOCK_W))
        conv2d_stage_kernel[grid](
            x_ptr=x,
            w_ptr=w1,
            b_ptr=b1,
            y_ptr=y1,
            B=B, C_in=C_in1, IH=IH, IW=IW, OC=OC1, OH=OH, OW=OW,
            x_stride_b=x_stride_b, x_stride_c=x_stride_c, x_stride_h=x_stride_h, x_stride_w=x_stride_w,
            w_stride_oc=w_stride_oc, w_stride_ci=w_stride_ci, w_stride_kh=w_stride_kh, w_stride_kw=w_stride_kw,
            y_stride_b=y_stride_b, y_stride_oc=y_stride_oc, y_stride_h=y_stride_h, y_stride_w=y_stride_w,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            has_bias=True,
        )

        # GELU Stage 1
        y1_flat = y1.reshape(-1)
        y1_out = torch.empty_like(y1_flat, dtype=torch.float32, device=y1.device)
        n_elements = y1_flat.numel()
        gelu_kernel_1d[(n_elements,)](y1_flat, y1_out, n_elements)
        y1 = y1_out.reshape(y1.shape)

        # Stage 2: Conv2d (384 -> 384), stride=2, padding=1
        x2 = y1  # (B, 384, 20, T//2)
        w2 = args[3]  # (384, 384, 3, 3)
        b2 = args[4]  # (384,)
        y2 = torch.empty((x2.shape[0], 384, 10, x2.shape[3]), dtype=torch.float32, device=x2.device)

        C_in2 = w2.shape[1]
        OC2 = w2.shape[0]
        IH2 = x2.shape[2]
        IW2 = x2.shape[3]
        OH2 = out_hw(IH2, IW2, KH=3, KW=3, STRIDE=2, PAD=1)[0]
        OW2 = x2.shape[3]

        x2_stride_b = x2.stride(0)
        x2_stride_c = x2.stride(1)
        x2_stride_h = x2.stride(2)
        x2_stride_w = x2.stride(3)

        w2_stride_oc = w2.stride(0)
        w2_stride_ci = w2.stride(1)
        w2_stride_kh = w2.stride(2)
        w2_stride_kw = w2.stride(3)

        y2_stride_b = y2.stride(0)
        y2_stride_oc = y2.stride(1)
        y2_stride_h = y2.stride(2)
        y2_stride_w = y2.stride(3)

        BLOCK_H2 = 4
        BLOCK_W2 = 4
        grid2 = (B, OC2, cdiv(OH2, BLOCK_H2), cdiv(OW2, BLOCK_W2))
        conv2d_stage_kernel[grid2](
            x_ptr=x2,
            w_ptr=w2,
            b_ptr=b2,
            y_ptr=y2,
            B=B, C_in=C_in2, IH=IH2, IW=IW2, OC=OC2, OH=OH2, OW=OW2,
            x_stride_b=x2_stride_b, x_stride_c=x2_stride_c, x_stride_h=x2_stride_h, x_stride_w=x2_stride_w,
            w_stride_oc=w2_stride_oc, w_stride_ci=w2_stride_ci, w_stride_kh=w2_stride_kh, w_stride_kw=w2_stride_kw,
            y_stride_b=y2_stride_b, y_stride_oc=y2_stride_oc, y_stride_h=y2_stride_h, y_stride_w=y2_stride_w,
            BLOCK_H=BLOCK_H2, BLOCK_W=BLOCK_W2,
            has_bias=True,
        )

        # GELU Stage 2
        y2_flat = y2.reshape(-1)
        y2_out = torch.empty_like(y2_flat, dtype=torch.float32, device=y2.device)
        n_elements2 = y2_flat.numel()
        gelu_kernel_1d[(n_elements2,)](y2_flat, y2_out, n_elements2)
        y2 = y2_out.reshape(y2.shape)

        # Stage 3: Conv2d (384 -> 384), stride=2, padding=1
        x3 = y2  # (B, 384, 10, T//4)
        w3 = args[5]  # (384, 384, 3, 3)
        b3 = args[6]  # (384,)
        y3 = torch.empty((x3.shape[0], 384, 5, x3.shape[3]), dtype=torch.float32, device=x3.device)

        C_in3 = w3.shape[1]
        OC3 = w3.shape[0]
        IH3 = x3.shape[2]
        IW3 = x3.shape[3]
        OH3 = out_hw(IH3, IW3, KH=3, KW=3, STRIDE=2, PAD=1)[0]
        OW3 = x3.shape[3]  # equals (T//8)

        x3_stride_b = x3.stride(0)
        x3_stride_c = x3.stride(1)
        x3_stride_h = x3.stride(2)
        x3_stride_w = x3.stride(3)

        w3_stride_oc = w3.stride(0)
        w3_stride_ci = w3.stride(1)
        w3_stride_kh = w3.stride(2)
        w3_stride_kw = w3.stride(3)

        y3_stride_b = y3.stride(0)
        y3_stride_oc = y3.stride(1)
        y3_stride_h = y3.stride(2)
        y3_stride_w = y3.stride(3)

        BLOCK_H3 = 2
        BLOCK_W3 = 2
        grid3 = (B, OC3, cdiv(OH3, BLOCK_H3), cdiv(OW3, BLOCK_W3))
        conv2d_stage_kernel[grid3](
            x_ptr=x3,
            w_ptr=w3,
            b_ptr=b3,
            y_ptr=y3,
            B=B, C_in=C_in3, IH=IH3, IW=IW3, OC=OC3, OH=OH3, OW=OW3,
            x_stride_b=x3_stride_b, x_stride_c=x3_stride_c, x_stride_h=x3_stride_h, x_stride_w=x3_stride_w,
            w_stride_oc=w3_stride_oc, w_stride_ci=w3_stride_ci, w_stride_kh=w3_stride_kh, w_stride_kw=w3_stride_kw,
            y_stride_b=y3_stride_b, y_stride_oc=y3_stride_oc, y_stride_h=y3_stride_h, y_stride_w=y3_stride_w,
            BLOCK_H=BLOCK_H3, BLOCK_W=BLOCK_W3,
            has_bias=True,
        )

        # GELU Stage 3
        y3_flat = y3.reshape(-1)
        y3_out = torch.empty_like(y3_flat, dtype=torch.float32, device=y3.device)
        n_elements3 = y3_flat.numel()
        gelu_kernel_1d[(n_elements3,)](y3_flat, y3_out, n_elements3)
        y3 = y3_out.reshape(y3.shape)

        # Final: permute (B, C, F, T) -> (B, T, C*F), linear to d_model=1024, scale, add pos
        Bf, Cf, Ff, Tf = y3.shape
        C = Cf
        F = Ff
        T = Tf
        N = C * F  # 384 * 5

        # Reshape X to (B, T, N) and launch linear kernel
        x_flat = y3.reshape(B, T, N).contiguous()  # (B, T, 1920) float32

        # Load conv_out_weight (1024, 1920)
        w_proj = args[7]  # (1024, 1920)
        pos_emb = args[8]  # (1500, 1024)
        embed_scale = args[9]  # float

        # Ensure w_proj, pos_emb are float32
        w_proj = w_proj.to(torch.float32)
        pos_emb = pos_emb.to(torch.float32)

        # Allocate output (B*T, 1024) in fp32
        y_proj = torch.empty((B * T, 1024), dtype=torch.float32, device=x_flat.device)

        # Choose BLOCK_M for tiles
        BLOCK_M = 128
        # Launch kernel over B*T rows
        grid4 = (B * T,)
        linear_scale_pos_kernel[grid4](
            x_ptr=x_flat.reshape(-1),           # (B*T*N,)
            w_ptr=w_proj.reshape(-1),           # (M*N,) but we use shape (M, N) by reading correctly inside
            pos_ptr=pos_emb.reshape(-1),        # (M*d_model,)
            y_ptr=y_proj,                       # (B*T, M)
            B=B, T=T, N=N, M=1024,
            embed_scale=embed_scale,
            BLOCK_M=BLOCK_M,
        )

        # Reshape back to (B, T, 1024)
        y_final = y_proj.reshape(B, T, 1024)

        # Return as tensor; forward must return a tensor (here we can return y_final)
        return y_final


def run(*args):
    return ModelNew()(*args)
