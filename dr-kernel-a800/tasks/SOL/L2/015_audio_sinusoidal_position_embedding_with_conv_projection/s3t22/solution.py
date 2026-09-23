import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, Ci, H, W, Co, Kh, Kw, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,
    w_s0, w_s1, w_s2, w_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # One program per output element (b, co, ho, wo)
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Sum over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh  # stride=2, padding=1
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0).to(tl.float32)
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off).to(tl.float32)
                acc += x_val * w_val

    # Add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # Store to output (cast to original dtype if needed)
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_kernel(
    x_ptr, y_ptr,
    B, Co, Ho, Wo,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # Apply GELU tanh approximation: y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    gelu = 0.5 * x_val * (1.0 + tl.math.tanh(c * (x_val + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def gather_to_long_3d_kernel(
    x_ptr,  # GELU output (B, Co, Ho, Wo) in fp32
    y_ptr,  # output (B, Tafter, Co*Ho*Wo) in fp32
    B, Co, Ho, Wo, Tafter,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2,
):
    # Enumerate k across [0, Co*Ho*Wo)
    # Each program handles one k and writes to all (b, t_idx) rows
    # We can restructure: use grid (B, Tafter, Co*Ho*Wo), loop wo for k? However Triton requires static loop length, better to make k part of grid via meta-programming.
    # Simpler approach: we have grid (B, Tafter, D), where D=Co*Ho*Wo. Then we can compute (co, ho, wo) from d via integer division/mod.
    # Here, Triton doesn't support returning multiple indices; instead, we launch with grid (B, Tafter, D) and compute co,ho,wo inside.

    # NOTE: Triton requires static loop bounds; to keep simple, we instead use a 1D launch over D and iterate across B*Tafter. But Triton needs 3D grid. We'll do:
    # Launch grid = (B, Tafter, D). For each (b,t,d), compute (co,ho,wo) from d and write y[b, t, d] = x[b, co, ho, wo]
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    # Map d_id -> (co, ho, wo)
    co = d_id // (Ho * Wo)
    rem = d_id % (Ho * Wo)
    ho = rem // Wo
    wo = rem % Wo

    x_off = b_id * x_s0 + co * x_s1 + ho * x_s2 + wo * x_s3
    val = tl.load(x_ptr + x_off).to(tl.float32)

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    tl.store(y_ptr + y_off, val)


@triton.jit
def linear_proj_kernel(
    x_ptr, w_ptr, y_ptr,
    B, T, D, K,
    x_s0, x_s1, x_s2,  # (B, T, K) float32
    w_s0, w_s1,        # (D, K) float32
    y_s0, y_s1, y_s2,  # (B, T, D) float32
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K):
        x_off = b_id * x_s0 + t_id * x_s1 + k * x_s2
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        w_off = d_id * w_s0 + k * w_s1
        w_val = tl.load(w_ptr + w_off).to(tl.float32)
        acc += x_val * w_val

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    tl.store(y_ptr + y_off, acc)


@triton.jit
def add_pos_embedding_kernel(
    y_ptr, pos_ptr, scale, B, T, D,
    y_s0, y_s1, y_s2,
    pos_s0, pos_s1,
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    y_val = tl.load(y_ptr + y_off).to(tl.float32)

    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)
    y_val += pos_val * scale

    tl.store(y_ptr + y_off, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Store weights as buffers (no torch computation in forward)
        self.register_buffer("conv2d1_weight", conv2d1_weight)  # (384, 1, 3, 3)
        self.register_buffer("conv2d1_bias", conv2d1_bias)      # (384,)
        self.register_buffer("conv2d2_weight", conv2d2_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d2_bias", conv2d2_bias)      # (384,)
        self.register_buffer("conv2d3_weight", conv2d3_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d3_bias", conv2d3_bias)      # (384,)
        self.register_buffer("conv_out_weight", conv_out_weight)  # (1024, 3840)
        self.register_buffer("positional_embedding", positional_embedding)  # (1500, 1024)
        self.embed_scale = float(embed_scale)  # 32.0

    def forward(self, input_features):
        # Ensure input is contiguous
        x0 = input_features.contiguous()
        B, Ci, H, W = x0.shape  # Ci=1
        T0 = W

        # Prepare int64 dims for Triton
        B64 = int(B)
        Ci64 = int(Ci)
        H64 = int(H)
        W64 = int(W)
        Co1 = int(self.conv2d1_weight.shape[0])  # 384
        Co2 = int(self.conv2d2_weight.shape[0])  # 384
        Co3 = int(self.conv2d3_weight.shape[0])  # 384
        D_out = int(self.conv_out_weight.shape[0])  # 1024

        # Conv1: (1 -> 384), stride=2, padding=1
        Ho1 = (H64 + 2*1 - 3) // 2 + 1  # K=3
        Wo1 = (W64 + 2*1 - 3) // 2 + 1
        x1 = torch.empty((B64, Co1, Ho1, Wo1), device=x0.device, dtype=x0.dtype)
        grid1 = (B64, Co1, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            x0, self.conv2d1_weight, self.conv2d1_bias, x1,
            B64, Ci64, H64, W64, Co1, 3, 3, Ho1, Wo1,
            x0.stride(0), x0.stride(1), x0.stride(2), x0.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # GELU on conv1 output
        x1_g = torch.empty_like(x1, dtype=torch.float32, device=x0.device)
        grid2 = (B64, Co1, Ho1, Wo1)
        gelu_tanh_kernel[grid2](
            x1, x1_g,
            B64, Co1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x1_g.stride(0), x1_g.stride(1), x1_g.stride(2), x1_g.stride(3),
        )

        # Conv2: (384 -> 384), stride=2, padding=1
        Ho2 = (Ho1 + 2*1 - 3) // 2 + 1
        Wo2 = (Wo1 + 2*1 - 3) // 2 + 1
        x2 = torch.empty((B64, Co2, Ho2, Wo2), device=x0.device, dtype=x0.dtype)
        grid3 = (B64, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid3](
            x1, self.conv2d2_weight, self.conv2d2_bias, x2,
            B64, Co1, Ho1, Wo1, Co2, 3, 3, Ho2, Wo2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # GELU on conv2 output
        x2_g = torch.empty((B64, Co2, Ho2, Wo2), device=x0.device, dtype=torch.float32)
        grid4 = (B64, Co2, Ho2, Wo2)
        gelu_tanh_kernel[grid4](
            x2, x2_g,
            B64, Co2, Ho2, Wo2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            x2_g.stride(0), x2_g.stride(1), x2_g.stride(2), x2_g.stride(3),
        )

        # Conv3: (384 -> 384), stride=2, padding=1
        Ho3 = (Ho2 + 2*1 - 3) // 2 + 1
        Wo3 = (Wo2 + 2*1 - 3) // 2 + 1
        Tafter = Wo3  # time dimension after conv3
        x3 = torch.empty((B64, Co3, Ho3, Tafter), device=x0.device, dtype=x0.dtype)
        grid5 = (B64, Co3, Ho3, Tafter)
        conv2d_stride2_kernel[grid5](
            x2, self.conv2d3_weight, self.conv2d3_bias, x3,
            B64, Co2, Ho2, Wo2, Co3, 3, 3, Ho3, Tafter,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # GELU on conv3 output
        x3_g = torch.empty((B64, Co3, Ho3, Tafter), device=x0.device, dtype=torch.float32)
        grid6 = (B64, Co3, Ho3, Tafter)
        gelu_tanh_kernel[grid6](
            x3, x3_g,
            B64, Co3, Ho3, Tafter,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_g.stride(0), x3_g.stride(1), x3_g.stride(2), x3_g.stride(3),
        )

        # Gather to form (B, Tafter, 3840): k in [0, Co3*Ho3*Tafter)
        D_long = Co3 * Ho3 * Tafter
        out_long = torch.empty((B64, Tafter, D_long), device=x0.device, dtype=torch.float32)
        grid_g = (B64, Tafter, D_long)
        gather_to_long_3d_kernel[grid_g](
            x3_g, out_long,
            B64, Co3, Ho3, Tafter, Tafter,
            x3_g.stride(0), x3_g.stride(1), x3_g.stride(2), x3_g.stride(3),
            out_long.stride(0), out_long.stride(1), out_long.stride(2),
        )

        # Linear projection to (B, Tafter, 1024)
        y = torch.empty((B64, Tafter, D_out), device=x0.device, dtype=torch.float32)
        K = int(self.conv_out_weight.shape[1])  # 3840
        grid_lp = (B64, Tafter, D_out)
        linear_proj_kernel[grid_lp](
            out_long, self.conv_out_weight, y,
            B64, Tafter, D_out, K,
            out_long.stride(0), out_long.stride(1), out_long.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
        )

        # Scale by embed_scale
        y_scaled = y * self.embed_scale  # 32.0

        # Add positional embedding (1500, 1024)
        # We broadcast add across batch. Only first Tafter rows are used.
        pos = self.positional_embedding.to(torch.float32)  # ensure fp32 for math
        grid_pe = (B64, Tafter, D_out)
        add_pos_embedding_kernel[grid_pe](
            y_scaled, pos, self.embed_scale, B64, Tafter, D_out,
            y_scaled.stride(0), y_scaled.stride(1), y_scaled.stride(2),
            pos.stride(0), pos.stride(1),
        )

        # Return result (already scaled and with pos embedding added)
        return y_scaled


def run(*args):
    return ModelNew()(*args)
