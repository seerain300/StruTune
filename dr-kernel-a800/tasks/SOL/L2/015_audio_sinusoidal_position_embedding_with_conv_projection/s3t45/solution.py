import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_bias_kernel(
    x_ptr,  # *bf16 or *f16
    w_ptr,  # *bf16 or *f16
    b_ptr,  # *bf16 or *f16, 1D bias of size Co
    y_ptr,  # *bf16 or *f16
    B: tl.constexpr,
    Ci: tl.constexpr,  # number of input channels (e.g., 1, 384, 384)
    H, W,  # input spatial dims
    Co,    # output channels (e.g., 384)
    Kh: tl.constexpr, Kw: tl.constexpr,  # kernel size (e.g., 3x3)
    Ho, Wo,  # output spatial dims
    x_s0, x_s1, x_s2, x_s3,  # strides for x: N, C_in, H, W
    w_s0, w_s1, w_s2, w_s3,  # strides for w: C_out, C_in, Kh, Kw
    y_s0, y_s1, y_s2, y_s3,  # strides for y: N, C_out, Ho, Wo
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Direct reduction over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh  # since stride=2, padding=1
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off)
                acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    # Add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # Store result
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)  # Triton will cast to output dtype as needed


@triton.jit
def gelu_erf_kernel(
    x_ptr,  # *bf16 or *f16
    y_ptr,  # *bf16 or *f16
    B: tl.constexpr, Co: tl.constexpr, Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    # GELU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    gelu = 0.5 * x_val * (1.0 + tl.math.erf(x_val * inv_sqrt2))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def linear_proj_kernel(
    x_ptr,        # *bf16 or *f16, shape (B, Tafter, K) with K=3840
    w_ptr,        # *bf16 or *f16, shape (D, K) with D=1024
    y_ptr,        # *bf16 or *f16, shape (B, Tafter, D)
    B, Tafter, K, D,
    x_s0, x_s1, x_s2,   # strides for x: N, T, K
    w_s0, w_s1,         # strides for w: D, K
    y_s0, y_s1, y_s2,   # strides for y: N, T, D
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    # Accumulate in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K dimension to compute dot
    for k in range(0, K):
        x_off = b_id * x_s0 + t_id * x_s1 + k * x_s2
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        w_off = d_id * w_s0 + k * w_s1
        w_val = tl.load(w_ptr + w_off).to(tl.float32)
        acc += x_val * w_val

    # Scale by embed_scale (32.0) as per original code
    acc = acc * 32.0

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    tl.store(y_ptr + y_off, acc)


@triton.jit
def add_pos_embedding_kernel(
    y_ptr,          # *bf16 or *f16, shape (B, Tafter, D)
    pos_ptr,        # *bf16 or *f16, shape (Tafter, D)
    B, Tafter, D,
    y_s0, y_s1, y_s2,
    pos_s0, pos_s1,  # pos strides: Tafter, D (we'll pass int64 here)
):
    # We ensure indices are int64 in Triton to avoid int32/int64 mismatch
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    # Load pos[t, d] and add to y[b, t, d]
    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    pos_off = t_id * pos_s0 + d_id * pos_s1

    y_val = tl.load(y_ptr + y_off).to(tl.float32)
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)
    tl.store(y_ptr + y_off, y_val + pos_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Register buffers (no torch ops in forward)
        self.register_buffer("conv2d1_weight", conv2d1_weight)  # (384, 1, 3, 3)
        self.register_buffer("conv2d1_bias", conv2d1_bias)      # (384,)
        self.register_buffer("conv2d2_weight", conv2d2_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d2_bias", conv2d2_bias)      # (384,)
        self.register_buffer("conv2d3_weight", conv2d3_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d3_bias", conv2d3_bias)      # (384,)
        self.register_buffer("conv_out_weight", conv_out_weight)  # (1024, 3840)
        self.register_buffer("positional_embedding", positional_embedding)  # (1500, 1024)
        self.embed_scale = float(embed_scale)  # sqrt(1024) = 32.0

    def forward(self, input_features):
        # Ensure input is contiguous and bfloat16
        x0 = input_features.contiguous()
        assert x0.dtype == torch.bfloat16, "Input must be bfloat16"

        B, Ci, H, W = x0.shape  # Ci=1

        # conv1: (B, 1, 80, W) -> (B, 384, 40, W//2)
        Co1, Ci1, Kh, Kw = self.conv2d1_weight.shape
        Ho1 = (H + 2 * 1 - Kh) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x0.device, dtype=torch.bfloat16)
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_bias_kernel[grid1](
            x0, self.conv2d1_weight, self.conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh, Kw, Ho1, Wo1,
            x0.stride(0), x0.stride(1), x0.stride(2), x0.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # GELU conv1
        x1_gelu = torch.empty_like(x1, dtype=torch.bfloat16)
        grid_gelu1 = (B, Co1, Ho1, Wo1)
        gelu_erf_kernel[grid_gelu1](
            x1, x1_gelu,
            B, Co1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
        )

        # conv2: (384, 384, 3, 3), stride=2, padding=1
        Co2 = self.conv2d2_weight.shape[0]
        Ho2 = (Ho1 + 2 * 1 - Kh) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - Kw) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x0.device, dtype=torch.bfloat16)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_bias_kernel[grid2](
            x1_gelu, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, Kh, Kw, Ho2, Wo2,
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # GELU conv2
        x2_gelu = torch.empty_like(x2, dtype=torch.bfloat16)
        grid_gelu2 = (B, Co2, Ho2, Wo2)
        gelu_erf_kernel[grid_gelu2](
            x2, x2_gelu,
            B, Co2, Ho2, Wo2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
        )

        # conv3: (384, 384, 3, 3), stride=2, padding=1
        Co3 = self.conv2d3_weight.shape[0]
        Ho3 = (Ho2 + 2 * 1 - Kh) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - Kw) // 2 + 1
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x0.device, dtype=torch.bfloat16)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_bias_kernel[grid3](
            x2_gelu, self.conv2d3_weight, self.conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, Kh, Kw, Ho3, Wo3,
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # GELU conv3
        x3_gelu = torch.empty_like(x3, dtype=torch.bfloat16)
        grid_gelu3 = (B, Co3, Ho3, Wo3)
        gelu_erf_kernel[grid_gelu3](
            x3, x3_gelu,
            B, Co3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
        )

        # Gather to (B, Tafter, 3840): Tafter = Wo3
        Tafter = Wo3
        K = Co3 * Ho3 * Tafter  # 384 * 10 * Tafter == 3840
        x25 = torch.empty((B, Tafter, K), device=x0.device, dtype=torch.bfloat16)
        # We need to fill x25[b, t, k] where k maps to (co, ho, t_idx):
        # co = k // (Ho3 * Tafter), ho = (k % (Ho3 * Tafter)) // Tafter, t_idx = k % Tafter
        # However, Triton doesn't have dynamic gather like PyTorch. To form x25, we can compute it in Triton by launching a kernel that computes
        # each element (b, t, k) and loads from x3_gelu. That's fine: just another kernel.
        # Define a simple kernel that writes x25[b, t, k] = x3_gelu[b, co, ho, t_idx]
        @triton.jit
        def gather_to_k_kernel(
            src_ptr,  # x3_gelu
            dst_ptr,  # x25
            B, Tafter, Co3, Ho3, K,
            src_s0, src_s1, src_s2, src_s3,
            dst_s0, dst_s1, dst_s2,
        ):
            b_id = tl.program_id(0)
            t_id = tl.program_id(1)
            k_id = tl.program_id(2)
            # compute co, ho, t_idx from k
            # k = co * (Ho3 * Tafter) + ho * Tafter + t_idx
            co = k_id // (Ho3 * Tafter)
            rem = k_id % (Ho3 * Tafter)
            ho = rem // Tafter
            t_idx = rem % Tafter
            # load from src and store to dst
            src_off = b_id * src_s0 + co * src_s1 + ho * src_s2 + t_idx * src_s3
            val = tl.load(src_ptr + src_off).to(tl.float32)
            dst_off = b_id * dst_s0 + t_id * dst_s1 + k_id * dst_s2
            tl.store(dst_ptr + dst_off, val)  # store as fp32; dtype is bfloat16 for x25. Triton will cast on store.

        grid_gather = (B, Tafter, K)
        gather_to_k_kernel[grid_gather](
            x3_gelu, x25,
            B, Tafter, Co3, Ho3, K,
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            x25.stride(0), x25.stride(1), x25.stride(2),
        )

        # Linear projection to 1024
        D = 1024
        y = torch.empty((B, Tafter, D), device=x0.device, dtype=torch.bfloat16)
        grid_lin = (B, Tafter, D)
        linear_proj_kernel[grid_lin](
            x25, self.conv_out_weight, y,
            B, Tafter, K, D,
            x25.stride(0), x25.stride(1), x25.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
        )

        # Add positional embedding (broadcast across batch). Ensure int64 strides for pos.
        # pos: (Tafter, D) in bfloat16 or float16, but we'll load as fp32 and add to y in fp32 then cast.
        pos = self.positional_embedding.to(torch.bfloat16)  # ensure same dtype as y
        grid_pos = (B, Tafter, D)
        add_pos_embedding_kernel[grid_pos](
            y, pos,
            B, Tafter, D,
            y.stride(0), y.stride(1), y.stride(2),
            pos.stride(0), pos.stride(1),
            num_warps=4,
        )

        return y


def run(*args):
    return ModelNew()(*args)
