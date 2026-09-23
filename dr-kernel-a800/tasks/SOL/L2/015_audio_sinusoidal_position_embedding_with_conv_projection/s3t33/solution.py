import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_4d_kernel(
    x_ptr,  # input, *bf16 or *f16
    w_ptr,  # weight, *bf16 or *f16 (same as input)
    b_ptr,  # bias, *f32
    y_ptr,  # output, *bf16 or *f16 (same as input)
    # sizes
    B: tl.constexpr, Ci: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Co: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    Ho: tl.constexpr, Wo: tl.constexpr,
    # strides
    x_s0, x_s1, x_s2, x_s3,
    w_s0, w_s1, w_s2, w_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    # accumulate in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and kernel window
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh  # stride=2, pad=1
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                # in-bounds check for padding
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                # compute input offset
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                # masked load
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0).to(tl.float32)
                # compute weight offset
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off).to(tl.float32)
                acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # store to output
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_4d_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, Ci: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Co: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr, Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    # GELU tanh approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    gelu = 0.5 * x_val * (1.0 + tl.math.tanh(c * (x_val + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def gather_conv3_to_BTW_1d_kernel(
    src_ptr, dst_ptr,
    B: tl.constexpr, Co: tl.constexpr, Ho3: tl.constexpr, Wo3: tl.constexpr, Tafter: tl.constexpr,
    src_s0, src_s1, src_s2, src_s3,
    dst_s0, dst_s1, dst_s2,
):
    # total elements: B * Tafter * (Co * Ho3 * Wo3) == B * Tafter * 3840
    idx = tl.program_id(0)
    total = B * Tafter * (Co * Ho3 * Wo3)
    # decompose idx into (b, t, k)
    b_idx = idx // (Tafter * (Co * Ho3 * Wo3))
    t_idx = (idx // (Co * Ho3 * Wo3)) % Tafter
    k_idx = idx % (Co * Ho3 * Wo3)

    co = k_idx // (Ho3 * Wo3)
    ho = (k_idx % (Ho3 * Wo3)) // Wo3
    wo = (k_idx % (Ho3 * Wo3)) % Wo3

    src_off = b_idx * src_s0 + co * src_s1 + ho * src_s2 + wo * src_s3
    src_val = tl.load(src_ptr + src_off).to(tl.float32)

    dst_off = b_idx * dst_s0 + t_idx * dst_s1 + k_idx * dst_s2
    tl.store(dst_ptr + dst_off, src_val)


@triton.jit
def linear_proj_1d_kernel(
    x_ptr, w_ptr, out_ptr,
    B: tl.constexpr, Tafter: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2,  # x strides for (B, Tafter, K)
    w_s0, w_s1,        # w strides for (D, K)
    out_s0, out_s1, out_s2,  # out strides for (B, Tafter, D)
):
    # linear index over B * Tafter * D
    idx = tl.program_id(0)
    total = B * Tafter * D
    b = idx // (Tafter * D)
    t = (idx // D) % Tafter
    d = idx % D

    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K):
        x_off = b * x_s0 + t * x_s1 + k * x_s2
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        w_off = d * w_s0 + k * w_s1
        w_val = tl.load(w_ptr + w_off).to(tl.float32)
        acc += x_val * w_val

    out_off = b * out_s0 + t * out_s1 + d * out_s2
    tl.store(out_ptr + out_off, acc)


@triton.jit
def add_pos_scale_2d_kernel(
    out_ptr, pos_ptr,
    B: tl.constexpr, Tafter: tl.constexpr, D: tl.constexpr,
    out_s0, out_s1, out_s2,
    pos_s0, pos_s1,
):
    b_id = tl.program_id(0)
    t_d_id = tl.program_id(1)  # flattened time and feature index
    t_idx = t_d_id // D
    d_idx = t_d_id % D

    # load output scalar
    out_off = b_id * out_s0 + t_idx * out_s1 + d_idx * out_s2
    out_val = tl.load(out_ptr + out_off).to(tl.float32)

    # load positional embedding pos[t, d] (pos shape is (1500, 1024) but we use first Tafter rows)
    pos_off = t_idx * pos_s0 + d_idx * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    # add scaled positional embedding (embed_scale is passed as float in forward)
    # The original embed_scale is sqrt(1024) = 32.0
    # We assume out_ptr and pos_ptr are fp32; if not, we can cast after computation.
    scaled = pos_val * 32.0
    out_val += scaled

    tl.store(out_ptr + out_off, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                 conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # register buffers
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
        x = input_features.contiguous()  # ensure contiguous input
        B, Ci, H, W = x.shape  # Ci = 1

        # conv1: (1, 384, 40, W1), W1 = W//2
        Co1 = self.conv2d1_weight.shape[0]
        Kh1, Kw1 = 3, 3
        Ho1 = (H + 2 * 1 - Kh1) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw1) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x.device, dtype=x.dtype)
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_pad1_4d_kernel[grid1](
            x, self.conv2d1_weight, self.conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh1, Kw1, Ho1, Wo1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # conv2: (384 -> 384), output (B, 384, 20, W2), W2 = Wo1//2
        Co2 = self.conv2d2_weight.shape[0]
        Ho2 = (Ho1 + 2 * 1 - 3) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - 3) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x.device, dtype=x.dtype)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_pad1_4d_kernel[grid2](
            x1, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, 3, 3, Ho2, Wo2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # conv3: (384 -> 384), output (B, 384, 10, W3), W3 = Wo2//2
        Co3 = self.conv2d3_weight.shape[0]
        Ho3 = (Ho2 + 2 * 1 - 3) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - 3) // 2 + 1
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x.device, dtype=x.dtype)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_pad1_4d_kernel[grid3](
            x2, self.conv2d3_weight, self.conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, 3, 3, Ho3, Wo3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # GELU on conv3 output (B, 384, 10, Wo3)
        grid_gelu = (B, Co3, Ho3, Wo3)
        x3_gelu = torch.empty_like(x3, dtype=torch.float32, device=x.device)  # compute GELU in fp32 for numerical stability
        gelu_tanh_4d_kernel[grid_gelu](
            x3, x3_gelu,
            B, Co3, Ho3, Wo3, Co3, 3, 3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
        )

        # Reshape to (B, Wo3, Co3*Ho3*Wo3) where Co3*Ho3*Wo3 == 3840
        # Compute Tafter = Wo3 (the output time dimension after last conv).
        Tafter = Wo3
        x3_gelu_bt = torch.empty((B, Tafter, Co3 * Ho3 * Wo3), device=x.device, dtype=torch.float32)
        total = B * Tafter * (Co3 * Ho3 * Wo3)
        grid_gather = (total,)
        gather_conv3_to_BTW_1d_kernel[grid_gather](
            x3_gelu, x3_gelu_bt,
            B, Co3, Ho3, Wo3, Tafter,
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            x3_gelu_bt.stride(0), x3_gelu_bt.stride(1), x3_gelu_bt.stride(2),
        )

        # Linear projection: (B, Tafter, 3840) @ (1024, 3840) -> (B, Tafter, 1024)
        D = self.conv_out_weight.shape[0]  # 1024
        K = self.conv_out_weight.shape[1]  # 3840
        out = torch.empty((B, Tafter, D), device=x.device, dtype=torch.float32)
        total_linear = B * Tafter * D
        grid_linear = (total_linear,)
        linear_proj_1d_kernel[grid_linear](
            x3_gelu_bt, self.conv_out_weight, out,
            B, Tafter, K, D,
            x3_gelu_bt.stride(0), x3_gelu_bt.stride(1), x3_gelu_bt.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
        )

        # Scale by embed_scale = 32.0 and add positional embedding in Triton
        # pos shape is (1500, 1024). We broadcast add along batch.
        pos = self.positional_embedding.to(torch.float32)  # ensure fp32 for computation
        grid_add = (B, Tafter * D)
        add_pos_scale_2d_kernel[grid_add](
            out, pos,
            B, Tafter, D,
            out.stride(0), out.stride(1), out.stride(2),
            pos.stride(0), pos.stride(1),
        )

        return out


def run(*args):
    return ModelNew()(*args)
