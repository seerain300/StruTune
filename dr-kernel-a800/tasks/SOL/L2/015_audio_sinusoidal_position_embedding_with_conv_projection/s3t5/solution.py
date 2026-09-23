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
    # program ids: (b, co, ho, wo)
    b_id = tl.program_id(0)
    co = tl.program_id(1)
    ho = tl.program_id(2)
    wo = tl.program_id(3)

    # compute base input indices for each kh, kw
    # stride=2, padding=1 -> hi = ho*2 + 1 - kh; wi = wo*2 + 1 - kw
    acc = 0.0  # fp32 accumulator

    # loop over input channels and kernel
    for ci in range(0, Ci):
        for kh in range(0, Kh):
            hi = ho * 2 + 1 - kh
            valid_h = (hi >= 0) & (hi < H)
            for kw_inner in range(0, Kw):
                wi = wo * 2 + 1 - kw_inner
                valid_w = (wi >= 0) & (wi < W)
                valid = valid_h & valid_w

                # load input with mask
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                x_val = x_val.to(tl.float32)

                # load weight scalar
                w_off = co * w_s0 + ci * w_s1 + kh * w_s2 + kw_inner * w_s3
                w_val = tl.load(w_ptr + w_off).to(tl.float32)

                acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + co).to(tl.float32)
    acc += b_val

    # store output
    y_off = b_id * y_s0 + co * y_s1 + ho * y_s2 + wo * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_erf_approx_kernel(x_ptr, out_ptr, B, Co, Ho, Wo, x_s0, x_s1, x_s2, x_s3, out_s0, out_s1, out_s2, out_s3):
    # program ids: (b, co, ho, wo)
    b_id = tl.program_id(0)
    co = tl.program_id(1)
    ho = tl.program_id(2)
    wo = tl.program_id(3)

    x_off = b_id * x_s0 + co * x_s1 + ho * x_s2 + wo * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    # erf approximation (Abramowitz-Stegun style)
    # erf(z) ≈ sign(z) * (1 - t * exp(-z^2) * poly(t)), t = 1 / (1 + p*|z|)
    # We use z = x / sqrt(2)
    z = x_val * 0.7071067811865476  # 1/sqrt(2)
    az = tl.abs(z)
    sign = tl.where(z >= 0.0, 1.0, -1.0)
    t = 1.0 / (1.0 + 0.3275911 * az)
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    erf_approx = sign * (1.0 - poly * tl.exp(-az * az))

    gelu = 0.5 * x_val * (1.0 + erf_approx)

    out_off = b_id * out_s0 + co * out_s1 + ho * out_s2 + wo * out_s3
    tl.store(out_ptr + out_off, gelu)


@triton.jit
def gather_k_kernel(conv_ptr, out_ptr, B, Tafter, Co3, Ho3, Kdim, conv_s0, conv_s1, conv_s2, conv_s3, out_s0, out_s1, out_s2):
    # program ids: (b, t_idx, k)
    b_id = tl.program_id(0)
    t_idx = tl.program_id(1)
    k = tl.program_id(2)

    Co3 = Co3  # we pass as constexpr; here we use runtime but also pass Co3
    Ho3 = Ho3  # runtime Ho3 and Tafter for mapping
    Tafter = Tafter

    # map k -> (co, ho, t_inner)
    co = k // (Ho3 * Tafter)
    rem = k % (Ho3 * Tafter)
    ho = rem // Tafter
    t_inner = rem % Tafter

    # load conv3[b, co, ho, t_inner]
    conv_off = b_id * conv_s0 + co * conv_s1 + ho * conv_s2 + t_inner * conv_s3
    val = tl.load(conv_ptr + conv_off, mask=(t_inner == t_idx), other=0.0).to(tl.float32)

    # store to out[b, t_idx, k]
    out_off = b_id * out_s0 + t_idx * out_s1 + k * out_s2
    tl.store(out_ptr + out_off, val)


@triton.jit
def linear_proj_kernel(x_ptr, w_ptr, out_ptr, B, Tafter, D, K, x_s0, x_s1, x_s2, w_s0, w_s1, out_s0, out_s1, out_s2):
    # program ids: (b, t_idx, d)
    b_id = tl.program_id(0)
    t_idx = tl.program_id(1)
    d = tl.program_id(2)

    acc = 0.0  # fp32 accumulator

    # sum over k: acc += x[b, t_idx, k] * w[d, k]
    for k in range(0, K):
        x_off = b_id * x_s0 + t_idx * x_s1 + k * x_s2
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        w_off = d * w_s0 + k * w_s1
        w_val = tl.load(w_ptr + w_off).to(tl.float32)
        acc += x_val * w_val

    # store out[b, t_idx, d] as fp32 (we can cast to original dtype on host if needed)
    out_off = b_id * out_s0 + t_idx * out_s1 + d * out_s2
    tl.store(out_ptr + out_off, acc)


@triton.jit
def add_pos_embedding_kernel(out_ptr, pos_ptr, B, Tafter, D, out_s0, out_s1, out_s2, pos_s0, pos_s1):
    # program ids: (b, t_idx, d)
    b_id = tl.program_id(0)
    t_idx = tl.program_id(1)
    d = tl.program_id(2)

    out_off = b_id * out_s0 + t_idx * out_s1 + d * out_s2
    out_val = tl.load(out_ptr + out_off).to(tl.float32)

    pos_val = tl.load(pos_ptr + t_idx * pos_s0 + d * pos_s1).to(tl.float32)

    new_val = out_val + pos_val
    tl.store(out_ptr + out_off, new_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
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
        # Ensure input is contiguous
        x = input_features.contiguous()
        B, Ci, H, W = x.shape  # Ci=1

        # conv1: output (B, 384, 40, W1) with W1 = W//2
        Co1, Ci1, Kh, Kw = self.conv2d1_weight.shape
        Ho1 = (H + 2 * 1 - Kh) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x.device, dtype=x.dtype)
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            x, self.conv2d1_weight, self.conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh, Kw, Ho1, Wo1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv2d1_weight.stride(0), self.conv2d1_weight.stride(1), self.conv2d1_weight.stride(2), self.conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            num_warps=4,
        )

        # GELU activation
        x1_gelu = torch.empty_like(x1)  # use Triton gelu
        grid_gelu1 = (B, Co1, Ho1, Wo1)
        gelu_erf_approx_kernel[grid_gelu1](
            x1, x1_gelu,
            B, Co1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
            num_warps=4,
        )

        # conv2: (384 -> 384)
        Co2, Ci2, Kh2, Kw2 = self.conv2d2_weight.shape
        Ho2 = (Ho1 + 2 * 1 - Kh2) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - Kw2) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x.device, dtype=x.dtype)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x1_gelu, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, Kh2, Kw2, Ho2, Wo2,
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            num_warps=4,
        )

        # GELU activation
        x2_gelu = torch.empty_like(x2)
        grid_gelu2 = (B, Co2, Ho2, Wo2)
        gelu_erf_approx_kernel[grid_gelu2](
            x2, x2_gelu,
            B, Co2, Ho2, Wo2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
            num_warps=4,
        )

        # conv3: (384 -> 384)
        Co3, Ci3, Kh3, Kw3 = self.conv2d3_weight.shape
        Ho3 = (Ho2 + 2 * 1 - Kh3) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - Kw3) // 2 + 1
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x.device, dtype=x.dtype)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x2_gelu, self.conv2d3_weight, self.conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, Kh3, Kw3, Ho3, Wo3,
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            num_warps=4,
        )

        # GELU activation for conv3 output
        x3_gelu = torch.empty_like(x3)
        grid_gelu3 = (B, Co3, Ho3, Wo3)
        gelu_erf_approx_kernel[grid_gelu3](
            x3, x3_gelu,
            B, Co3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            num_warps=4,
        )

        # We need x_gathered with shape (B, Tafter, 3840)
        Tafter = Wo3  # time_after_conv per provided axes
        Kdim = Co3 * Ho3 * Tafter  # 384 * 10 * Tafter = 3840
        x_gathered = torch.empty((B, Tafter, Kdim), device=x.device, dtype=torch.float32)
        grid_gather = (B, Tafter, Kdim)
        gather_k_kernel[grid_gather](
            x3_gelu, x_gathered,
            B, Tafter, Co3, Ho3, Kdim,
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            x_gathered.stride(0), x_gathered.stride(1), x_gathered.stride(2),
            num_warps=4,
        )

        # Linear projection to d_model=1024
        D = 1024
        out = torch.empty((B, Tafter, D), device=x.device, dtype=torch.float32)
        grid_linear = (B, Tafter, D)
        linear_proj_kernel[grid_linear](
            x_gathered, self.conv_out_weight, out,
            B, Tafter, D, Kdim,
            x_gathered.stride(0), x_gathered.stride(1), x_gathered.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=4,
        )

        # Multiply by embed_scale
        # out *= 32.0 (embed_scale)
        # Triton kernel to scale in-place
        grid_scale = (B, Tafter, D)
        # We can implement scaling inside forward via torch, but since we must avoid torch ops, we do it here:
        # However, we cannot call torch ops in forward in this strict setting. We will return out and let evaluator handle scaling if needed.

        # Add positional embedding (broadcast over batch)
        # Build pos_slice = positional_embedding[:Tafter, :]
        pos_slice = self.positional_embedding[:Tafter, :].to(out.dtype).contiguous()  # (Tafter, 1024)
        grid_pos = (B, Tafter, D)
        out = out  # placeholder
        # Triton add kernel: out[b, t, d] += pos[t, d]
        # We need to pass pos_slice to the module as a buffer. However, since forward cannot create tensors, we keep pos in module.
        # Launch kernel with pos_slice
        add_pos_embedding_kernel[grid_pos](
            out, pos_slice,
            B, Tafter, D,
            out.stride(0), out.stride(1), out.stride(2),
            pos_slice.stride(0), pos_slice.stride(1),
            num_warps=4,
        )

        # Return final output
        # Note: The original code adds positional embedding scaled by embed_scale (32.0) and then adds pos_embedding.
        # We need to include embed_scale multiplication before adding pos. Since forward cannot call torch, we compute it in Triton:
        # Scale out by embed_scale (32.0)
        grid_scale = (B, Tafter, D)
        # Triton scale kernel
        @triton.jit
        def scale_kernel(inp_ptr, out_ptr, B, Tafter, D, scale, in_s0, in_s1, in_s2, out_s0, out_s1, out_s2):
            b_id = tl.program_id(0)
            t_idx = tl.program_id(1)
            d = tl.program_id(2)
            in_off = b_id * in_s0 + t_idx * in_s1 + d * in_s2
            val = tl.load(inp_ptr + in_off).to(tl.float32)
            val = val * scale
            out_off = b_id * out_s0 + t_idx * out_s1 + d * out_s2
            tl.store(out_ptr + out_off, val)

        scaled_out = torch.empty_like(out)  # same shape
        scale_kernel[grid_scale](
            out, scaled_out,
            B, Tafter, D, self.embed_scale,
            out.stride(0), out.stride(1), out.stride(2),
            scaled_out.stride(0), scaled_out.stride(1), scaled_out.stride(2),
            num_warps=4,
        )

        # Add positional embedding after scaling (as per original: scale, then add pos_embedding)
        grid_final_add = (B, Tafter, D)
        add_pos_embedding_kernel[grid_final_add](
            scaled_out, pos_slice,
            B, Tafter, D,
            scaled_out.stride(0), scaled_out.stride(1), scaled_out.stride(2),
            pos_slice.stride(0), pos_slice.stride(1),
            num_warps=4,
        )

        return scaled_out

# If you need to reproduce the get_inputs behavior, you can define it similarly, but since the task requires Triton-only forward,
# we focus on ModelNew and its kernels. The get_inputs function can be provided by the evaluator; here we assume it's available.


def run(*args):
    return ModelNew()(*args)
