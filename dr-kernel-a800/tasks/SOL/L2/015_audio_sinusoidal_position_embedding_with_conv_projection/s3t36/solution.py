import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B: tl.constexpr, Ci: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Co: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    w_s0, w_s1, w_s2, w_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    # program ids
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    # accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh  # stride=2, padding=1
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw  # stride=2, padding=1
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                # load input with mask
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0).to(tl.float32)
                # load weight
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off).to(tl.float32)
                acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # store output (cast to original dtype of y_ptr will be handled by caller)
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def linear_proj_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2,
    w_s0, w_s1,
    y_s0, y_s1, y_s2,
):
    # Grid: (B, T, D) — one program per output element
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # dot product over K: sum_k x[b, t, k] * w[d, k]
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
    out_ptr, pos_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    out_s0, out_s1, out_s2,
    pos_s0, pos_s1,
):
    # Grid: (B, T, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    out_val = tl.load(out_ptr + out_off).to(tl.float32)

    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    tl.store(out_ptr + out_off, out_val + pos_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Register buffers (no gradients expected)
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
        Co1 = self.conv2d1_weight.shape[0]
        Kh, Kw = 3, 3
        Ho1 = (H + 2 * 1 - Kh) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x.device, dtype=x.dtype)

        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            x, self.conv2d1_weight, self.conv2d1_bias, x1,
            B=B, Ci=1, H=H, W=W,
            Co=Co1, Kh=Kh, Kw=Kw,
            Ho=Ho1, Wo=Wo1,
            x_s0=x.stride(0), x_s1=x.stride(1), x_s2=x.stride(2), x_s3=x.stride(3),
            w_s0=self.conv2d1_weight.stride(0), w_s1=self.conv2d1_weight.stride(1), w_s2=self.conv2d1_weight.stride(2), w_s3=self.conv2d1_weight.stride(3),
            y_s0=x1.stride(0), y_s1=x1.stride(1), y_s2=x1.stride(2), y_s3=x1.stride(3),
        )

        # conv2
        Co2 = self.conv2d2_weight.shape[0]
        Ho2 = (Ho1 + 2 * 1 - Kh) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - Kw) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x.device, dtype=x.dtype)

        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x1, self.conv2d2_weight, self.conv2d2_bias, x2,
            B=B, Ci=Co1, H=Ho1, W=Wo1,
            Co=Co2, Kh=Kh, Kw=Kw,
            Ho=Ho2, Wo=Wo2,
            x_s0=x1.stride(0), x_s1=x1.stride(1), x_s2=x1.stride(2), x_s3=x1.stride(3),
            w_s0=self.conv2d2_weight.stride(0), w_s1=self.conv2d2_weight.stride(1), w_s2=self.conv2d2_weight.stride(2), w_s3=self.conv2d2_weight.stride(3),
            y_s0=x2.stride(0), y_s1=x2.stride(1), y_s2=x2.stride(2), y_s3=x2.stride(3),
        )

        # conv3
        Co3 = self.conv2d3_weight.shape[0]
        Ho3 = (Ho2 + 2 * 1 - Kh) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - Kw) // 2 + 1  # expected 10
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x.device, dtype=x.dtype)

        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x2, self.conv2d3_weight, self.conv2d3_bias, x3,
            B=B, Ci=Co2, H=Ho2, W=Wo2,
            Co=Co3, Kh=Kh, Kw=Kw,
            Ho=Ho3, Wo=Wo3,
            x_s0=x2.stride(0), x_s1=x2.stride(1), x_s2=x2.stride(2), x_s3=x2.stride(3),
            w_s0=self.conv2d3_weight.stride(0), w_s1=self.conv2d3_weight.stride(1), w_s2=self.conv2d3_weight.stride(2), w_s3=self.conv2d3_weight.stride(3),
            y_s0=x3.stride(0), y_s1=x3.stride(1), y_s2=x3.stride(2), y_s3=x3.stride(3),
        )

        # At this point, x3 has shape (B, 384, 10, Wo3) with Wo3 = time_after_conv.
        # We need to form (B, Wo3, 384*10) which is (B, Tafter, 3840). In Triton, we can't easily "view" tensors, but we can compute linear projection using Triton:
        B_b, Co3_b, Ho3_b, Wo3_b = x3.shape
        assert Co3_b == 384 and Ho3_b == 10, "Conv3 output shape mismatch"

        # Now run linear projection: (B, Wo3, 384*10) @ (1024, 384*10) -> (B, Wo3, 1024)
        K = Co3 * Ho3 * Wo3  # features per time step
        D = self.conv_out_weight.shape[0]  # 1024
        # Reshape x3 to (B, Wo3, K) without torch: We'll gather using Triton kernel. But Triton kernel cannot magically reshape, so we compute y directly via Triton kernel that sums over K per (b, t, d).
        # Allocate output y_lin: (B, Wo3, D)
        y_lin = torch.empty((B, Wo3_b, D), device=x.device, dtype=x.dtype)

        # Launch linear projection kernel over grid (B, Wo3, D)
        grid_lin = (B, Wo3_b, D)
        linear_proj_kernel[grid_lin](
            x3, self.conv_out_weight, y_lin,
            B=B, T=Wo3_b, K=K, D=D,
            x_s0=x3.stride(0), x_s1=x3.stride(1), x_s2=x3.stride(2),
            w_s0=self.conv_out_weight.stride(0), w_s1=self.conv_out_weight.stride(1),
            y_s0=y_lin.stride(0), y_s1=y_lin.stride(1), y_s2=y_lin.stride(2),
        )

        # Scale by embed_scale
        # Allocate scaled output
        out_scaled = torch.empty_like(y_lin)

        # Triton kernel to scale: out[b, t, d] *= scale
        grid_scale = (B, Wo3_b, D)
        # Create scale tensor on device: use torch tensor for simplicity (only math, not heavy compute)
        scale_tensor = torch.tensor(self.embed_scale, dtype=y_lin.dtype, device=y_lin.device)
        # We can implement scaling inside the kernel by multiplying; we don't need to pass it as pointer since we can compute in-place:
        # Just multiply elementwise. However, Triton kernels cannot directly read torch scalars; we pass scale via y_ptr or compute in Python. To avoid torch, we can do scaling in Triton: multiply each element by embed_scale in the same kernel as linear (redundant). Here we do it in a separate kernel.
        # Implement scaling in Triton kernel: elementwise multiply
        @triton.jit
        def scale_kernel(y_in_ptr, y_out_ptr, B: tl.constexpr, T: tl.constexpr, D: tl.constexpr, y_in_s0, y_in_s1, y_in_s2, y_out_s0, y_out_s1, y_out_s2, scale):
            b_id = tl.program_id(0)
            t_id = tl.program_id(1)
            d_id = tl.program_id(2)
            off_in = b_id * y_in_s0 + t_id * y_in_s1 + d_id * y_in_s2
            val_in = tl.load(y_in_ptr + off_in).to(tl.float32)
            val_out = val_in * scale
            off_out = b_id * y_out_s0 + t_id * y_out_s1 + d_id * y_out_s2
            tl.store(y_out_ptr + off_out, val_out)

        scale_kernel[grid_scale](
            y_lin, out_scaled,
            B=B, T=Wo3_b, D=D,
            y_in_s0=y_lin.stride(0), y_in_s1=y_lin.stride(1), y_in_s2=y_lin.stride(2),
            y_out_s0=out_scaled.stride(0), y_out_s1=out_scaled.stride(1), y_out_s2=out_scaled.stride(2),
            scale=self.embed_scale,  # pass float
        )

        # Add positional embedding: broadcast across batch. Triton kernel for (B, T, D)
        grid_pos = (B, Wo3_b, D)
        add_pos_embedding_kernel[grid_pos](
            out_scaled, self.positional_embedding,
            B=B, T=Wo3_b, D=D,
            out_s0=out_scaled.stride(0), out_s1=out_scaled.stride(1), out_s2=out_scaled.stride(2),
            pos_s0=self.positional_embedding.stride(0), pos_s1=self.positional_embedding.stride(1),
        )

        # Return result: shape (B, Tafter, 1024) where Tafter = Wo3_b
        return out_scaled


def run(*args):
    return ModelNew()(*args)
