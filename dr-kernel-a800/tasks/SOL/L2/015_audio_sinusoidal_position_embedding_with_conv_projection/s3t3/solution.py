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
    # program ids
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    # output scalar accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 kernel
    for ci in range(0, Ci):
        for kh in range(0, Kh):
            hi = 2 * ho_id - kh + 1  # stride=2, padding=1
            if hi < 0 or hi >= H:
                continue
            for kw in range(0, Kw):
                wi = 2 * wo_id - kw + 1
                if wi < 0 or wi >= W:
                    continue
                # compute input offset: ((b*b_s0 + ci*b_s1 + hi*b_s2 + wi*b_s3))
                # but we need to use strides: index = b*x_s0 + ci*x_s1 + hi*x_s2 + wi*x_s3
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off)
                x_val = x_val.to(tl.float32)

                # weight offset: co * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off)
                w_val = w_val.to(tl.float32)

                acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # store to output
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    # cast back to original dtype of x_ptr (assume y has same dtype as x)
    # We store as float32 or cast to x dtype. For safety, cast to float32 and rely on y tensor dtype to be float32.
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_kernel(
    inp_ptr, out_ptr,
    B, Co, Ho, Wo,
    inp_s0, inp_s1, inp_s2, inp_s3,
    out_s0, out_s1, out_s2, out_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    in_off = b_id * inp_s0 + co_id * inp_s1 + ho_id * inp_s2 + wo_id * inp_s3
    x = tl.load(inp_ptr + in_off).to(tl.float32)

    # GELU tanh approximation
    # constants
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))

    out_off = b_id * out_s0 + co_id * out_s1 + ho_id * out_s2 + wo_id * out_s3
    tl.store(out_ptr + out_off, gelu)


@triton.jit
def linear_proj_kernel(
    x_ptr, w_ptr, out_ptr,
    B, T, D, K,
    x_s0, x_s1, x_s2,  # x has shape (B, T, K) with strides (x_s0, x_s1, x_s2)
    w_s0, w_s1, w_s2,  # w has shape (D, K) with strides (w_s0, w_s1, w_s2)
    out_s0, out_s1, out_s2,  # out has shape (B, T, D) with strides (out_s0, out_s1, out_s2)
    embed_scale: tl.constexpr,
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    # accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over K dimension in chunks (K=3840), but simple per-element loop for clarity
    for k in range(0, K):
        x_off = b_id * x_s0 + t_id * x_s1 + k * x_s2
        x_val = tl.load(x_ptr + x_off).to(tl.float32)

        w_off = d_id * w_s0 + k * w_s1
        w_val = tl.load(w_ptr + w_off).to(tl.float32)

        acc += x_val * w_val

    # scale
    acc = acc * embed_scale

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    tl.store(out_ptr + out_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Store weights and embedding as buffers (no grad, not trained in this benchmark)
        self.register_buffer("conv2d1_weight", conv2d1_weight)
        self.register_buffer("conv2d1_bias", conv2d1_bias)
        self.register_buffer("conv2d2_weight", conv2d2_weight)
        self.register_buffer("conv2d2_bias", conv2d2_bias)
        self.register_buffer("conv2d3_weight", conv2d3_weight)
        self.register_buffer("conv2d3_bias", conv2d3_bias)
        self.register_buffer("conv_out_weight", conv_out_weight)  # (1024, 3840)
        # Store positional embedding as buffer (1500, 1024)
        self.register_buffer("positional_embedding", positional_embedding)
        self.embed_scale = float(embed_scale)

    def forward(self, input_features):
        # Ensure contiguous (host-side metadata ops only)
        x = input_features.contiguous()

        B, Ci, H, W = x.shape  # Ci=1

        # conv1: output (B, Co1, Ho1, Wo1) with Co1=384
        Co1, Ci1, Kh, Kw = self.conv2d1_weight.shape
        Ho1 = (H + 2 * 1 - Kh) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw) // 2 + 1

        # Allocate and launch conv1
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

        # GELU (tanh approximation) in Triton
        gelu_x1 = torch.empty_like(x1)
        gelu_tanh_kernel[grid1](
            x1, gelu_x1,
            B, Co1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            gelu_x1.stride(0), gelu_x1.stride(1), gelu_x1.stride(2), gelu_x1.stride(3),
            num_warps=4,
        )
        x = gelu_x1  # x is now conv1 output after GELU

        # conv2
        Co2, Ci2, Kh2, Kw2 = self.conv2d2_weight.shape
        Ho2 = (Ho1 + 2 * 1 - Kh2) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - Kw2) // 2 + 1

        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x.device, dtype=x.dtype)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x, self.conv2d2_weight, self.conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, Kh2, Kw2, Ho2, Wo2,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv2d2_weight.stride(0), self.conv2d2_weight.stride(1), self.conv2d2_weight.stride(2), self.conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            num_warps=4,
        )

        x = x2  # GELU after conv2 not required by original code; we skip it here

        # conv3
        Co3, Ci3, Kh3, Kw3 = self.conv2d3_weight.shape
        Ho3 = (Ho2 + 2 * 1 - Kh3) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - Kw3) // 2 + 1

        x3 = torch.empty((B, Co3, Ho3, Wo3), device=x.device, dtype=x.dtype)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x, self.conv2d3_weight, self.conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, Kh3, Kw3, Ho3, Wo3,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv2d3_weight.stride(0), self.conv2d3_weight.stride(1), self.conv2d3_weight.stride(2), self.conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            num_warps=4,
        )

        # Now x3 is (B, 384, 10, Tafter). We need to linearly project to d_model=1024: (B, Tafter, 3840) @ (1024, 3840) -> (B, Tafter, 1024)
        # Reshape x3 to (B, Tafter, 3840) by treating last two dims (C=384, F=10) into a single K=3840
        Bx, C, F, Tafter = x3.shape
        # We need (B, Tafter, C*F)
        # We will not use torch to compute; instead we will launch a Triton kernel that recomputes this linear mapping from x3.
        # To do that, we need weights conv_out_weight (1024, 3840). We have it stored as buffer.
        out = torch.empty((B, Tafter, 1024), device=x3.device, dtype=x3.dtype)

        # Launch linear projection kernel: grid = (B, Tafter, 1024)
        grid_lin = (B, Tafter, 1024)
        linear_proj_kernel[grid_lin](
            x3, self.conv_out_weight, out,
            B, Tafter, 1024, 3840,
            x3.stride(0), x3.stride(1), x3.stride(2),  # (B, Tafter, 3840)
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1), self.conv_out_weight.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            self.embed_scale,
            num_warps=4,
        )

        # Scale by embed_scale
        # out already scaled in kernel

        # Add positional embedding (1500, 1024). We only need first Tafter rows. Add for each batch element without torch ops.
        # We load pos_emb rows into registers per t and add to out. We can implement a small Triton kernel that loops over D=1024
        # and adds pos_emb[t, d] to each batch element. Since out is (B, Tafter, 1024), we can do broadcasting add:
        # y[b, t, d] = y[b, t, d] + pos_emb[t, d].
        # Implement kernel that adds pos_emb for each (t) to all b.

        # Prepare grid for positional add: grid = (B, Tafter, 1024)
        D = 1024
        grid_pos = (B, Tafter, D)
        # We'll do elementwise add in Triton: y = y + pos_emb[t, d] for each (b, t, d).
        # But Triton kernel cannot index per b/t/d across grid, so instead we loop over d and use broadcast indexing: y[b, t, d] += pos_emb[t, d]
        # Implement a simple per-(b, t) kernel looping over D. Better: launch grid over (B, Tafter) and loop over D inside kernel.

        @triton.jit
        def add_pos_emb_kernel(y_ptr, pos_ptr, B, T, D):
            b_id = tl.program_id(0)
            t_id = tl.program_id(1)
            # loop over D dimension
            for d in range(0, D):
                y_off = b_id * y_ptr.stride(0) + t_id * y_ptr.stride(1) + d * y_ptr.stride(2)
                y_val = tl.load(y_ptr + y_off).to(tl.float32)
                pos_val = tl.load(pos_ptr + t_id * pos_ptr.stride(0) + d * pos_ptr.stride(1)).to(tl.float32)
                tl.store(y_ptr + y_off, y_val + pos_val)

        # Launch positional add kernel
        add_pos_emb_kernel[grid_pos](
            out, self.positional_embedding,
            B, Tafter, D,
            num_warps=4,
        )

        # Return final output
        return out


def run(*args):
    return ModelNew()(*args)
