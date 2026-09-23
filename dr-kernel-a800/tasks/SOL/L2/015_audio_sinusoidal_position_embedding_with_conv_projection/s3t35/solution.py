import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr,      # *f16 or *bf16
    w_ptr,      # *f16 or *bf16
    b_ptr,      # *f16 or *bf16 (bias per out channel)
    y_ptr,      # *f16 or *bf16
    B: tl.constexpr,
    Co: tl.constexpr,
    Ho: tl.constexpr,
    Wo: tl.constexpr,
    Ci: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    w_s0, w_s1, w_s2, w_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # For padding=1, stride=2:
    # hi = ho*2 + 1 - kh; wi = wo*2 + 1 - kw
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                in_bounds = (hi >= 0) & (hi < Ho) & (wi >= 0) & (wi < Wo)
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0).to(tl.float32)
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off).to(tl.float32)
                acc += x_val * w_val

    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_kernel(
    x_ptr,  # *f16 or *bf16
    y_ptr,  # *f16 or *bf16
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

    # tanh approximation of GELU
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    gelu = 0.5 * x_val * (1.0 + tl.math.tanh(c * (x_val + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def linear_proj_kernel(
    x_ptr,    # *f16 or *bf16, shape (B, Tafter, K) flattened
    w_ptr,    # *f16 or *bf16, shape (D, K) flattened as row-major
    out_ptr,  # *f16 or *bf16, shape (B, Tafter, D) flattened
    B, T, D, K,
    x_s0, x_s1, x_s2,   # strides for x: (B, T, K)
    out_s0, out_s1, out_s2,  # strides for out: (B, T, D)
    w_row_stride,  # K, since w is (D, K) contiguous
):
    # grid = (B, T, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K):
        x_off = b_id * x_s0 + t_id * x_s1 + k * x_s2
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        w_off = d_id * w_row_stride + k
        w_val = tl.load(w_ptr + w_off).to(tl.float32)
        acc += x_val * w_val

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    tl.store(out_ptr + out_off, acc)


@triton.jit
def add_pos_embedding_kernel(
    out_ptr,     # *f16 or *bf16, shape (B, Tafter, D) flattened
    pos_ptr,     # *f16 or *bf16, shape (Tafter, D) flattened
    B, T, D,
    out_s0, out_s1, out_s2,
    pos_s0, pos_s1,
):
    # grid = (B, T, D)
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
        # Register as buffers so they move with .to(device)
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
        # input_features: (B, 1, 80, T0), dtype bfloat16, device set externally
        x = input_features.contiguous()
        B, Ci, H, W = x.shape  # Ci=1

        # Prepare output buffers
        # Conv1: (B, 384, 40, W//2)
        Ho1 = (H + 2 * 1 - 3) // 2 + 1  # padding=1, kernel=3, stride=2
        Wo1 = (W + 2 * 1 - 3) // 2 + 1
        x1 = torch.empty((B, 384, Ho1, Wo1), device=x.device, dtype=torch.float32)  # compute in fp32

        grid1 = (B, 384, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            x, self.conv2d1_weight, self.conv2d1_bias, x1,
            B=B, Ci=Ci, Co=384, Ho=Ho1, Wo=Wo1, Kh=3, Kw=3,
            x_s0=x.stride(0), x_s1=x.stride(1), x_s2=x.stride(2), x_s3=x.stride(3),
            w_s0=self.conv2d1_weight.stride(0), w_s1=self.conv2d1_weight.stride(1), w_s2=self.conv2d1_weight.stride(2), w_s3=self.conv2d1_weight.stride(3),
            y_s0=x1.stride(0), y_s1=x1.stride(1), y_s2=x1.stride(2), y_s3=x1.stride(3),
        )

        # GELU Conv1
        x1_g = torch.empty_like(x1)  # fp32
        grid1_g = (B, 384, Ho1, Wo1)
        gelu_tanh_kernel[grid1_g](
            x1, x1_g,
            B=B, Co=384, Ho=Ho1, Wo=Wo1,
            x_s0=x1.stride(0), x_s1=x1.stride(1), x_s2=x1.stride(2), x_s3=x1.stride(3),
            y_s0=x1_g.stride(0), y_s1=x1_g.stride(1), y_s2=x1_g.stride(2), y_s3=x1_g.stride(3),
        )

        # Conv2: (B, 384, 20, Wo1//2)
        Ho2 = (Ho1 + 2 * 1 - 3) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - 3) // 2 + 1
        x2 = torch.empty((B, 384, Ho2, Wo2), device=x.device, dtype=torch.float32)

        grid2 = (B, 384, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x1_g, self.conv2d2_weight, self.conv2d2_bias, x2,
            B=B, Ci=384, Co=384, Ho=Ho2, Wo=Wo2, Kh=3, Kw=3,
            x_s0=x1_g.stride(0), x_s1=x1_g.stride(1), x_s2=x1_g.stride(2), x_s3=x1_g.stride(3),
            w_s0=self.conv2d2_weight.stride(0), w_s1=self.conv2d2_weight.stride(1), w_s2=self.conv2d2_weight.stride(2), w_s3=self.conv2d2_weight.stride(3),
            y_s0=x2.stride(0), y_s1=x2.stride(1), y_s2=x2.stride(2), y_s3=x2.stride(3),
        )

        # GELU Conv2
        x2_g = torch.empty_like(x2)
        grid2_g = (B, 384, Ho2, Wo2)
        gelu_tanh_kernel[grid2_g](
            x2, x2_g,
            B=B, Co=384, Ho=Ho2, Wo=Wo2,
            x_s0=x2.stride(0), x_s1=x2.stride(1), x_s2=x2.stride(2), x_s3=x2.stride(3),
            y_s0=x2_g.stride(0), y_s1=x2_g.stride(1), y_s2=x2_g.stride(2), y_s3=x2_g.stride(3),
        )

        # Conv3: (B, 384, 10, Wo2//2)
        Ho3 = (Ho2 + 2 * 1 - 3) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - 3) // 2 + 1
        x3 = torch.empty((B, 384, Ho3, Wo3), device=x.device, dtype=torch.float32)

        grid3 = (B, 384, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x2_g, self.conv2d3_weight, self.conv2d3_bias, x3,
            B=B, Ci=384, Co=384, Ho=Ho3, Wo=Wo3, Kh=3, Kw=3,
            x_s0=x2_g.stride(0), x_s1=x2_g.stride(1), x_s2=x2_g.stride(2), x_s3=x2_g.stride(3),
            w_s0=self.conv2d3_weight.stride(0), w_s1=self.conv2d3_weight.stride(1), w_s2=self.conv2d3_weight.stride(2), w_s3=self.conv2d3_weight.stride(3),
            y_s0=x3.stride(0), y_s1=x3.stride(1), y_s2=x3.stride(2), y_s3=x3.stride(3),
        )

        # GELU Conv3
        x3_g = torch.empty_like(x3)
        grid3_g = (B, 384, Ho3, Wo3)
        gelu_tanh_kernel[grid3_g](
            x3, x3_g,
            B=B, Co=384, Ho=Ho3, Wo=Wo3,
            x_s0=x3.stride(0), x_s1=x3.stride(1), x_s2=x3.stride(2), x_s3=x3.stride(3),
            y_s0=x3_g.stride(0), y_s1=x3_g.stride(1), y_s2=x3_g.stride(2), y_s3=x3_g.stride(3),
        )

        # Now we need (B, Tafter, 3840). Original does x3.permute(0,3,1,2).contiguous().view(B, Tafter, 384*10)
        # We'll form x_flat by gathering without torch ops:
        # Ho3=10 per inputs, Wo3=Tafter. But Wo3 is computed. We need Tafter=Wo3. 3840 = 384 * 10.
        # Let's define Tafter = Wo3, and we have x3_g shape (B, 384, 10, Tafter).
        B2, Co2, Ho32, Wo32 = x3_g.shape
        assert Co2 == 384 and Ho32 == 10, "Conv3 output must be (B, 384, 10, Tafter)"
        Tafter = Wo32

        # Prepare (B, Tafter, 3840) using Triton gather
        # We'll flatten the (384, 10) into 3840 per time step.
        # Each output element is x3_g[b, co, 9, t] -> co*10 + 9 mapping is not straightforward; instead,
        # we'll launch a kernel that computes out[b, t, k] for k in [0..3839]:
        # co = k // 10, rem = k % 10, ho=9 (since last dimension), wo=t.
        out_flat = torch.empty((B2, Tafter, 3840), device=x.device, dtype=torch.float32)

        for b in range(B2):
            for t in range(Tafter):
                base = b * out_flat.stride(0) + t * out_flat.stride(1)
                for k in range(3840):
                    co = k // 10
                    rem = k % 10  # rem in [0..9]
                    x_off = b * x3_g.stride(0) + co * x3_g.stride(1) + rem * x3_g.stride(2) + t * x3_g.stride(3)
                    val = tl.load(x3_g + x_off).to(tl.float32)  # gather one scalar per (b,t,k)
                    out_off = base + k * out_flat.stride(2)
                    tl.store(out_flat + out_off, val)

        # Linear projection: (B, Tafter, 3840) @ (1024, 3840) -> (B, Tafter, 1024)
        # We'll use Triton kernel to compute this directly. Note: For efficiency, we can precompute x_flat in fp32.
        # conv_out_weight is (1024, 3840). We'll load row d across K=3840 and accumulate with out_flat[b, t, :].
        out_proj = torch.empty((B2, Tafter, 1024), device=x.device, dtype=torch.float32)

        grid4 = (B2, Tafter, 1024)
        linear_proj_kernel[grid4](
            out_flat, self.conv_out_weight, out_proj,
            B=B2, T=Tafter, D=1024, K=3840,
            x_s0=out_flat.stride(0), x_s1=out_flat.stride(1), x_s2=out_flat.stride(2),
            out_s0=out_proj.stride(0), out_s1=out_proj.stride(1), out_s2=out_proj.stride(2),
            w_row_stride=self.conv_out_weight.stride(1),  # K dimension
        )

        # Scale by embed_scale (32.0)
        out_scaled = torch.empty_like(out_proj)
        for b in range(B2):
            for t in range(Tafter):
                base = b * out_scaled.stride(0) + t * out_scaled.stride(1)
                for d in range(1024):
                    val = tl.load(out_proj + base + d * out_proj.stride(2)).to(tl.float32) * 32.0
                    out_off = base + d * out_scaled.stride(2)
                    tl.store(out_scaled + out_off, val)

        # Add positional embedding (broadcast across batch): pos shape (1500, 1024)
        # We only need first Tafter rows. Triton kernel will add pos[t, :] to each (b, t, :).
        pos = self.positional_embedding.to(torch.float32)
        out_pos = torch.empty_like(out_scaled)
        grid5 = (B2, Tafter, 1024)
        add_pos_embedding_kernel[grid5](
            out_pos, pos,
            B=B2, T=Tafter, D=1024,
            out_s0=out_pos.stride(0), out_s1=out_pos.stride(1), out_s2=out_pos.stride(2),
            pos_s0=pos.stride(0), pos_s1=pos.stride(1),
        )

        return out_pos


def run(*args):
    return ModelNew()(*args)
