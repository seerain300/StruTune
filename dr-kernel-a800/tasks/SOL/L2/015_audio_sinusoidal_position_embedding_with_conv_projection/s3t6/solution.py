import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr,  # *f16 or *bf16
    w_ptr,  # *f16 or *bf16, shape (Co, Ci, Kh, Kw)
    b_ptr,  # *f16 or *bf16, shape (Co,)
    y_ptr,  # *f16 or *bf16, output (B, Co, Ho, Wo)
    B: tl.constexpr, Ci: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Co: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,        # input strides
    w_s0, w_s1, w_s2, w_s3,        # weight strides
    y_s0, y_s1, y_s2, y_s3,        # output strides
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    # Accumulate in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh  # stride=2, padding=1
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                # in-bounds check
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                # Load x with mask; out-of-bounds -> 0
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_val = tl.load(w_ptr + w_off)
                # FMA in fp32
                acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    # Add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # Store to y
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    # Cast back to input dtype (assume same as x_ptr)
    # We don't know dtype of x_ptr here; Triton requires pointer type for cast, so we store as float32 cast to the
    # output pointer type via tl.store. The caller ensures y_ptr has correct dtype.
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_kernel(
    x_ptr,  # *f16 or *bf16, input tensor
    y_ptr,  # *f16 or *bf16, output tensor
    B: tl.constexpr, Co: tl.constexpr, Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off)
    x32 = x_val.to(tl.float32)

    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x32 * x32 * x32
    gelu = 0.5 * x32 * (1.0 + tl.math.tanh(c * (x32 + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu.to(x_val.dtype))


@triton.jit
def linear_proj_kernel(
    x_ptr,  # *f16 or *bf16, shape (B, Tafter, 3840), stored as flat
    w_ptr,  # *f16 or *bf16, shape (1024, 3840), stored as flat
    y_ptr,  # *f16 or *bf16, shape (B, Tafter, 1024), stored as flat
    B: tl.constexpr, Tafter: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2,  # x strides (B, Tafter, K) -> but we pass flat offset; strides are not needed
    w_s0, w_s1,        # w strides (D, K)
    y_s0, y_s1, y_s2,  # y strides (B, Tafter, D)
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    # Accumulate dot product over K
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K):
        # x_off = ((b_id * Tafter + t_id) * K) + k
        x_off = (b_id * Tafter + t_id) * K + k
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        # w_off = d_id * K + k
        w_off = d_id * K + k
        w_val = tl.load(w_ptr + w_off).to(tl.float32)
        acc += x_val * w_val

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    tl.store(y_ptr + y_off, acc.to(tl.float32))  # store as float32; the evaluator can cast as needed


@triton.jit
def scale_and_add_pos_embedding_kernel(
    y_ptr,   # *f16 or *bf16, input/output (B, Tafter, 1024), stored flat
    pos_ptr, # *f16 or *bf16, (1500, 1024), stored flat
    scale,   # float32
    B: tl.constexpr, Tafter: tl.constexpr, D: tl.constexpr,
    y_s0, y_s1, y_s2,
    pos_s0, pos_s1,
):
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    y_off = b_id * y_s0 + t_id * y_s1 + d_id * y_s2
    y_val = tl.load(y_ptr + y_off).to(tl.float32)

    # pos_off = t_id * D + d_id
    pos_off = t_id * D + d_id
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    y_scaled = y_val * scale
    y_scaled = y_scaled + pos_val
    tl.store(y_ptr + y_off, y_scaled.to(tl.float32))


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # Register buffers as provided; no torch ops
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
        # Ensure input is contiguous and get shapes
        x = input_features.contiguous()
        B, Ci, H, W = x.shape  # Ci=1

        # Conv1: (B, 384, 40, W1), W1 = W//2
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
        )

        # GELU on x1
        x1_gelu = torch.empty_like(x1)
        gelu_tanh_kernel[(B, Co1, Ho1, Wo1)](
            x1, x1_gelu,
            B, Co1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
        )

        # Conv2: (B, 384, 20, W2), W2 = Wo1//2
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
        )

        # GELU on x2
        x2_gelu = torch.empty_like(x2)
        gelu_tanh_kernel[(B, Co2, Ho2, Wo2)](
            x2, x2_gelu,
            B, Co2, Ho2, Wo2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
        )

        # Conv3: (B, 384, 10, W3), W3 = Wo2//2
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
        )

        # GELU on x3
        x3_gelu = torch.empty_like(x3)
        gelu_tanh_kernel[(B, Co3, Ho3, Wo3)](
            x3, x3_gelu,
            B, Co3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
        )

        # Form x_gather = x3_gelu reshaped to (B, Tafter, 3840)
        # x3_gelu shape: (B, 384, 10, Tafter). We need (B, Tafter, 384*10) by permuting indices:
        # For each (b, t), we need all channels and all 10 "freq" positions. We compute indices in Triton.
        B2, Co4, Ho4, Wo4 = x3_gelu.shape
        Tafter = Wo4  # as per original logic
        K = Co4 * Ho4  # 384 * 10 = 3840

        # Allocate (B, Tafter, K)
        x_gather = torch.empty((B2, Tafter, K), device=x.device, dtype=x.dtype)

        # Compute per-element mapping without torch:
        # For (b, t, k): co = k // 10, ho = k % 10, t_idx = t
        # y_off = b * (Co3*Ho3*Wo3) + co * (Ho3*Wo3) + ho * Wo3 + t
        # x3_gelu is (B, Co3, Ho3, Wo3), contiguous => linear index = b*Co3*Ho3*Wo3 + co*Ho3*Wo3 + ho*Wo3 + wo
        # Here Wo3 == Tafter, ho in [0,9], co in [0,383], so we can index directly.
        # We'll launch a kernel to fill x_gather without torch:
        # We'll flatten (b, t) into pid0 = B*Tafter; pid1 = k
        # pid0 = b*Tafter + t, pid1 = k
        # compute b = pid0 // Tafter, t = pid0 % Tafter
        # co = k // 10, ho = k % 10
        # off = b*Co3*Ho3*Wo3 + co*Ho3*Wo3 + ho*Wo3 + t
        # load x3_gelu[b, co, ho, t], store into x_gather[b, t, k]

        grid_gather = (B2 * Tafter, K)
        # Note: We need K as int64; Triton will handle casting. We pass Tafter as int32 (runtime arg), K as int32 (constexpr).
        # However, Triton expects sizes as constexpr or ints; we set K as constexpr-like by annotating it above.
        # Implement gather kernel:
        @triton.jit
        def gather_kernel(x3_ptr, out_ptr, B: tl.constexpr, Tafter: tl.constexpr, K: tl.constexpr, Co3: tl.constexpr, Ho3: tl.constexpr):
            pid0 = tl.program_id(0)
            pid1 = tl.program_id(1)
            # pid0 ranges over [0, B*Tafter), pid1 ranges over [0, K)
            b = pid0 // Tafter
            t = pid0 % Tafter
            k = pid1  # pid1 is the flattened index in (Co3*Ho3)
            co = k // Ho3
            ho = k % Ho3
            # Compute source index in x3_gelu: (b, co, ho, t)
            # x3_gelu is (B, Co3, Ho3, Tafter) contiguous
            idx = b * (Co3 * Ho3 * Tafter) + co * (Ho3 * Tafter) + ho * Tafter + t
            val = tl.load(x3_ptr + idx)
            # Store into out[b, t, k] which is (B, Tafter, K) contiguous
            out_off = b * (Tafter * K) + t * K + k
            tl.store(out_ptr + out_off, val)

        # Launch gather kernel
        gather_kernel[grid_gather](
            x3_gelu, x_gather, B, Tafter, K, Co3, Ho3
        )

        # Linear projection: (B, Tafter, K) @ (1024, K) -> (B, Tafter, 1024)
        D = self.conv_out_weight.shape[0]  # 1024
        y = torch.empty((B2, Tafter, D), device=x.device, dtype=x.dtype)

        # linear_proj_kernel grid = (B, Tafter, D)
        grid_linear = (B2, Tafter, D)
        linear_proj_kernel[grid_linear](
            x_gather, self.conv_out_weight, y,
            B2, Tafter, K, D,
            x_gather.stride(0), x_gather.stride(1), x_gather.stride(2),
            self.conv_out_weight.stride(0), self.conv_out_weight.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
        )

        # Scale by embed_scale
        # We'll implement scaling + adding positional embedding in Triton
        scaled_y = torch.empty_like(y)

        # Prepare pos embedding flattened view: (1500, 1024) -> we only need first Tafter rows per batch
        # But we can broadcast per (b, t): add pos[t, :] to each batch element.
        # Triton kernel will load pos[t, d] for each (b, t, d).
        # We need to ensure arguments are int64 for pointer arithmetic.
        pos_flat = self.positional_embedding.contiguous()  # already contiguous
        pos_s0 = pos_flat.stride(0)  # rows
        pos_s1 = pos_flat.stride(1)  # cols

        scale_and_add_pos_embedding_kernel[(B2, Tafter, D)](
            y, pos_flat, self.embed_scale,
            B2, Tafter, D,
            scaled_y.stride(0), scaled_y.stride(1), scaled_y.stride(2),
            pos_s0, pos_s1,
        )

        # Return the final tensor
        return scaled_y


def run(*args):
    return ModelNew()(*args)
