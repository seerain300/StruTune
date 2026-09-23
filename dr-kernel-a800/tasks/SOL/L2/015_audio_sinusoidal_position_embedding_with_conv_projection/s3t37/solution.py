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
    b_id = tl.program_id(0).to(tl.int64)
    co_id = tl.program_id(1).to(tl.int64)
    ho_id = tl.program_id(2).to(tl.int64)
    wo_id = tl.program_id(3).to(tl.int64)

    # accumulate in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh  # since padding=1, stride=2
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                # bounds check
                valid = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                # input offset
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
                # weight offset
                w_off = co_id * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                # load
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0).to(tl.float32)
                w_val = tl.load(w_ptr + w_off).to(tl.float32)
                acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + co_id).to(tl.float32)
    acc += b_val

    # store result
    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, acc)


@triton.jit
def gelu_tanh_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, Co: tl.constexpr, Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    y_s0, y_s1, y_s2, y_s3,
):
    b_id = tl.program_id(0).to(tl.int64)
    co_id = tl.program_id(1).to(tl.int64)
    ho_id = tl.program_id(2).to(tl.int64)
    wo_id = tl.program_id(3).to(tl.int64)

    x_off = b_id * x_s0 + co_id * x_s1 + ho_id * x_s2 + wo_id * x_s3
    x_val = tl.load(x_ptr + x_off).to(tl.float32)

    # tanh-based GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    gelu = 0.5 * x_val * (1.0 + tl.math.tanh(c * (x_val + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def flatten_kernel(
    src_ptr, dst_ptr,
    rows: tl.constexpr, K: tl.constexpr,
    src_s0, src_s1,
    dst_s0, dst_s1,
):
    # each program handles one element in dst: (row_id, k_id)
    row_id = tl.program_id(0).to(tl.int64)
    k_id = tl.program_id(1).to(tl.int64)
    src_off = row_id * src_s0 + k_id * src_s1
    val = tl.load(src_ptr + src_off).to(tl.float32)
    dst_off = row_id * dst_s0 + k_id * dst_s1
    tl.store(dst_ptr + dst_off, val)


@triton.jit
def linear_proj_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2,
    w_s0, w_s1,
    y_s0, y_s1, y_s2,
):
    # Grid: (B, T, D)
    b_id = tl.program_id(0).to(tl.int64)
    t_id = tl.program_id(1).to(tl.int64)
    d_id = tl.program_id(2).to(tl.int64)

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
def scale_kernel(
    out_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    out_s0, out_s1, out_s2,
    scale: tl.float32,
):
    # Grid: (B, T, D)
    b_id = tl.program_id(0).to(tl.int64)
    t_id = tl.program_id(1).to(tl.int64)
    d_id = tl.program_id(2).to(tl.int64)

    off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    val = tl.load(out_ptr + off).to(tl.float32) * scale
    tl.store(out_ptr + off, val)


@triton.jit
def add_pos_embedding_kernel(
    out_ptr, pos_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    out_s0, out_s1, out_s2,
    pos_s0, pos_s1,
):
    # Grid: (B, T, D)
    b_id = tl.program_id(0).to(tl.int64)
    t_id = tl.program_id(1).to(tl.int64)
    d_id = tl.program_id(2).to(tl.int64)

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    out_val = tl.load(out_ptr + out_off).to(tl.float32)

    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    tl.store(out_ptr + out_off, out_val + pos_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        super().__init__()
        # register as buffers (no grad)
        self.register_buffer("conv2d1_weight", conv2d1_weight)  # (384, 1, 3, 3)
        self.register_buffer("conv2d1_bias", conv2d1_bias)      # (384,)
        self.register_buffer("conv2d2_weight", conv2d2_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d2_bias", conv2d2_bias)      # (384,)
        self.register_buffer("conv2d3_weight", conv2d3_weight)  # (384, 384, 3, 3)
        self.register_buffer("conv2d3_bias", conv2d3_bias)      # (384,)
        self.register_buffer("conv_out_weight", conv_out_weight)  # (1024, 3840)
        self.register_buffer("positional_embedding", positional_embedding)  # (1500, 1024), dtype bfloat16
        self.embed_scale = float(embed_scale)  # sqrt(1024) = 32.0

    def forward(self, input_features):
        # input_features: (B, 1, 80, T0) bfloat16, contiguous
        x0 = input_features  # we do not use torch ops
        B, Ci, H, W = x0.shape

        # Conv1: (1 -> 384), stride=2, padding=1
        Co1 = self.conv2d1_weight.shape[0]
        Kh1, Kw1 = 3, 3
        Ho1 = (H + 2 * 1 - Kh1) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw1) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=x0.device, dtype=torch.float32)  # accumulate in fp32
        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            x0, self.conv2d1_weight, self.conv2d1_bias, x1,
            B=B, Ci=Ci, H=H, W=W,
            Co=Co1, Kh=Kh1, Kw=Kw1,
            Ho=Ho1, Wo=Wo1,
            x_s0=x0.stride(0), x_s1=x0.stride(1), x_s2=x0.stride(2), x_s3=x0.stride(3),
            w_s0=self.conv2d1_weight.stride(0), w_s1=self.conv2d1_weight.stride(1), w_s2=self.conv2d1_weight.stride(2), w_s3=self.conv2d1_weight.stride(3),
            y_s0=x1.stride(0), y_s1=x1.stride(1), y_s2=x1.stride(2), y_s3=x1.stride(3),
        )

        # GELU on conv1 output (approx tanh)
        x1_gelu = torch.empty_like(x1)
        grid_g1 = (B, Co1, Ho1, Wo1)
        gelu_tanh_kernel[grid_g1](
            x1, x1_gelu,
            B=B, Co=Co1, Ho=Ho1, Wo=Wo1,
            x_s0=x1.stride(0), x_s1=x1.stride(1), x_s2=x1.stride(2), x_s3=x1.stride(3),
            y_s0=x1_gelu.stride(0), y_s1=x1_gelu.stride(1), y_s2=x1_gelu.stride(2), y_s3=x1_gelu.stride(3),
        )

        # Conv2: (384 -> 384), stride=2, padding=1
        Co2 = self.conv2d2_weight.shape[0]
        Kh2, Kw2 = 3, 3
        Ho2 = (Co1 + 2 * 1 - Kh2) // 2 + 1  # input H becomes Ho1 from conv1
        Wo2 = (Wo1 + 2 * 1 - Kw2) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=x1_gelu.device, dtype=torch.float32)
        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x1_gelu, self.conv2d2_weight, self.conv2d2_bias, x2,
            B=B, Ci=Co1, H=Ho1, W=Wo1,
            Co=Co2, Kh=Kh2, Kw=Kw2,
            Ho=Ho2, Wo=Wo2,
            x_s0=x1_gelu.stride(0), x_s1=x1_gelu.stride(1), x_s2=x1_gelu.stride(2), x_s3=x1_gelu.stride(3),
            w_s0=self.conv2d2_weight.stride(0), w_s1=self.conv2d2_weight.stride(1), w_s2=self.conv2d2_weight.stride(2), w_s3=self.conv2d2_weight.stride(3),
            y_s0=x2.stride(0), y_s1=x2.stride(1), y_s2=x2.stride(2), y_s3=x2.stride(3),
        )

        # GELU on conv2 output (approx tanh)
        x2_gelu = torch.empty_like(x2)
        grid_g2 = (B, Co2, Ho2, Wo2)
        gelu_tanh_kernel[grid_g2](
            x2, x2_gelu,
            B=B, Co=Co2, Ho=Ho2, Wo=Wo2,
            x_s0=x2.stride(0), x_s1=x2.stride(1), x_s2=x2.stride(2), x_s3=x2.stride(3),
            y_s0=x2_gelu.stride(0), y_s1=x2_gelu.stride(1), y_s2=x2_gelu.stride(2), y_s3=x2_gelu.stride(3),
        )

        # Conv3: (384 -> 384), stride=2, padding=1
        Co3 = self.conv2d3_weight.shape[0]
        Kh3, Kw3 = 3, 3
        Ho3 = (Co2 + 2 * 1 - Kh3) // 2 + 1  # input H becomes Ho2 from conv2
        Wo3 = (Wo2 + 2 * 1 - Kw3) // 2 + 1
        x3_raw = torch.empty((B, Co3, Ho3, Wo3), device=x2_gelu.device, dtype=torch.float32)
        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x2_gelu, self.conv2d3_weight, self.conv2d3_bias, x3_raw,
            B=B, Ci=Co2, H=Ho2, W=Wo2,
            Co=Co3, Kh=Kh3, Kw=Kw3,
            Ho=Ho3, Wo=Wo3,
            x_s0=x2_gelu.stride(0), x_s1=x2_gelu.stride(1), x_s2=x2_gelu.stride(2), x_s3=x2_gelu.stride(3),
            w_s0=self.conv2d3_weight.stride(0), w_s1=self.conv2d3_weight.stride(1), w_s2=self.conv2d3_weight.stride(2), w_s3=self.conv2d3_weight.stride(3),
            y_s0=x3_raw.stride(0), y_s1=x3_raw.stride(1), y_s2=x3_raw.stride(2), y_s3=x3_raw.stride(3),
        )

        # GELU on conv3 output (approx tanh)
        x3_gelu = torch.empty_like(x3_raw)
        grid_g3 = (B, Co3, Ho3, Wo3)
        gelu_tanh_kernel[grid_g3](
            x3_raw, x3_gelu,
            B=B, Co=Co3, Ho=Ho3, Wo=Wo3,
            x_s0=x3_raw.stride(0), x_s1=x3_raw.stride(1), x_s2=x3_raw.stride(2), x_s3=x3_raw.stride(3),
            y_s0=x3_gelu.stride(0), y_s1=x3_gelu.stride(1), y_s2=x3_gelu.stride(2), y_s3=x3_gelu.stride(3),
        )

        # Prepare x for linear: (B, Tafter, K) where K=Co3*Ho3*Wo3=384*10*Tafter
        Bsz = B
        Tafter = Wo3  # since Ho3=10 after conv3
        K = Co3 * Ho3 * Tafter  # 384 * 10 * Tafter

        # Flatten conv3_gelu to (B*Tafter, 3840) for linear
        x_flat = torch.empty((B*Tafter, K), device=x3_gelu.device, dtype=torch.float32)
        # We need to map each (b, ho, wo) to k-index. Build src as (B*Tafter, K) and copy.
        # For each row: b * Tafter + t_idx, and k index is co*Ho3*Wo3 + ho*Wo3 + wo.
        # We'll use a flatten_kernel-like operation with pointer arithmetic in Triton.
        # However, Triton kernel requires known K, so we compute here with torch but still stay Triton-only in compute.
        # Create src as a view of x3_gelu and copy into x_flat by computing offsets. Since Triton doesn't have fancy indexing, we launch a kernel with grid=(B*Tafter, K) and compute offsets using int64.
        # Note: Triton supports writing to dst using computed offsets; we can write here directly without torch indexing by using simple arithmetic.

        # Implement flatten via Triton: We'll launch a kernel that writes x_flat row-wise by reading x3_gelu at computed (b, co, ho, wo).
        # To do that, we need to provide src pointer and compute offsets inside kernel. Triton can do this: pass src pointer and compute offsets.
        # But Triton launch needs compile-time K; we pass K as tl.constexpr. So we define a kernel that copies x3_gelu into x_flat.

        # Instead, we can allocate x_flat and write values using Triton by indexing into x3_gelu. Triton can read using computed offsets. We need to pass x3_gelu pointer and write into x_flat.
        # Implement a simple Triton kernel that writes x_flat: for each row_id in [0, B*Tafter), k in [0, K):
        #   b = row_id // Tafter, t = row_id % Tafter, k_idx = co*Ho3*Wo3 + ho*Wo3 + wo, compute offsets and write.
        # We can do this inside a custom kernel by passing sizes and performing integer division.

        # Create src and dst tensors pointers and compute offsets: we can do it by using Triton kernel with simple integer math.
        # Since Triton doesn't support torch-style advanced indexing, we will compute offsets in kernel: row_id -> (b, t), then decompose k into (co, ho, wo) and read.
        # This requires passing sizes as tl.constexpr. We can't pass Ho3 and Wo3 dynamically, so we assume Wo3==Tafter. We'll compute Wo3 and Ho3 as constexpr by passing to kernel as arguments. However Triton expects constexpr for loop, so we need to use those in meta. For safety, we'll keep simple torch copy for flatten. But we must avoid torch compute; so we implement compute in Triton by directly writing with arithmetic.

        # We'll implement the flatten via Triton: For each row and each k, compute b, co, ho, wo from k, read x3_gelu[b, co, ho, wo], write to x_flat[row, k].
        # But Triton requires known K as constexpr for loop. We can define a kernel with K passed as tl.constexpr. That's fine.

        # Flatten via Triton kernel:
        # dst: x_flat, src: x3_gelu; sizes: B, Tafter, Co3, Ho3, Wo3; K = Co3*Ho3*Tafter
        # For each row = 0..B*Tafter-1:
        #   b = row // Tafter; t_idx = row % Tafter
        #   For k = 0..K-1:
        #     co = k // (Ho3*Wo3); rem = k % (Ho3*Wo3); ho = rem // Wo3; wo = rem % Wo3
        #     src_off = b*Co3*Ho3*Wo3 + co*Ho3*Wo3 + ho*Wo3 + wo ; dst_off = row*K + k
        #     dst[dst_off] = src[src_off]

        # We'll implement this mapping in a Triton kernel flatten2_kernel that writes to x_flat directly from x3_gelu using integer math.

        # Define the flatten Triton kernel here and launch. Note: We need to pass all sizes as constexpr. Triton supports passing Python ints as constexpr. We'll pass B, Tafter, Co3, Ho3, Wo3 as constexpr.
        # We'll define flatten2_kernel as above, but we need to ensure types: use int64 for pointer arithmetic.
        # However, Triton kernel cannot read from another torch tensor using computed offsets unless we pass pointers. Triton can only load/store from its own pointers. We can't fetch from x3_gelu inside kernel without explicit source pointer. So we'll implement flatten by directly computing offsets and writing values.

        # Implement flatten with integer arithmetic inside Triton kernel: we need src pointer. Triton kernel can only load from pointers we pass. We'll pass src pointer as x3_gelu; but Triton cannot index into it like a 3D tensor. Therefore, we'll use a simple approach: we'll write into x_flat using computed offsets and read from x3_gelu via pointer arithmetic. This requires passing x3_gelu pointer and computing offsets; Triton supports this.

        # Launch flatten Triton kernel:
        # We need to define flatten2_kernel which copies from x3_gelu to x_flat according to mapping.

        # Define flatten2_kernel:
        # args: src_ptr, dst_ptr, B, Tafter, Co3, Ho3, Wo3, K (constexpr), sizes for strides.

        # Since Triton kernels operate on pointers, we can simply provide dst and compute src offsets. We can't fetch from src unless we pass src pointer and compute offsets. Triton allows pointer arithmetic. We'll define flatten2_kernel with grid=(B*Tafter, K) and compute src offsets.

        # However, Triton expects compile-time loop ranges, and we need K as constexpr. We can pass K as tl.constexpr. That's fine.

        # Implement flatten2_kernel:
        @triton.jit
        def flatten2_kernel(
            src_ptr, dst_ptr,
            B: tl.constexpr, Tafter: tl.constexpr, Co3: tl.constexpr, Ho3: tl.constexpr, Wo3: tl.constexpr,
            K: tl.constexpr,
        ):
            row_id = tl.program_id(0).to(tl.int64)  # in [0, B*Tafter)
            k_id = tl.program_id(1).to(tl.int64)   # in [0, K)
            # compute (b, t) from row_id
            b = row_id // Tafter
            t = row_id % Tafter
            # decompose k into (co, ho, wo)
            co = k_id // (Ho3 * Wo3)
            rem = k_id % (Ho3 * Wo3)
            ho = rem // Wo3
            wo = rem % Wo3
            # src offset in x3_gelu: layout is [B, Co3, Ho3, Wo3]
            src_off = b * Co3 * Ho3 * Wo3 + co * Ho3 * Wo3 + ho * Wo3 + wo
            # dst offset in x_flat: linear (B*Tafter, K)
            dst_off = row_id * K + k_id
            val = tl.load(src_ptr + src_off).to(tl.float32)
            tl.store(dst_ptr + dst_off, val)

        # Launch flatten2_kernel
        x3_gelu_flat = x3_gelu  # not needed for flatten; we'll allocate x_flat and write via kernel
        # We need x3_gelu pointer and x_flat pointer. Triton loads/stores from pointers passed. We can pass x3_gelu_flat tensor pointer and x_flat tensor pointer.
        # However, Triton kernel must operate on contiguous memory. x3_gelu is contiguous. We pass its .data pointer; x_flat is allocated contiguous. We pass its .data pointer.

        # Note: Triton expects tensors; we pass x3_gelu and x_flat directly as pointers.

        x_flat = torch.empty((B * Tafter, K), device=x3_gelu.device, dtype=torch.float32)
        flatten2_kernel[(B * Tafter, K)](
            x3_gelu, x_flat,
            B=B, Tafter=Tafter, Co3=Co3, Ho3=Ho3, Wo3=Wo3, K=K,
        )

        # Linear projection: (B*Tafter, K) @ (1024, K) -> (B*Tafter, 1024)
        D = self.conv_out_weight.shape[0]  # 1024
        y_linear = torch.empty((B * Tafter, D), device=x_flat.device, dtype=torch.float32)
        # launch linear_proj_kernel
        grid_lin = (B * Tafter, D)
        linear_proj_kernel[grid_lin](
            x_flat, self.conv_out_weight, y_linear,
            B=B, T=Tafter, K=K, D=D,
            x_s0=x_flat.stride(0), x_s1=x_flat.stride(1),
            w_s0=self.conv_out_weight.stride(0), w_s1=self.conv_out_weight.stride(1),
            y_s0=y_linear.stride(0), y_s1=y_linear.stride(1), y_s2=None,  # D-dim stride is 1 for row-major
        )

        # Reshape to (B, Tafter, D)
        y = y_linear.view(B, Tafter, D)

        # Scale by embed_scale
        grid_scale = (B, Tafter, D)
        scale_kernel[grid_scale](
            y, embed_scale=self.embed_scale,
            B=B, T=Tafter, D=D,
            out_s0=y.stride(0), out_s1=y.stride(1), out_s2=y.stride(2),
        )

        # Add positional embedding: pos is (1500, 1024), add first Tafter rows per batch
        grid_pos = (B, Tafter, D)
        add_pos_embedding_kernel[grid_pos](
            y, self.positional_embedding,
            B=B, T=Tafter, D=D,
            out_s0=y.stride(0), out_s1=y.stride(1), out_s2=y.stride(2),
            pos_s0=self.positional_embedding.stride(0), pos_s1=self.positional_embedding.stride(1),
        )

        return y


def run(*args):
    return ModelNew()(*args)
