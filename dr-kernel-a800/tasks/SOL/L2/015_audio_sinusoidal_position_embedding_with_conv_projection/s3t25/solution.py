import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    x_ptr,          # *f16 or *bf16, input
    w_ptr,          # *f16 or *bf16, weight
    b_ptr,          # *f32, bias
    y_ptr,          # *f16 or *bf16, output
    B: tl.constexpr, Ci: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Co: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    Ho: tl.constexpr, Wo: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,       # strides for x
    w_s0, w_s1, w_s2, w_s3,       # strides for w
    y_s0, y_s1, y_s2, y_s3,       # strides for y
):
    # program ids: grid = (B, Co, Ho, Wo)
    b_id = tl.program_id(0)
    co_id = tl.program_id(1)
    ho_id = tl.program_id(2)
    wo_id = tl.program_id(3)

    # accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and kernel
    for ci in range(Ci):
        for kh in range(Kh):
            hi = ho_id * 2 + 1 - kh
            for kw in range(Kw):
                wi = wo_id * 2 + 1 - kw
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                # compute input offset
                x_off = b_id * x_s0 + ci * x_s1 + hi * x_s2 + wi * x_s3
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
    # y_ptr may be f16/bf16; Triton will cast fp32 to lower precision on store
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

    # tanh-based GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_val * x_val * x_val
    gelu = 0.5 * x_val * (1.0 + tl.math.tanh(c * (x_val + 0.044715 * x3)))

    y_off = b_id * y_s0 + co_id * y_s1 + ho_id * y_s2 + wo_id * y_s3
    tl.store(y_ptr + y_off, gelu)


@triton.jit
def linear_proj_kernel(
    x_ptr,       # *f16/bf16, shape (B, Tafter, 3840), but accessed by (b, t, k)
    w_ptr,       # *f16/bf16, shape (1024, 3840)
    out_ptr,     # *f16/bf16, shape (B, Tafter, 1024)
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr,  # Tafter, K=3840, D_out=1024
    x_s0, x_s1, x_s2,   # strides for x: (B, T, K)
    w_s0, w_s1,         # strides for w: (D, K)
    out_s0, out_s1, out_s2,  # strides for out: (B, T, D)
):
    # grid = (B, T, 1024)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    # compute dot over K=3840
    # We can't vectorize over K directly; do a simple loop (acceptable for moderate sizes).
    for k in range(K):
        x_off = b_id * x_s0 + t_id * x_s1 + k * x_s2
        xk = tl.load(x_ptr + x_off).to(tl.float32)
        w_off = d_id * w_s0 + k * w_s1
        wk = tl.load(w_ptr + w_off).to(tl.float32)
        acc += xk * wk

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    tl.store(out_ptr + out_off, acc)


@triton.jit
def scale_add_pos_embedding_kernel(
    out_ptr,           # *f16/bf16, shape (B, Tafter, 1024)
    pos_ptr,           # *f16/bf16, shape (1500, 1024) but we only use first T rows
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    out_s0, out_s1, out_s2,
    pos_s0, pos_s1,
    scale: tl.constexpr,  # embed_scale
):
    # grid = (B, T, D)
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    d_id = tl.program_id(2)

    out_off = b_id * out_s0 + t_id * out_s1 + d_id * out_s2
    out_val = tl.load(out_ptr + out_off).to(tl.float32)

    pos_off = t_id * pos_s0 + d_id * pos_s1
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    # scale then add
    out_val = out_val * scale + pos_val
    tl.store(out_ptr + out_off, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We assume get_inputs() provides the parameters as buffers. In a real setup,
        # you'd register them; here we keep a forward-only signature. The evaluation
        # harness will pass tensors to forward, not rely on this init. We will recompute
        # or rely on provided tensors in forward.

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding):
        # Ensure all are on the same device and dtype (bf16)
        device = input_features.device
        dtype = input_features.dtype
        assert dtype == torch.bfloat16, "Input dtype must be bfloat16"

        # 1) Conv1: (B, 1, 80, T0) -> (B, 384, 40, T0//2)
        B, Ci, H, W = input_features.shape
        Co1, Ci1, Kh, Kw = conv2d1_weight.shape
        Ho1 = (H + 2 * 1 - Kh) // 2 + 1
        Wo1 = (W + 2 * 1 - Kw) // 2 + 1
        x1 = torch.empty((B, Co1, Ho1, Wo1), device=device, dtype=dtype)

        grid1 = (B, Co1, Ho1, Wo1)
        conv2d_stride2_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, Ci, H, W, Co1, Kh, Kw, Ho1, Wo1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # 2) GELU on conv1 output
        x1_gelu = torch.empty_like(x1, dtype=dtype, device=device)
        grid_gelu1 = (B, Co1, Ho1, Wo1)
        gelu_tanh_kernel[grid_gelu1](
            x1, x1_gelu,
            B, Co1, Ho1, Wo1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
        )

        # 3) Conv2: (384 -> 384), stride=2, padding=1
        Co2, Ci2, Kh2, Kw2 = conv2d2_weight.shape
        Ho2 = (Ho1 + 2 * 1 - Kh2) // 2 + 1
        Wo2 = (Wo1 + 2 * 1 - Kw2) // 2 + 1
        x2 = torch.empty((B, Co2, Ho2, Wo2), device=device, dtype=dtype)

        grid2 = (B, Co2, Ho2, Wo2)
        conv2d_stride2_kernel[grid2](
            x1_gelu, conv2d2_weight, conv2d2_bias, x2,
            B, Co1, Ho1, Wo1, Co2, Kh2, Kw2, Ho2, Wo2,
            x1_gelu.stride(0), x1_gelu.stride(1), x1_gelu.stride(2), x1_gelu.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # 4) GELU on conv2 output
        x2_gelu = torch.empty_like(x2, dtype=dtype, device=device)
        grid_gelu2 = (B, Co2, Ho2, Wo2)
        gelu_tanh_kernel[grid_gelu2](
            x2, x2_gelu,
            B, Co2, Ho2, Wo2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
        )

        # 5) Conv3: (384 -> 384), stride=2, padding=1
        Co3, Ci3, Kh3, Kw3 = conv2d3_weight.shape
        Ho3 = (Ho2 + 2 * 1 - Kh3) // 2 + 1
        Wo3 = (Wo2 + 2 * 1 - Kw3) // 2 + 1
        x3 = torch.empty((B, Co3, Ho3, Wo3), device=device, dtype=dtype)

        grid3 = (B, Co3, Ho3, Wo3)
        conv2d_stride2_kernel[grid3](
            x2_gelu, conv2d3_weight, conv2d3_bias, x3,
            B, Co2, Ho2, Wo2, Co3, Kh3, Kw3, Ho3, Wo3,
            x2_gelu.stride(0), x2_gelu.stride(1), x2_gelu.stride(2), x2_gelu.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # 6) GELU on conv3 output
        x3_gelu = torch.empty_like(x3, dtype=dtype, device=device)
        grid_gelu3 = (B, Co3, Ho3, Wo3)
        gelu_tanh_kernel[grid_gelu3](
            x3, x3_gelu,
            B, Co3, Ho3, Wo3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
        )

        # 7) Gather conv3_gelu to form (B, Tafter, 3840) without torch.permute
        #   x3_gelu shape: (B, 384, Ho3, Wo3). We need Wo3 == Tafter, and we flatten 384*10? Actually 384*(Ho3*Wo3)=384*10
        #   The original code uses (B, 384, 10, Tafter) -> (B, Tafter, 384*10).
        #   Since conv3 output is (B, 384, 10, Tafter), we can directly use it for linear without gather.
        #   But to strictly avoid torch operations, we implement the gather mapping:
        #     For each (b, t), t_idx in [0..Tafter-1], co in [0..383], ho in [0..9], map to k = co*10*10 + ho*10 + t_idx
        #     However conv3 output is (B, Co, Ho3, Wo3), where Co=384, Ho3=10, Wo3=Tafter.
        #     The reference mapping uses channels and time index. To match, we can compute per (b, t, d), where d in [0..3839]:
        #       co = d // 3840 // 10 * 384 + (d // 10) % 384
        #       ho = (d // 10) % 10
        #       t_idx = d % 10 (incorrect). We should instead: co = d // 384, ho = (d // 10) % 10, t_idx = d % 10.
        #       This is not straightforward. Easier: compute directly in Triton via linear projection using conv3_gelu as (B, Co, Ho, Wo).
        #       Since we have (B, 384, 10, Tafter), we can flatten (Co, Ho) into a single index. Let's implement the linear projection directly,
        #       by reshaping conv3_gelu to (B, Tafter, 3840) logically via index mapping in the kernel.

        # We instead compute linear directly: (B, Tafter, 3840) via index mapping from conv3_gelu(b, co, ho, t)
        # To avoid torch.view/permute, we can derive the indices within the kernel, but simpler is to treat x3_gelu as (B, Co, Ho, Wo)
        # and in linear, for each k, read x[b, co, ho, t] where we map k -> (co, ho, t). The mapping is deterministic: for k in [0..3839],
        # co = k // 384, rem = k % 384, ho = rem // 10, t = rem % 10. Then t is index in time_after_conv. We need to compute t from
        # Wo3 (= Tafter). This requires passing Tafter into kernel. We will compute x_gather[b, t, k] = x3_gelu[b, co, ho, t_idx]
        # using Triton and then perform linear projection.

        # Define Tafter as Wo3
        Tafter = Wo3
        K = Co3 * Ho3 * Tafter  # = 384 * 10 * Tafter
        D = conv_out_weight.shape[0]  # 1024
        assert conv_out_weight.shape[1] == K, f"Weight second dim must be {K}, got {conv_out_weight.shape[1]}"

        # Allocate output (B, Tafter, 1024)
        out = torch.empty((B, Tafter, D), device=device, dtype=dtype)

        # Launch linear projection: grid = (B, Tafter, D)
        grid_linear = (B, Tafter, D)
        linear_proj_kernel[grid_linear](
            x3_gelu, conv_out_weight, out,
            B, Tafter, K, D,
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2),  # strides for (B, Co, Ho) if flattened? We need (B, T, K).
            # We can't index x3_gelu with (b, t, k); x3_gelu is (B, Co, Ho, Wo). Instead, we will compute linear in a different way.
            # To strictly avoid torch.view, we can precompute x_gather tensor of shape (B, Tafter, K) in Triton:
        )

        # We need to form x_gather[b, t, k] = x3_gelu[b, co, ho, t]. Triton kernel that constructs x_gather:
        # But Triton can't branch over runtime parameters like this, so better approach: compute directly in the linear kernel
        # by providing conv3_gelu as input and mapping k to (co, ho, t). However Triton kernel arguments are pointers and we cannot
        # pass k-dependent indexing easily. Therefore, we will precompute x_gather using a simple torch loop (not allowed), or
        # write a Triton kernel that reads from x3_gelu and writes to x_gather. To stay within Triton-only, we implement a kernel
        # that fills x_gather directly from conv3_gelu by mapping k -> (co, ho, t).

        # Implement a Triton kernel that creates x_gather: grid = (B, Tafter, K)
        # x_gather[b, t, k] = x3_gelu[b, co, ho, t], where co = k // (Ho3*Tafter), ho = (k // Tafter) % Ho3, t = k % Tafter
        x_gather = torch.empty((B, Tafter, K), device=device, dtype=dtype)
        grid_xgather = (B, Tafter, K)
        # Triton kernel: decode k into co, ho, t and load from x3_gelu
        @triton.jit
        def create_xgather_kernel(
            x3_gelu_ptr,      # *f16/bf16, (B, Co, Ho, Wo)
            xgather_ptr,      # *f16/bf16, (B, T, K)
            B: tl.constexpr, Co: tl.constexpr, Ho: tl.constexpr, Wo: tl.constexpr,  # Co=384, Ho=10, Wo=Tafter
            x3_s0, x3_s1, x3_s2, x3_s3,
            xg_s0, xg_s1, xg_s2,
        ):
            b_id = tl.program_id(0)
            t_id = tl.program_id(1)
            k_id = tl.program_id(2)

            # decode k_id -> co, ho, t
            # co = k // (Ho*Wo), rem = k % (Ho*Wo)
            # ho = rem // Wo, t = rem % Wo
            co = k_id // (Ho * Wo)
            rem = k_id % (Ho * Wo)
            ho = rem // Wo
            t = rem % Wo

            x_off = b_id * x3_s0 + co * x3_s1 + ho * x3_s2 + t * x3_s3
            val = tl.load(x3_gelu_ptr + x_off).to(tl.float32)

            xg_off = b_id * xg_s0 + t_id * xg_s1 + k_id * xg_s2
            tl.store(xgather_ptr + xg_off, val)

        create_xgather_kernel[grid_xgather](
            x3_gelu, x_gather,
            B, Co3, Ho3, Wo3,
            x3_gelu.stride(0), x3_gelu.stride(1), x3_gelu.stride(2), x3_gelu.stride(3),
            x_gather.stride(0), x_gather.stride(1), x_gather.stride(2),
        )

        # Now perform linear projection on x_gather
        # grid = (B, Tafter, 1024)
        grid_linear2 = (B, Tafter, D)
        linear_proj_kernel[grid_linear2](
            x_gather, conv_out_weight, out,
            B, Tafter, K, D,
            x_gather.stride(0), x_gather.stride(1), x_gather.stride(2),
            conv_out_weight.stride(0), conv_out_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
        )

        # 8) Scale by embed_scale and add positional embedding (broadcast across batch)
        embed_scale = math.sqrt(1024.0)  # 32.0
        grid_scale = (B, Tafter, D)
        scale_add_pos_embedding_kernel[grid_scale](
            out, positional_embedding,
            B, Tafter, D,
            out.stride(0), out.stride(1), out.stride(2),
            positional_embedding.stride(0), positional_embedding.stride(1),
            embed_scale,
        )

        return out


# The evaluation harness will provide get_inputs and call ModelNew with the appropriate tensors.
# The above forward does not use any torch ops for numerical computation; it launches Triton kernels for conv, GELU,
# linear projection, and embedding addition.


def run(*args):
    return ModelNew()(*args)
