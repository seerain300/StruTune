import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# =========================
# Triton kernels
# =========================

@triton.jit
def conv_ci1_stride2_bias_gelu_kernel(
    x_ptr,            # input [N, 1, F_in, T_in], bfloat16
    w_ptr,            # weights [Co, 1, 3, 3], bfloat16
    b_ptr,            # bias [Co], bfloat16
    y_ptr,            # output [N, Co, F_in, T_out], bfloat16
    N, F_in, T_in, Co, T_out,
    BLOCK_CO: tl.constexpr,
):
    # program id across batch and channel blocks
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < Co

    # initialize accumulator
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # iterate over time positions
    for to in range(0, T_out):
        t_in = to * 2 + 1
        # iterate over frequency positions
        for fi in range(0, F_in):
            # base input index for this (n, fi, t_in)
            base = pid_n * (F_in * T_in) + fi * T_in + t_in
            # load input value (Ci=1)
            x_val = tl.load(x_ptr + base, mask=True, other=0.0).to(tl.float32)

            # accumulate over 3x3 kernel
            for kh in range(0, 3):
                for kw in range(0, 3):
                    fi_k = fi + kh - 1
                    ti_k = t_in + kw - 1
                    in_bounds = (0 <= fi_k < F_in) and (0 <= ti_k < T_in)
                    if in_bounds:
                        # compute input index for (fi_k, ti_k)
                        base_k = pid_n * (F_in * T_in) + fi_k * T_in + ti_k
                        xk = tl.load(x_ptr + base_k, mask=True, other=0.0).to(tl.float32)
                        # load corresponding weight for co block
                        w_idx = co_offsets * (1 * 3 * 3) + (kh * 3 + kw)
                        w_vals = tl.load(w_ptr + w_idx, mask=mask_co, other=0.0).to(tl.float32)
                        acc += xk * w_vals

    # add bias and apply GELU (tanh approximation)
    b_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
    acc += b_vals

    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + 0.044715 * x3)))

    # store result
    y_base = pid_n * (Co * F_in * T_out) + co_offsets * (F_in * T_out) + fi * T_out + to
    tl.store(y_ptr + y_base, gelu.to(tl.bfloat16), mask=mask_co)


@triton.jit
def conv_stride2_bias_gelu_3d_kernel(
    x_ptr,            # input [N, Ci, F_in, T_in], bfloat16
    w_ptr,            # weights [Co, Ci, 3, 3], bfloat16
    b_ptr,            # bias [Co], bfloat16
    y_ptr,            # output [N, Co, F_out, T_out], bfloat16
    N, Ci, Co, F_in, T_in, F_out, T_out,
    BLOCK_CO: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < Co

    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # iterate over output time positions
    for to in range(0, T_out):
        t_in = to * 2 + 1
        # iterate over output frequency positions
        for fo in range(0, F_out):
            fi = fo * 2 + 1
            # base input index for this (n, fi, t_in)
            base = pid_n * (Ci * F_in * T_in) + fi * (Ci * T_in) + t_in * Ci
            # loop over input channels
            for ci in range(0, Ci):
                # load input value for channel ci
                x_val = tl.load(x_ptr + base + ci * (F_in * T_in), mask=True, other=0.0).to(tl.float32)

                # accumulate over 3x3 kernel
                for kh in range(0, 3):
                    for kw in range(0, 3):
                        fi_k = fi + kh - 1
                        ti_k = t_in + kw - 1
                        in_bounds = (0 <= fi_k < F_in) and (0 <= ti_k < T_in)
                        if in_bounds:
                            base_k = pid_n * (Ci * F_in * T_in) + fi_k * (Ci * T_in) + ti_k * Ci
                            xk = tl.load(x_ptr + base_k + ci * (F_in * T_in), mask=True, other=0.0).to(tl.float32)
                            w_idx = co_offsets * (Ci * 3 * 3) + ci * (3 * 3) + (kh * 3 + kw)
                            w_vals = tl.load(w_ptr + w_idx, mask=mask_co, other=0.0).to(tl.float32)
                            acc += xk * w_vals

    # add bias
    b_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
    acc += b_vals

    # GELU tanh approximation
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + 0.044715 * x3)))

    # store result
    y_base = pid_n * (Co * F_out * T_out) + co_offsets * (F_out * T_out) + fo * T_out + to
    tl.store(y_ptr + y_base, gelu.to(tl.bfloat16), mask=mask_co)


@triton.jit
def linear_bmm_kernel(
    x_ptr,            # [N, T, M], bfloat16
    wT_ptr,           # [M, K], bfloat16 (W^T), K=1024, M=3840
    y_ptr,            # [N, T, K], bfloat16
    N, T, M, K,
    BLOCK_CO: tl.constexpr,  # tile size for K
):
    # grid: (N, T, ceil_div(K, BLOCK_CO))
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_co = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < K

    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # compute dot over M
    for m in range(0, M):
        # load x[n, t, m]
        x_val = tl.load(x_ptr + pid_n * (T * M) + pid_t * M + m, mask=True, other=0.0).to(tl.float32)
        # load W^T[m, co_offsets]
        w_vals = tl.load(wT_ptr + m * K + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
        acc += x_val * w_vals

    # store y[n, t, co_offsets]
    tl.store(y_ptr + pid_n * (T * K) + pid_t * K + co_offsets, acc.to(tl.bfloat16), mask=mask_co)


@triton.jit
def scale_embed_kernel(
    y_ptr,            # [N, T, K], bfloat16
    scale,            # float32 scalar
    N, T, K,
):
    # grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # load y
    y_val = tl.load(y_ptr + pid_n * (T * K) + pid_t * K + pid_k)
    # scale
    y_val = y_val.to(tl.float32) * scale
    # store (implicitly casts to y_ptr dtype if needed)
    tl.store(y_ptr + pid_n * (T * K) + pid_t * K + pid_k, y_val)


@triton.jit
def add_pos_emb_kernel(
    y_ptr,            # [N, T, K], bfloat16
    pos_ptr,          # [T, K], bfloat16 (sliced positional embedding)
    N, T, K,
):
    # grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    y_val = tl.load(y_ptr + pid_n * (T * K) + pid_t * K + pid_k)
    pos_val = tl.load(pos_ptr + pid_t * K + pid_k)
    y_val = y_val.to(tl.float32) + pos_val.to(tl.float32)
    tl.store(y_ptr + pid_n * (T * K) + pid_t * K + pid_k, y_val)


# =========================
# ModelNew
# =========================

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features,        # [N, 1, 80, T], bfloat16
        conv2d1_weight,        # [384, 1, 3, 3], bfloat16
        conv2d1_bias,          # [384], bfloat16
        conv2d2_weight,        # [384, 384, 3, 3], bfloat16
        conv2d2_bias,          # [384], bfloat16
        conv2d3_weight,        # [384, 384, 3, 3], bfloat16
        conv2d3_bias,          # [384], bfloat16
        conv_out_weight,       # [1024, 3840], bfloat16 (PyTorch F.linear weight is [out_features, in_features])
        positional_embedding,  # [max_source_positions, 1024], bfloat16
        embed_scale,           # float (e.g., 32.0)
    ):
        device = input_features.device
        dtype = input_features.dtype
        N = input_features.shape[0]
        F_in = 80
        T_in = input_features.shape[-1]
        Co = 384

        # Ensure contiguity
        input_features = input_features.contiguous()
        conv2d1_weight = conv2d1_weight.contiguous()
        conv2d1_bias = conv2d1_bias.contiguous()
        conv2d2_weight = conv2d2_weight.contiguous()
        conv2d2_bias = conv2d2_bias.contiguous()
        conv2d3_weight = conv2d3_weight.contiguous()
        conv2d3_bias = conv2d3_bias.contiguous()
        conv_out_weight = conv_out_weight.contiguous()

        # conv1: Ci=1, Co=384, 3x3, stride=2, padding=1
        T_out1 = (T_in - 3) // 2 + 1
        y1 = torch.empty((N, Co, F_in, T_out1), dtype=dtype, device=device)
        BLOCK_CO = 128
        grid1 = (N, (Co + BLOCK_CO - 1) // BLOCK_CO)
        conv_ci1_stride2_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            N, F_in, T_in, Co, T_out1,
            BLOCK_CO=BLOCK_CO,
            num_warps=4, num_stages=2
        )

        # conv2: Ci=384, Co=384, 3x3, stride=2, padding=1
        F_out2 = (F_in - 1) // 2 + 1  # 40
        T_in2 = T_out1
        T_out2 = (T_in2 - 3) // 2 + 1
        y2 = torch.empty((N, Co, F_out2, T_out2), dtype=dtype, device=device)
        grid2 = (N, (Co + BLOCK_CO - 1) // BLOCK_CO)
        conv_stride2_bias_gelu_3d_kernel[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            N, Co, Co, F_in, T_in2, F_out2, T_out2,
            BLOCK_CO=BLOCK_CO,
            num_warps=4, num_stages=2
        )

        # conv3: Ci=384, Co=384, 3x3, stride=2, padding=1
        F_out3 = (F_out2 - 1) // 2 + 1  # 20
        T_in3 = T_out2
        T_out3 = (T_in3 - 3) // 2 + 1
        y3 = torch.empty((N, Co, F_out3, T_out3), dtype=dtype, device=device)
        grid3 = (N, (Co + BLOCK_CO - 1) // BLOCK_CO)
        conv_stride2_bias_gelu_3d_kernel[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            N, Co, Co, F_out2, T_in3, F_out3, T_out3,
            BLOCK_CO=BLOCK_CO,
            num_warps=4, num_stages=2
        )

        # Permute: (N, Co, F_out3, T_out3) -> (N, T_out3, Co*F_out3)
        K_linear = Co * F_out3  # 7680
        x_perm = y3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, K_linear)

        # Linear projection: y = x @ conv_out_weight^T, conv_out_weight: [1024, 3840] (out_features=1024, in_features=3840)
        # Prepare W^T: [M=3840, K=1024]
        W_T = conv_out_weight.permute(1, 0).contiguous()  # [3840, 1024]
        y_linear = torch.empty((N, T_out3, 1024), dtype=dtype, device=device)
        BLOCK_CO = 128
        grid_linear = (N, T_out3, (1024 + BLOCK_CO - 1) // BLOCK_CO)
        linear_bmm_kernel[grid_linear](
            x_perm, W_T, y_linear,
            N, T_out3, K_linear, 1024,
            BLOCK_CO=BLOCK_CO,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale (e.g., 32.0)
        scale = float(embed_scale)
        y_scaled = y_linear
        grid_scale = (N, T_out3, 1024)
        scale_embed_kernel[grid_scale](
            y_scaled, scale,
            N, T_out3, 1024,
            num_warps=4, num_stages=2
        )

        # Add positional embedding: pos_emb [T_out3, 1024], sliced from positional_embedding[:T_out3, :]
        pos_emb = positional_embedding[:T_out3, :].contiguous()
        y_final = torch.empty_like(y_scaled, dtype=dtype, device=device)
        grid_pos = (N, T_out3, 1024)
        add_pos_emb_kernel[grid_pos](
            y_scaled, pos_emb,
            N, T_out3, 1024,
            num_warps=4, num_stages=2
        )
        y_final = y_scaled

        return y_final


def run(*args):
    return ModelNew()(*args)
