import math

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: conv1 specialized (Ci=1, Co=384, 3x3, stride=2, padding=1), fused bias+gelu
@triton.jit
def conv_ci1_stride2_bias_gelu_kernel(
    input_ptr,         # *fp16 (or bf16), [N, 1, F_in, T_in]
    w_ptr,             # *fp16, [384, 1, 3, 3]
    b_ptr,             # *fp16, [384]
    output_ptr,        # *fp16, [N, 384, F_in, T_out]
    N, Ci, Co, F_in, T_in, F_out, T_out,
    BLOCK_CO: tl.constexpr,
):
    # Grid: (N, ceil(384/BLOCK_CO))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)

    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < Co

    # Initialize accumulator for this (n, co_group)
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # For each input time index t_out, compute corresponding t_in = 2*t_out - 1
    # Loop over t_out from 0 to T_out-1
    for t_out in range(0, T_out):
        t_in = 2 * t_out
        # Loop over F_out channels
        for f_out in range(0, F_out):
            # Accumulate over 3x3 window and input channel (Ci=1)
            # conv output at (n, co, f_out, t_out) = sum over kernel and input channel
            # input index: in_f = f_out*2 + di, in_t = t_in + dj; valid if in_f in [0, F_in), in_t in [0, T_in)
            for di in range(0, 3):
                for dj in range(0, 3):
                    in_f = f_out * 2 + di
                    in_t = t_in + dj
                    valid = (in_f >= 0) & (in_f < F_in) & (in_t >= 0) & (in_t < T_in)
                    # input element: [n, 0, in_f, in_t]
                    in_offset = pid_n * (Ci * F_in * T_in) + 0 * (F_in * T_in) + in_f * T_in + in_t
                    x_val = tl.load(input_ptr + in_offset, mask=valid, other=0.0)
                    # weight vector for this co group: w[co_offsets, 0, di, dj]
                    w_base = (co_offsets * (Ci * 3 * 3)) + (0 * (3 * 3)) + (di * 3) + dj
                    w_vals = tl.load(w_ptr + w_base, mask=mask_co, other=0.0)
                    acc += x_val * w_vals

    # Add bias
    b_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0)
    acc += b_vals

    # GELU tanh approximation: y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    x = acc
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    gelu = 0.5 * x * (1.0 + tl.math.tanh(c0 * (x + 0.044715 * x3)))

    # Store result to output: [N, Co, F_in, T_out]
    out_base = pid_n * (Co * F_in * T_out) + co_offsets * (F_in * T_out) + (f_out) * T_out + t_out
    tl.store(output_ptr + out_base, gelu, mask=mask_co)


# Kernel 2: generic conv stride=2, padding=1, fused bias+gelu
@triton.jit
def conv_stride2_bias_gelu_kernel(
    input_ptr,         # *fp16 (or bf16), [N, Ci, F_in, T_in]
    w_ptr,             # *fp16, [Co, Ci, 3, 3]
    b_ptr,             # *fp16, [Co]
    output_ptr,        # *fp16, [N, Co, F_out, T_out]
    N, Ci, Co, F_in, T_in, F_out, T_out,
    BLOCK_CO: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < Co

    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    for t_out in range(0, T_out):
        t_in = 2 * t_out
        for f_out in range(0, F_out):
            for ci in range(0, Ci):
                for di in range(0, 3):
                    for dj in range(0, 3):
                        in_f = f_out * 2 + di
                        in_t = t_in + dj
                        valid = (in_f >= 0) & (in_f < F_in) & (in_t >= 0) & (in_t < T_in)
                        in_offset = pid_n * (Ci * F_in * T_in) + ci * (F_in * T_in) + in_f * T_in + in_t
                        x_val = tl.load(input_ptr + in_offset, mask=valid, other=0.0)
                        # w[co, ci, di, dj]
                        w_base = co_offsets * (Ci * 3 * 3) + ci * (3 * 3) + di * 3 + dj
                        w_vals = tl.load(w_ptr + w_base, mask=mask_co, other=0.0)
                        acc += x_val * w_vals

        b_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0)
        acc += b_vals

        # GELU
        x = acc
        c0 = 0.7978845608028654
        x3 = x * x * x
        gelu = 0.5 * x * (1.0 + tl.math.tanh(c0 * (x + 0.044715 * x3)))

        out_base = pid_n * (Co * F_out * T_out) + co_offsets * (F_out * T_out) + f_out * T_out + t_out
        tl.store(output_ptr + out_base, gelu, mask=mask_co)


# Kernel 3: Linear projection (batched GEMV), output [N, T_out3, K]
@triton.jit
def linear_bmm_kernel(
    x_ptr,      # *fp16, [N, T_out3, M]
    w_ptr,      # *fp16, [M, K] (we pass conv_out_weight_T here)
    y_ptr,      # *fp16, [N, T_out3, K]
    N, T, M, K,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # For each input feature j in M, accumulate dot product
    for j in range(0, M):
        x_val = tl.load(x_ptr + pid_n * (T * M) + pid_t * M + j, mask=True, other=0.0)  # x[n, t, j]
        w_vals = tl.load(w_ptr + j * K + k_offsets, mask=mask_k, other=0.0)            # w[j, k_offsets]
        acc += x_val * w_vals

    # Store result
    y_out_base = pid_n * (T * K) + pid_t * K + k_offsets
    tl.store(y_ptr + y_out_base, acc, mask=mask_k)


# Kernel 4: scale embeddings by scalar (embed_scale)
@triton.jit
def scale_embed_kernel(
    y_ptr,      # *fp16, [N, T_out3, K]
    scale,      # scalar float
    N, T, K,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Each program handles a tile of K
    k_offsets = pid_k * 64 + tl.arange(0, 64)
    mask_k = k_offsets < K

    y_in = tl.load(y_ptr + pid_n * (T * K) + pid_t * K + k_offsets, mask=mask_k, other=0.0)
    y_out = y_in * scale
    tl.store(y_ptr + pid_n * (T * K) + pid_t * K + k_offsets, y_out, mask=mask_k)


# Kernel 5: add positional embedding [N, T_out3, K] += pos_emb[:T_out3, :].unsqueeze(0)
@triton.jit
def add_pos_emb_kernel(
    y_ptr,          # *fp16, [N, T_out3, K]
    pos_ptr,        # *fp16, [T_out3, K]
    N, T, K,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_offsets = pid_k * 64 + tl.arange(0, 64)
    mask_k = k_offsets < K

    y_vals = tl.load(y_ptr + pid_n * (T * K) + pid_t * K + k_offsets, mask=mask_k, other=0.0)
    pos_vals = tl.load(pos_ptr + pid_t * K + k_offsets, mask=mask_k, other=0.0)
    y_vals += pos_vals
    tl.store(y_ptr + pid_n * (T * K) + pid_t * K + k_offsets, y_vals, mask=mask_k)


# Entry point model
class ModelNew(nn.Module):
    def __init__(self, axes_and_scalars, device):
        super().__init__()
        # We don't need to keep axes; we will accept the same inputs as original run's get_inputs.
        self.device = device

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
        # Ensure device/dtype and contiguity
        N = input_features.shape[0]
        T_in = input_features.shape[-1]
        # Stage 1: conv1 (Ci=1 -> Co=384), stride=2, padding=1, fused bias+gelu
        F_in = 80
        T_out1 = (T_in - 3) // 2 + 1
        y1 = torch.empty((N, 384, F_in, T_out1), dtype=torch.bfloat16, device=self.device)
        grid1 = (N, triton.cdiv(384, 128))
        conv_ci1_stride2_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            N, 1, 384, F_in, T_in, F_in, T_out1,
            BLOCK_CO=128,
            num_warps=4, num_stages=2,
        )

        # Stage 2: conv2 (Ci=384 -> Co=384), stride=2, padding=1, fused bias+gelu
        F_out1 = (F_in - 1) // 2 + 1  # 40
        T_in2 = T_out1
        T_out2 = (T_in2 - 3) // 2 + 1
        y2 = torch.empty((N, 384, F_out1, T_out2), dtype=torch.bfloat16, device=self.device)
        grid2 = (N, triton.cdiv(384, 128))
        conv_stride2_bias_gelu_kernel[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            N, 384, 384, F_in, T_in2, F_out1, T_out2,
            BLOCK_CO=128,
            num_warps=4, num_stages=2,
        )

        # Stage 3: conv3 (Ci=384 -> Co=384), stride=2, padding=1, fused bias+gelu
        F_out2 = (F_out1 - 1) // 2 + 1  # 20
        T_in3 = T_out2
        T_out3 = (T_in3 - 3) // 2 + 1
        y3 = torch.empty((N, 384, F_out2, T_out3), dtype=torch.bfloat16, device=self.device)
        grid3 = (N, triton.cdiv(384, 128))
        conv_stride2_bias_gelu_kernel[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            N, 384, 384, F_out1, T_in3, F_out2, T_out3,
            BLOCK_CO=128,
            num_warps=4, num_stages=2,
        )

        # Reshape: (N, 384, 20, T_out3) -> (N, T_out3, 384*20)
        M = 384 * F_out2  # 7680
        x = y3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, M)

        # Linear projection: Y[n, t, k] = sum_j X[n, t, j] * W[j, k], W shape [K=1024, M=3840] provided
        # We use W_T = W^T of shape [M, K] for simpler indexing; create a permuted view (no heavy compute)
        W_T = conv_out_weight.permute(1, 0).contiguous()  # [3840, 1024]
        y = torch.empty((N, T_out3, 1024), dtype=torch.bfloat16, device=self.device)
        grid4 = (N, T_out3, triton.cdiv(1024, 64))
        linear_bmm_kernel[grid4](
            x, W_T, y,
            N, T_out3, M, 1024,
            BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Scale embeddings by embed_scale
        grid5 = (N, T_out3, triton.cdiv(1024, 64))
        scale_embed_kernel[grid5](
            y, embed_scale,
            N, T_out3, 1024,
            num_warps=4, num_stages=2,
        )

        # Add positional embedding [:T_out3, :]
        pos = positional_embedding[:T_out3, :].to(torch.bfloat16).to(self.device).contiguous()
        grid6 = (N, T_out3, triton.cdiv(1024, 64))
        add_pos_emb_kernel[grid6](
            y, pos,
            N, T_out3, 1024,
            num_warps=4, num_stages=2,
        )

        return y


def run(*args):
    return ModelNew()(*args)
