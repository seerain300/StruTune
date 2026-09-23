import math
import triton
import triton.language as tl


@triton.jit
def conv_gelu_inplace_bf16(x_ptr, w_ptr, b_ptr, y_ptr,
                            B, H_in, W_in, C_in, C_out,
                            K_h, K_w, stride, pad,
                            BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr):
    # Each program computes one output element y[b, oc, oh, ow]
    b = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    oc = tl.program_id(3)

    # Bounds
    if (b >= B) or (oh >= H_in) or (ow >= W_in) or (oc >= C_out):
        return

    acc = tl.zeros((), dtype=tl.bfloat16)

    # 3x3 kernel, stride=2, padding=1
    for ky in range(K_h):
        for kx in range(K_w):
            ih = oh * stride + ky - pad
            iw = ow * stride + kx - pad
            valid = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
            for ic in range(C_in):
                # Linearized input offset: ((b*C_in + ic)*H_in + ih)*W_in + iw
                x_off = ((b * C_in + ic) * H_in + ih) * W_in + iw
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                # Weight offset: [C_out, C_in, K_h, K_w] contiguous
                w_off = oc * (C_in * (K_h * K_w)) + ic * (K_h * K_w) + ky * K_w + kx
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    # Bias
    b_val = tl.load(b_ptr + oc)
    acc = acc + b_val

    # GELU tanh approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    inner = c * (acc + 0.044715 * x3)
    t = tl.tanh(inner)
    y_val = 0.5 * acc * (1.0 + t)

    # Store y[b, oc, oh, ow]
    # y layout: [B, C_out, H_in, W_in] (note: H_out should be used; here we reuse H_in for indexing simplicity)
    # We should use actual output sizes; for conv_gelu_inplace_bf16, we allocate y with correct H_out and W_out.
    # To keep it simple, we index y with oh, ow in input space but store at output coordinates. We'll pass H_out and W_out via shape, not here.
    # Instead, compute H_out and W_out on host and pass into forward, and allocate y accordingly. Here, we assume y is allocated as output.
    # Since Triton cannot directly receive H_out/W_out, we pass them via B, H_in, W_in, and let host allocate correct y. We'll treat oh, ow as output coords.
    # In practice, we allocate y with correct H_out and W_out on host. The kernel then writes to y[b, oc, oh, ow].
    # y layout: [B, C_out, H_out, W_out]
    y_off = ((b * C_out) + oc) * (H_out * W_out) + oh * W_out + ow
    tl.store(y_ptr + y_off, y_val)


@triton.jit
def linear_gemm_gelu_bf16(x_ptr, w_ptr, y_ptr,
                          B, T, K, N,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute y[b, t, n] = GELU(sum_k x[b, t, k] * w[n, k])
    # x_ptr: [B*T, K]
    # w_ptr: [N, K]
    # y_ptr: [B*T, N]
    pid_m = tl.program_id(0)  # over B*T
    pid_n = tl.program_id(1)  # over N

    b = pid_m // T
    t = pid_m % T
    n = pid_n

    if (b >= B) or (t >= T) or (n >= N):
        return

    acc = tl.zeros((), dtype=tl.bfloat16)

    # Tile over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        x_off = pid_m * K + k_offsets
        x_vec = tl.load(x_ptr + x_off, mask=mask_k, other=0.0)  # [BLOCK_K]

        w_off = n * K + k_offsets
        w_vec = tl.load(w_ptr + w_off, mask=mask_k, other=0.0)  # [BLOCK_K]

        acc += tl.sum(x_vec * w_vec, axis=0)

    # GELU
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    inner = c * (acc + 0.044715 * x3)
    t = tl.tanh(inner)
    y_val = 0.5 * acc * (1.0 + t)

    y_off = pid_m * N + n
    tl.store(y_ptr + y_off, y_val)


@triton.jit
def add_pos_embed_bf16(y_ptr, pos_ptr, B, T, N):
    # y_ptr: [B*T*N] flattened view as [B, T, N] by using strides
    # Triton does not have 3D indexing; we pass B, T, N and compute offsets.
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_n = tl.program_id(2)

    if (pid_b >= B) or (pid_t >= T) or (pid_n >= N):
        return

    base = (pid_b * T + pid_t) * N + pid_n
    y_val = tl.load(y_ptr + base)
    pos_val = tl.load(pos_ptr + (pid_t * N + pid_n))
    y_val = y_val + pos_val
    tl.store(y_ptr + base, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features,
                conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv3_bias,
                conv_out_weight,
                positional_embedding,
                embed_scale):
        # Ensure CUDA and bfloat16
        assert input_features.is_cuda and conv2d1_weight.is_cuda and conv_out_weight.is_cuda and positional_embedding.is_cuda
        assert input_features.dtype == torch.bfloat16 and conv_out_weight.dtype == torch.bfloat16 and positional_embedding.dtype == torch.bfloat16

        # Input: [B, 1, H_in=80, W_in=time_dim]
        B, C_in, H_in, W_in = input_features.shape
        assert C_in == 1, "Input must have C_in=1"

        # Conv1: in_channels=1 -> out_channels=384
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H_in + 2 * 1 - 3) // 2 + 1  # padding=1, kernel=3, stride=2
        W_out1 = (W_in + 2 * 1 - 3) // 2 + 1

        x1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.bfloat16, device=input_features.device)
        grid1 = (B, H_out1, W_out1, C_out1)
        conv_gelu_inplace_bf16[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, H_in, W_in, 1, C_out1, 3, 3, 2, 1,
            BLOCK_W=1, BLOCK_C=1
        )

        # Conv2: in_channels=384 -> out_channels=384
        C_in2 = C_out1
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H_out1 + 2 * 1 - 3) // 2 + 1
        W_out2 = (W_out1 + 2 * 1 - 3) // 2 + 1

        x2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.bfloat16, device=input_features.device)
        grid2 = (B, H_out2, W_out2, C_out2)
        conv_gelu_inplace_bf16[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, H_out1, W_out1, C_in2, C_out2, 3, 3, 2, 1,
            BLOCK_W=1, BLOCK_C=1
        )

        # Conv3: in_channels=384 -> out_channels=384
        C_in3 = C_out2
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = (H_out2 + 2 * 1 - 3) // 2 + 1
        W_out3 = (W_out2 + 2 * 1 - 3) // 2 + 1

        x3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.bfloat16, device=input_features.device)
        grid3 = (B, H_out3, W_out3, C_out3)
        conv_gelu_inplace_bf16[grid3](
            x2, conv2d3_weight, conv3_bias, x3,
            B, H_out2, W_out2, C_in3, C_out3, 3, 3, 2, 1,
            BLOCK_W=1, BLOCK_C=1
        )

        # Reshape to [B, T, K] where T=W_out3, K=C_out3
        T = W_out3
        K = C_out3

        # Flatten x3 to [B, T, K] by viewing as [B, H_out3, W_out3, C_out3] and permute to [B, W_out3, C_out3] then view
        # Easiest: x3_reshaped = x3.permute(0, 3, 1, 2).contiguous().view(B, W_out3, C_out3)
        # But to keep it simple, we can directly view: since x3 shape is [B, C_out3, H_out3, W_out3], we need to permute.
        x3_perm = x3.permute(0, 2, 3, 1).contiguous()  # [B, H_out3, W_out3, C_out3]
        x3_reshaped = x3_perm.view(B, W_out3, C_out3).contiguous()  # [B, T, K]

        # Linear projection: y[B, T, N] = GELU(x3[B, T, K] @ conv_out_weight^T [N, K])
        N = conv_out_weight.shape[0]  # d_model = 1024 (expected)
        y = torch.empty((B, T, N), dtype=torch.bfloat16, device=input_features.device)

        # Launch linear


def run(*args):
    return ModelNew()(*args)
