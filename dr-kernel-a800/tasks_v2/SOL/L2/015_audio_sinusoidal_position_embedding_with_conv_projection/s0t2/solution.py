import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton matmul kernel: A[M, K] x B[K, N] -> C[M, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise kernel: Y[B*T, N] = X[B*T, N] * SCALE + POS[B*T, N]
@triton.jit
def scale_add_pos_emb_kernel(
    X_ptr, POS_ptr, Y_ptr,
    L, N, SCALE,
    stride_xm, stride_xn,
    stride_pm, stride_pn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < L
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    pos_ptrs = POS_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    pos = tl.load(pos_ptrs, mask=mask, other=0.0)
    y = x * SCALE + pos
    tl.store(y_ptrs, y, mask=mask)


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time_dim = axes_and_scalars["time_dim"]
    d_model = 1024
    max_source_positions = 1500
    downsample_hidden_size = 384
    conv_out_dim = 3840  # 384 * 10
    kernel_size = 3
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        w = (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in))
        return w.to(dtype)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    # Sinusoidal positional embedding
    pe = torch.zeros(max_source_positions, d_model, device=device)
    position = torch.arange(0, max_source_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    embed_scale = math.sqrt(d_model)

    input_features = torch.randn(batch_size, 1, 80, time_dim, device=device, generator=g).to(dtype)
    conv2d1_weight = kaiming_conv(downsample_hidden_size, 1, kernel_size, kernel_size)  # [384, 1, 3, 3]
    conv2d1_bias = torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype)
    conv2d2_weight = kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size)  # [384, 384, 3, 3]
    conv2d2_bias = torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype)
    conv2d3_weight = kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size)  # [384, 384, 3, 3]
    conv2d3_bias = torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype)
    conv_out_weight = xavier(d_model, conv_out_dim)  # [1024, 3840]
    positional_embedding = pe  # [1500, 1024]
    embed_scale = embed_scale

    return {
        "input_features": input_features,
        "conv2d1_weight": conv2d1_weight,
        "conv2d1_bias": conv2d1_bias,
        "conv2d2_weight": conv2d2_weight,
        "conv2d2_bias": conv2d2_bias,
        "conv2d3_weight": conv2d3_weight,
        "conv2d3_bias": conv2d3_bias,
        "conv_out_weight": conv_out_weight,
        "positional_embedding": positional_embedding,
        "embed_scale": embed_scale,
    }


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Convolution 1: 1 -> 384 channels, stride=2, padding=1
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Convolution 2: 384 -> 384 channels, stride=2, padding=1
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Convolution 3: 384 -> 384 channels, stride=2, padding=1
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection via Triton matmul
        B = b
        T3 = t
        K = c * f  # 384 * 10 = 3840
        N = conv_out_weight.shape[0]  # 1024

        A = x  # [B, T3, K]
        A_2d = A.view(B * T3, K).contiguous()
        B_mat = conv_out_weight.t().contiguous()  # [K, N]

        C_2d = torch.empty((B * T3, N), dtype=torch.float32, device=A_2d.device)

        if TRITON_AVAILABLE:
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 128
            grid = (triton.cdiv(B * T3, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_kernel[grid](
                A_2d, B_mat, C_2d,
                B * T3, N, K,
                A_2d.stride(0), A_2d.stride(1),
                B_mat.stride(0), B_mat.stride(1),
                C_2d.stride(0), C_2d.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
        else:
            C_2d = A_2d.matmul(conv_out_weight)  # [B*T3, 1024]

        # Reshape back to [B, T3, 1024]
        C = C_2d.view(B, T3, N)

        # Scale by embed_scale
        scale = float(embed_scale)
        Y = C * scale  # float32

        # Add positional embedding (first T3 rows)
        pos_emb = positional_embedding[:T3, :].to(Y.dtype)

        # If Triton available, do final add in PyTorch (small op); otherwise Y already scaled in Triton path above.
        Y = Y + pos_emb

        return Y


def run(*args):
    return ModelNew()(*args)
