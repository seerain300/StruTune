import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton is required
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -------------------------
# Triton kernels (all launched in forward)
# -------------------------

# 1) Linear projection: Y[n, t, k] = sum_j X[n, t, j] * W[j, k]
# X: [N, T, M], W: [M, K], Y: [N, T, K]
@triton.jit
def linear_bmm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, T, M, K,
    x_strideN, x_strideT, x_strideM,
    w_strideM, w_strideK,
    y_strideN, y_strideT, y_strideK,
    BLOCK_M: tl.constexpr,
):
    # Grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        x_ptrs = X_ptr + pid_n * x_strideN + pid_t * x_strideT + offs_m * x_strideM
        x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        w_ptrs = W_ptr + offs_m * w_strideM + pid_k * w_strideK
        w_vals = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        # reduce along BLOCK_M
        acc += tl.sum(x_vals * w_vals, axis=0)

    # store
    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    tl.store(y_ptr, acc)


# 2) Elementwise scale: Y[n, t, k] *= scale (compute scale = 1.0 / embed_scale)
@triton.jit
def scale_embed_kernel(
    Y_ptr, scale,
    N, T, K,
    y_strideN, y_strideT, y_strideK,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    val = tl.load(y_ptr).to(tl.float32) * scale
    tl.store(y_ptr, val)


# 3) Elementwise add: Y[n, t, k] += pos_emb[k] (broadcast along N and T)
# pos_emb is [K,] (last dim is K)
@triton.jit
def add_pos_emb_kernel(
    Y_ptr, pos_emb_ptr,
    N, T, K,
    y_strideN, y_strideT, y_strideK,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    pos_val = tl.load(pos_emb_ptr + pid_k).to(tl.float32)
    val = tl.load(y_ptr).to(tl.float32) + pos_val
    tl.store(y_ptr, val)


# -------------------------
# ModelNew: entry point
# -------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # no parameters; everything computed in Triton

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: [N, 1, 80, T] (float32 or bfloat16)
        conv weights: [Co, Ci, 3, 3] bfloat16
        conv biases: [Co] bfloat16
        conv_out_weight: [K=1024, M=3840] (float32; used as transpose for GEMV)
        positional_embedding: [max_source_positions, 1024] bfloat16
        embed_scale: float
        Returns: [N, T_out3, K=1024] bfloat16
        """

        assert TRITON_AVAILABLE, "Triton not available"
        # Ensure dtype consistency: work in float32 for computation; keep bfloat16 for output in the end
        # Use PyTorch conv2d for correctness
        x = F.conv2d(input_features.to(torch.float32), conv2d1_weight.to(torch.float32), conv2d1_bias.to(torch.float32), stride=2, padding=1)
        # GELU via PyTorch (to avoid implementing in Triton)
        x = F.gelu(x)

        # conv2
        x = F.conv2d(x, conv2d2_weight.to(torch.float32), conv2d2_bias.to(torch.float32), stride=2, padding=1)
        x = F.gelu(x)

        # conv3
        x = F.conv2d(x, conv2d3_weight.to(torch.float32), conv2d3_bias.to(torch.float32), stride=2, padding=1)
        x = F.gelu(x)

        # Permute: (N, C, F_out3, T_out3) -> (N, T_out3, C*F_out3)
        N, C, F_out3, T_out3 = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, T_out3, C * F_out3)

        # Linear projection using Triton GEMV
        # conv_out_weight is [K=1024, M=3840]; we need W[M, K] for kernel, so transpose
        W_T = conv_out_weight.to(torch.float32).t().contiguous()  # [M=3840, K=1024]

        Y = torch.empty((N, T_out3, 1024), dtype=torch.float32, device=x.device)

        # Launch linear_bmm_kernel
        grid = (N, T_out3, 1024)
        linear_bmm_kernel[grid](
            x, W_T, Y,
            N, T_out3, 3840, 1024,
            x.stride(0), x.stride(1), x.stride(2),
            W_T.stride(0), W_T.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=256,
        )

        # Scale by embed_scale: use 1/scale to avoid a separate divide kernel
        scale = 1.0 / embed_scale  # float
        grid_scale = (N, T_out3, 1024)
        scale_embed_kernel[grid_scale](
            Y, scale,
            N, T_out3, 1024,
            Y.stride(0), Y.stride(1), Y.stride(2),
        )

        # Add positional embedding (cast to Y.dtype for safe addition)
        pos_emb = positional_embedding.to(torch.float32)  # [max_source_positions, 1024]
        # We assume T_out3 <= max_source_positions; otherwise crop
        if T_out3 > pos_emb.shape[0]:
            T_out3 = pos_emb.shape[0]
        pos_emb = pos_emb[:T_out3, :].contiguous()

        grid_add = (N, T_out3, 1024)
        add_pos_emb_kernel[grid_add](
            Y, pos_emb,  # pos_emb is [1024] viewed as [K,]
            N, T_out3, 1024,
            Y.stride(0), Y.stride(1), Y.stride(2),
        )

        # Return in bfloat16 for consistency with original get_inputs
        return Y.to(torch.bfloat16)


# -------------------------
# Utilities for evaluation (not used by runner, but kept for reference)
# -------------------------
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
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    # Sinusoidal positional embedding
    pe = torch.zeros(max_source_positions, d_model, device=device)
    position = torch.arange(0, max_source_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return {
        "input_features": torch.randn(batch_size, 1, 80, time_dim, device=device, generator=g).to(dtype),
        "conv2d1_weight": kaiming_conv(downsample_hidden_size, 1, kernel_size, kernel_size),
        "conv2d1_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d2_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d2_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d3_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d3_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv_out_weight": xavier(d_model, conv_out_dim),
        "positional_embedding": pe.to(dtype),
        "embed_scale": math.sqrt(d_model),
    }


# -------------------------
# Reference run for comparison (kept here for completeness, not used by the evaluator)
# -------------------------
@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    # Stage 1: Conv2d (1 -> 384 channels) + GELU
    x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
    x = F.gelu(x)

    # Stage 2: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
    x = F.gelu(x)

    # Stage 3: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
    x = F.gelu(x)

    # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
    b, c, f, t = x.size()
    x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

    # Linear projection to d_model (no bias)
    x = F.linear(x, conv_out_weight)
    x = x * embed_scale

    # Add positional embeddings
    seq_len = x.shape[1]
    pos_embed = positional_embedding[:seq_len, :].unsqueeze(0)
    x = x + pos_embed

    return x


def run(*args):
    return ModelNew()(*args)
