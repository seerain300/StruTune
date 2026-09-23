import math
import torch
import torch.nn as nn

# Triton is required
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -------------------------
# Triton kernels
# -------------------------

# 1) Batched GEMV: Y[n, t, k] = sum_j X[n, t, j] * W[j, k]
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

    # Iterate over M in chunks of BLOCK_M
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        # X[n, t, j] for all j in offs_m
        x_ptrs = X_ptr + pid_n * x_strideN + pid_t * x_strideT + offs_m * x_strideM
        x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        # W[j, k] for all j in offs_m (M dimension maps to j)
        w_ptrs = W_ptr + offs_m * w_strideM + pid_k * w_strideK
        w_vals = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        # Accumulate dot product for this k
        acc += tl.sum(x_vals * w_vals, axis=0)

    # Store result to Y[n, t, k]
    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    tl.store(y_ptr, acc)


# 2) Elementwise scaling: Y_scaled = Y * scale (scale = 1.0 / embed_scale)
@triton.jit
def scale_embed_kernel(
    Y_ptr, S_ptr, Out_ptr,
    N, T, K,
    y_strideN, y_strideT, y_strideK,
    s_strideT, s_strideK,
    out_strideN, out_strideT, out_strideK,
    BLOCK: tl.constexpr,
):
    # Grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    s_ptr = S_ptr + pid_t * s_strideT + pid_k * s_strideK
    out_ptr = Out_ptr + pid_n * out_strideN + pid_t * out_strideT + pid_k * out_strideK

    y_val = tl.load(y_ptr).to(tl.float32)
    s_val = tl.load(s_ptr).to(tl.float32)
    out_val = y_val * s_val
    tl.store(out_ptr, out_val)


# 3) Add positional embedding: Out = Out + pos_emb
# pos_emb is [T, K], we broadcast over N dimension
@triton.jit
def add_pos_emb_kernel(
    Out_ptr, Pos_ptr, Out2_ptr,
    N, T, K,
    out_strideN, out_strideT, out_strideK,
    pos_strideT, pos_strideK,
    out2_strideN, out2_strideT, out2_strideK,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    out_ptr = Out_ptr + pid_n * out_strideN + pid_t * out_strideT + pid_k * out_strideK
    pos_ptr = Pos_ptr + pid_t * pos_strideT + pid_k * pos_strideK
    out2_ptr = Out2_ptr + pid_n * out2_strideN + pid_t * out2_strideT + pid_k * out2_strideK

    out_val = tl.load(out_ptr).to(tl.float32)
    pos_val = tl.load(pos_ptr).to(tl.float32)
    tl.store(out2_ptr, out_val + pos_val)


# -------------------------
# ModelNew: forward using Triton for post-conv stages
# -------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        input_features,  # [N, 1, 80, T], dtype: bfloat16
        conv2d1_weight,  # [Co=384, Ci=1, 3, 3], dtype: bfloat16
        conv2d1_bias,    # [Co=384], dtype: bfloat16
        conv2d2_weight,  # [Co=384, Ci=384, 3, 3], dtype: bfloat16
        conv2d2_bias,    # [Co=384], dtype: bfloat16
        conv2d3_weight,  # [Co=384, Ci=384, 3, 3], dtype: bfloat16
        conv2d3_bias,    # [Co=384], dtype: bfloat16
        conv_out_weight, # [d_model=1024, conv_out_dim=3840], dtype: bfloat16
        positional_embedding,  # [max_source_positions, d_model], dtype: bfloat16
        embed_scale,             # float
    ):
        # Ensure CUDA tensors
        assert input_features.is_cuda, "All tensors must be on CUDA for Triton execution"
        device = input_features.device

        # 1) Conv2d layers using torch to match original exactly
        # conv1: in_channels=1 -> out_channels=384
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        # conv2: in_channels=384 -> out_channels=384
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        # conv3: in_channels=384 -> out_channels=384
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)

        # 2) Permute and flatten to [N, T, C*F]
        N, C, F, T = x.shape  # C=384, F=10
        x = x.permute(0, 3, 1, 2).contiguous().view(N, T, C * F)

        # Ensure dtype for kernels
        x = x.to(torch.bfloat16).contiguous()

        N, T, M = x.shape  # M = C * F = 3840
        K = 1024  # d_model

        # Allocate output for linear
        Y = torch.empty((N, T, K), device=device, dtype=torch.bfloat16)

        # Prepare W: conv_out_weight is [K=1024, M=3840]; we need W[j, k] for j in [0..M-1], k in [0..K-1]
        # So W[j, k] = conv_out_weight[k, j]. Use permute to create [M, K].
        W = conv_out_weight.permute(0, 1).contiguous()  # [M=3840, K=1024], bfloat16

        # Launch linear_bmm_kernel: grid = (N, T, K)
        grid = (N, T, K)
        BLOCK_M = 128
        linear_bmm_kernel[grid](
            x, W, Y,
            N, T, M, K,
            x.stride(0), x.stride(1), x.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2,
        )

        # 3) Scale by embed_scale: scale = 1.0 / embed_scale
        scale = torch.tensor(1.0 / float(embed_scale), device=device, dtype=torch.bfloat16)  # scalar tensor
        Y_scaled = torch.empty_like(Y)

        # Launch scale_embed_kernel: grid = (N, T, K)
        scale_embed_kernel[grid](
            Y, scale, Y_scaled,
            N, T, K,
            Y.stride(0), Y.stride(1), Y.stride(2),
            scale.stride(0), scale.stride(1),  # scale is [T, K] but stride1 used for K
            Y_scaled.stride(0), Y_scaled.stride(1), Y_scaled.stride(2),
            BLOCK=1,
            num_warps=1, num_stages=1,
        )

        # 4) Add positional embedding: pos_emb is [max_source_positions, d_model] = [T_max, K]
        # Only need first T rows for our current T_out
        pos_emb = positional_embedding.to(torch.bfloat16)[:T, :].contiguous()  # [T, K], bfloat16
        Out = torch.empty_like(Y_scaled)

        # Launch add_pos_emb_kernel: grid = (N, T, K)
        add_pos_emb_kernel[grid](
            Y_scaled, pos_emb, Out,
            N, T, K,
            Y_scaled.stride(0), Y_scaled.stride(1), Y_scaled.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK=1,
            num_warps=1, num_stages=1,
        )

        return Out


def run(*args):
    return ModelNew()(*args)
