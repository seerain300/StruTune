import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton import and availability flag
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


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
        # These conv weights are unused in forward (we keep them for API compatibility).
        fan_in = in_c * kh * kw
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    # Sinusoidal positional embedding
    pe = torch.zeros(max_source_positions, d_model, device=device)
    position = torch.arange(0, max_source_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return {
        "input_features": torch.randn(batch_size, 1, 80, time_dim, device=device, generator=g).to(dtype),
        # Conv weights (unused in forward but kept for consistency)
        "conv2d1_weight": kaiming_conv(downsample_hidden_size, 1, kernel_size, kernel_size),
        "conv2d1_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d2_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d2_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d3_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d3_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        # Linear projection weight (note: original code names this conv_out_weight, but it is a projection weight of shape [1024, 3840])
        "conv_out_weight": xavier(d_model, conv_out_dim),  # [1024, 3840]
        # Sinusoidal positional embedding
        "positional_embedding": pe.to(dtype),
        # embed_scale = sqrt(d_model)
        "embed_scale": math.sqrt(d_model),
    }


# Triton kernels (only defined if Triton is available)
if TRITON_AVAILABLE:
    @triton.jit
    def linear_proj_kernel(x, w_t, y_flat,
                            M, K, N,
                            stride_x_m, stride_x_k,
                            stride_w_k, stride_w_n,
                            stride_y_m, stride_y_n,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # Each program computes a tile of [BLOCK_N x BLOCK_K] for a given row m in x
        # x: [M, K], w_t: [K, N], y_flat: [M, N]
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m = pid_m
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + k_offsets
            x_ptrs = x + m * stride_x_m + k * stride_x_k  # [BLOCK_K]
            w_ptrs = w_t + k * stride_w_k + n_offsets * stride_w_n  # [BLOCK_N, BLOCK_K]
            x_vals = tl.load(x_ptrs, mask=k < K, other=0.0)  # [BLOCK_K]
            w_vals = tl.load(w_ptrs, mask=(n_offsets < N) & (k < K), other=0.0)  # [BLOCK_N, BLOCK_K]
            acc += tl.sum(w_vals.to(tl.float32) * x_vals[:, None].to(tl.float32), axis=0)

        y_ptrs = y_flat + m * stride_y_m + n_offsets * stride_y_n
        tl.store(y_ptrs, acc, mask=n_offsets < N)


    @triton.jit
    def scale_elementwise_kernel(y_flat, out_flat, N_elems, scale, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_elems
        vals = tl.load(y_flat + offsets, mask=mask, other=0.0)
        vals = vals * scale
        tl.store(out_flat + offsets, vals, mask=mask)


    @triton.jit
    def add_pos_emb_kernel(y_flat, pos_flat, N_elems, S, N, BLOCK: tl.constexpr):
        # y_flat: [B*S*N], pos_flat: [S*N]
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_elems
        b_idx = offsets // (S * N)
        rem = offsets % (S * N)
        t_idx = rem // N
        n_idx = rem % N
        y_ptrs = y_flat + offsets
        pos_ptrs = pos_flat + t_idx * N + n_idx
        vals = tl.load(y_ptrs, mask=mask, other=0.0)
        add_vals = tl.load(pos_ptrs, mask=mask, other=0.0)
        vals = vals + add_vals
        tl.store(y_ptrs, vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args correspond to get_inputs fields in order:
        # input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        input_features = args[0]  # [B, 1, 80, T], bfloat16
        # We will use PyTorch conv2d + GELU for correctness (cuDNN optimized)
        # Conv1: in_channels=1, out_channels=384
        x = F.conv2d(input_features, args[1], args[2], stride=2, padding=1)  # [B, 384, 40, T/2]
        x = F.gelu(x)

        # Conv2: in_channels=384, out_channels=384
        x = F.conv2d(x, args[3], args[4], stride=2, padding=1)  # [B, 384, 20, T/4]
        x = F.gelu(x)

        # Conv3: in_channels=384, out_channels=384
        x = F.conv2d(x, args[5], args[6], stride=2, padding=1)  # [B, 384, 10, T/8]
        x = F.gelu(x)

        # Reshape to [B, S, K], where K = 384 * 10 = 3840
        B = x.shape[0]
        C = x.shape[1]  # 384
        H = x.shape[2]  # 10
        S = x.shape[3]  # time_after_conv = T // 8
        K = C * H  # 3840
        x = x.permute(0, 3, 1, 2).contiguous().view(B, S, K)  # [B, S, 3840]

        # Triton: Linear projection x @ conv_out_weight.T, where conv_out_weight is [1024, 3840]
        conv_out_weight = args[7]  # [1024, 3840]
        # We need w_t = conv_out_weight.T -> [3840, 1024]
        w_t = conv_out_weight.t().contiguous()  # [3840, 1024]

        M = B * S
        x_rowwise = x.reshape(M, K).contiguous()  # [M, K]

        # Allocate flat output [M, N]
        N = w_t.shape[1]  # 1024
        y_flat = torch.empty((M, N), device=x_rowwise.device, dtype=torch.float32)  # accumulate in fp32

        # Launch Triton matmul kernel
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (M, triton.cdiv(N, BLOCK_N))
        linear_proj_kernel[grid](
            x_rowwise, w_t, y_flat,
            M, K, N,
            x_rowwise.stride(0), x_rowwise.stride(1),
            w_t.stride(0), w_t.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Scale elementwise by embed_scale
        y_flat_scaled = torch.empty_like(y_flat)
        N_elems = y_flat.numel()
        BLOCK_SCALE = 1024
        grid_scale = (triton.cdiv(N_elems, BLOCK_SCALE),)
        scale_elementwise_kernel[grid_scale](y_flat, y_flat_scaled, N_elems, float(args[9]), BLOCK=BLOCK_SCALE, num_warps=4, num_stages=2)

        # Reshape to [B, S, N]
        y = y_flat_scaled.view(B, S, N).to(torch.bfloat16)

        # Add positional embedding [S, N], broadcast over batch
        pos_emb = args[8][:S, :].contiguous().to(torch.bfloat16)  # [S, N]
        y_flat_add = y.view(-1)  # [B*S*N]
        N_elems_add = y_flat_add.numel()
        BLOCK_ADD = 1024
        grid_add = (triton.cdiv(N_elems_add, BLOCK_ADD),)
        add_pos_emb_kernel[grid_add](y_flat_add, pos_emb.view(-1), N_elems_add, S, N, BLOCK=BLOCK_ADD, num_warps=4, num_stages=2)

        y = y_flat_add.view(B, S, N)

        return y


def run(*args):
    return ModelNew()(*args)
