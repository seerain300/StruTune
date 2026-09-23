import math
import torch
import triton
import triton.language as tl

# Triton kernel: conv2d stride=2, padding=1, 3x3, output per (b, oc, oh, ow)
@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W_in, C_out, H_out, W_out,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros([1], dtype=tl.float32)

    for ic in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                ih = 2 * oh + kh - 1
                iw = 2 * ow + kw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W_in)
                x_ptr = X_ptr + b * (C_in * H * W_in) + ic * (H * W_in) + ih * W_in + iw
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)
                w_ptr = W_ptr + oc * (C_in * 3 * 3) + ic * (3 * 3) + kh * 3 + kw
                w_val = tl.load(w_ptr)
                acc += x_val * w_val

    bias_val = tl.load(BIAS_ptr + oc)
    acc = acc + bias_val

    y_ptr = Y_ptr + b * (C_out * H_out * W_out) + oc * (H_out * W_out) + oh * W_out + ow
    tl.store(y_ptr, acc)


# Triton kernel: GELU exact (erf-based), 1D over flattened tensor
@triton.jit
def gelu_exact_kernel(X_ptr, Y_ptr, NUMEL):
    pid = tl.program_id(0)
    # Each program processes 1024 elements
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < NUMEL
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    erf_arg = x * inv_sqrt2
    erf_val = tl.libdevice.erf(erf_arg)
    y = 0.5 * x * (1.0 + erf_val)
    tl.store(Y_ptr + offsets, y, mask=mask)


# Triton kernel: linear projection and add positional embedding
# X: [B, T_final, N], W: [M, N], pos: [T_final, M], Y: [B, T_final, M]
@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T_final, N, M,
    scale: tl.float32,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # For each m, accumulate over N in tiles
    for m0 in range(0, M, 128):
        m_offsets = m0 + tl.arange(0, 128)
        mask_m = m_offsets < M
        acc = tl.zeros([128], dtype=tl.float32)
        for n0 in range(0, N, 256):
            n_offsets = n0 + tl.arange(0, 256)
            mask_n = n_offsets < N
            # Load X[b, t, n_offsets]
            x_ptr_elem = X_ptr + b * (T_final * N) + t * N + n_offsets
            x_vals = tl.load(x_ptr_elem, mask=mask_n, other=0.0)  # [256]
            # Load W[m_offsets, n_offsets] as [128, 256]
            w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
            w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)  # [128, 256]
            # Accumulate: sum over n of x * w per m
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        # Apply scale and add pos_emb[t, :]
        acc = acc * scale
        pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
        acc = acc + pos_vec
        # Store to Y[b, t, m_offsets]
        y_ptrs = Y_ptr + b * (T_final * M) + t * M + m_offsets
        tl.store(y_ptrs, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), bfloat16
        conv weights/bias: bfloat16
        conv_out_weight: (1024, 3840), bfloat16
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float (e.g., 32.0)
        Returns: (B, time_after_conv, 1024)
        """

        # Ensure CUDA
        device = input_features.device
        if device.type != "cuda":
            input_features = input_features.to("cuda")
            conv2d1_weight = conv2d1_weight.to("cuda")
            conv2d1_bias = conv2d1_bias.to("cuda")
            conv2d2_weight = conv2d2_weight.to("cuda")
            conv2d2_bias = conv2d2_bias.to("cuda")
            conv2d3_weight = conv2d3_weight.to("cuda")
            conv2d3_bias = conv2d3_bias.to("cuda")
            conv_out_weight = conv_out_weight.to("cuda")
            positional_embedding = positional_embedding.to("cuda")

        # Stage 1 conv: (1 -> 384)
        B, C_in, H, W_in = input_features.shape  # (B,1,80,T)
        C_out = 384
        H1 = (H - 1) // 2 + 1  # 40
        W1 = (W_in - 1) // 2 + 1  # OW1
        X1 = torch.empty((B, C_out, H1, W1), dtype=torch.float32, device=device)
        grid1 = (B, C_out, H1, W1)
        conv2d_stride2_kernel[grid1](
            input_features.float(), conv2d1_weight.float(), conv2d1_bias.float(), X1,
            B, C_in, H, W_in, C_out, H1, W1,
        )
        # GELU after conv1
        X1_flat = X1.reshape(-1)
        Y1_flat = torch.empty_like(X1_flat, dtype=torch.float32, device=device)
        numel1 = X1_flat.numel()
        gelu_exact_kernel[(numel1 + 1023) // 1024](X1_flat, Y1_flat, numel1)
        X1 = Y1_flat.reshape_as(X1)

        # Stage 2 conv: (384 -> 384)
        C_in2 = C_out
        H2 = (H1 - 1) // 2 + 1  # 20
        W2 = (W1 - 1) // 2 + 1  # OW2
        X2 = torch.empty((B, C_out, H2, W2), dtype=torch.float32, device=device)
        grid2 = (B, C_out, H2, W2)
        conv2d_stride2_kernel[grid2](
            X1, conv2d2_weight.float(), conv2d2_bias.float(), X2,
            B, C_in2, H1, W1, C_out, H2, W2,
        )
        # GELU after conv2
        X2_flat = X2.reshape(-1)
        Y2_flat = torch.empty_like(X2_flat, dtype=torch.float32, device=device)
        numel2 = X2_flat.numel()
        gelu_exact_kernel[(numel2 + 1023) // 1024](X2_flat, Y2_flat, numel2)
        X2 = Y2_flat.reshape_as(X2)

        # Stage 3 conv: (384 -> 384)
        H3 = (H2 - 1) // 2 + 1  # 10
        W3 = (W2 - 1) // 2 + 1  # OW3
        X3 = torch.empty((B, C_out, H3, W3), dtype=torch.float32, device=device)
        grid3 = (B, C_out, H3, W3)
        conv2d_stride2_kernel[grid3](
            X2, conv2d3_weight.float(), conv2d3_bias.float(), X3,
            B, C_out, H2, W2, C_out, H3, W3,
        )
        # GELU after conv3
        X3_flat = X3.reshape(-1)
        Y3_flat = torch.empty_like(X3_flat, dtype=torch.float32, device=device)
        numel3 = X3_flat.numel()
        gelu_exact_kernel[(numel3 + 1023) // 1024](X3_flat, Y3_flat, numel3)
        X3 = Y3_flat.reshape_as(X3)

        # Final linear: (B, W3, 384*10) -> (B, W3, 1024), then scale and add pos
        T_final = W3  # time_after_conv
        N = C_out * 10  # 3840
        # Reshape X3 to (B, T_final, N)
        X3_perm = X3.permute(0, 3, 1, 2).contiguous()  # (B, W3, 384, 10)
        X3_reshaped = X3_perm.view(B, T_final, N)  # (B, W3, 3840)

        M = 1024
        Y = torch.empty((B, T_final, M), dtype=torch.float32, device=device)

        # Launch linear + pos kernel
        linear_project_pos_kernel[(B, T_final)](
            X3_reshaped, conv_out_weight.float(), positional_embedding.float(), Y,
            B, T_final, N, M,
            scale=embed_scale,  # 32.0
        )

        return Y


def run(*args):
    return ModelNew()(*args)
