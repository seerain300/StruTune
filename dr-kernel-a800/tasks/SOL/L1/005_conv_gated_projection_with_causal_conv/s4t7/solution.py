import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H), expected float32 compute buffer
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H, float32
    BIAS_ptr,      # *const float, in_proj_bias: (I,), or dummy if BIAS_flag=0
    Out_ptr,       # *float, output BCx: (B, S, I), pointer dtype determines output element type
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    BIAS_flag: tl.int32,  # 1 if bias provided, 0 otherwise
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride

    # Loop over output channels i in [0, I)
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        # Reduction over input H
        for h in range(0, H):
            x_val = tl.load(x_base + h * x_h_stride).to(tl.float32)
            w_val = tl.load(W_ptr + i * H + h).to(tl.float32)
            acc += x_val * w_val
        if BIAS_flag == 1:
            bias_val = tl.load(BIAS_ptr + i).to(tl.float32)
            acc += bias_val
        out_ptr = Out_ptr + b * out_b_stride + s * out_s_stride + i * out_i_stride
        # Store as float32; Triton will implicitly cast to pointer dtype if needed.
        tl.store(out_ptr, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *const float, input after gating: (B, H, S), float32 compute buffer
    W_ptr,         # *const float, conv_weight: (H, 1, K), float32
    Bias_ptr,      # *const float, conv_bias: (H,), or dummy if BIAS_flag=0
    Out_ptr,       # *float, output conv_out: (B, H, S), pointer dtype determines output element type
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    K: tl.int32,   # kernel_size, e.g., 4
    BIAS_flag: tl.int32,  # 1 if bias provided
    bx_b_stride: tl.int32, bx_h_stride: tl.int32, bx_s_stride: tl.int32,
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
    BLOCK_T: tl.constexpr,  # tile over S
):
    # Each program handles one (b, g)
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    num_tiles = (S + BLOCK_T - 1) // BLOCK_T
    for tile in range(0, num_tiles):
        t_offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # K-loop: causal kernel_size
        for k in range(0, K):
            pos = t_offsets + k - 1  # causal: input at t - k + 1
            valid = (pos >= 0) & (pos < S) & t_mask

            # Load Bx[b, g, pos]
            bx_ptrs = Bx_ptr + b * bx_b_stride + g * bx_h_stride + pos * bx_s_stride
            bx_vals = tl.load(bx_ptrs, mask=valid, other=0.0).to(tl.float32)

            # Load weight[g, k]
            w_val = tl.load(W_ptr + g * (1 * K) + k).to(tl.float32)

            acc += bx_vals * w_val

        if BIAS_flag == 1:
            bias_val = tl.load(Bias_ptr + g).to(tl.float32)
            acc += bias_val

        out_ptrs = Out_ptr + b * out_b_stride + g * out_h_stride + t_offsets * out_s_stride
        tl.store(out_ptrs, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H), float32 compute buffer
    W_ptr,         # *const float, out_proj_weight: (H, H), float32
    BIAS_ptr,      # *const float, bias: (H,) or dummy if BIAS_flag=0
    Out_ptr,       # *float, output: (B, S, H), pointer dtype determines output element type
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    BIAS_flag: tl.int32,
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for h_in in range(0, H):
            y_val = tl.load(y_base + h_in * y_h_stride).to(tl.float32)
            w_val = tl.load(W_ptr + h_out * H + h_in).to(tl.float32)
            acc += y_val * w_val
        if BIAS_flag == 1:
            bias_val = tl.load(BIAS_ptr + h_out).to(tl.float32)
            acc += bias_val
        out_ptr = out_base + h_out * out_h_stride
        tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Triton-optimized fused model:
        1) in_proj: x (B, S, H) -> BCx (B, S, 3H)
        2) Slice into B, C, x_proj
        3) Elementwise gating Bx = B * x_proj
        4) Grouped causal conv1d with kernel_size=4 and groups=H on Bx
        5) Output gating: y = C * conv_out
        6) out_proj: y -> (B, S, H)
        All heavy ops (in_proj, conv, out_proj) are Triton kernels; elementwise ops are PyTorch.
        """
        device = x.device
        B, S, H = x.shape
        I = 3 * H
        K = 4

        # 1) Compute BCx via Triton linear: use float32 compute buffers
        x32 = x.contiguous().to(torch.float32)           # (B, S, H)
        in_proj_weight32 = in_proj_weight.contiguous().to(torch.float32)  # (I, H)
        # Allocate BCx with desired output dtype: match x dtype (typically float32)
        BCx = torch.empty((B, S, I), device=device, dtype=x.dtype)

        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x32, in_proj_weight32,
            in_proj_bias.contiguous().to(torch.float32) if in_proj_bias is not None else torch.empty(1, device=device, dtype=torch.float32),
            BCx,
            B, S, H, I,
            1 if in_proj_bias is not None else 0,
            x32.stride(0), x32.stride(1), x32.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Slice BCx into B_tensor, C_tensor, x_proj_tensor (PyTorch views)
        B_tensor = BCx[:, :, :H]          # (B, S, H)
        C_tensor = BCx[:, :, H:2*H]       # (B, S, H)
        x_proj_tensor = BCx[:, :, 2*H:]   # (B, S, H)

        # 3) Elementwise gating Bx = B * x_proj (torch op)
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 4) Grouped causal conv in Triton: input (B, H, S) derived from Bx
        Bx_trans = Bx.transpose(1, 2).contiguous().to(torch.float32)  # (B, S, H) -> (B, H, S)
        conv_weight32 = conv_weight.contiguous().to(torch.float32)    # (H, 1, K)
        conv_bias32 = conv_bias.contiguous().to(torch.float32) if conv_bias is not None else torch.empty(1, device=device, dtype=torch.float32)
        conv_out = torch.empty((B, H, S), device=device, dtype=x.dtype)  # output dtype matches input x dtype (typically float32)

        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_trans, conv_weight32, conv_bias32,
            conv_out,
            B, S, H, K,
            1 if conv_bias is not None else 0,
            Bx_trans.stride(0), Bx_trans.stride(1), Bx_trans.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=128,
            num_warps=4,
        )

        # 5) Output gating y = C * conv_out (torch op). conv_out: (B, H, S), C_tensor: (B, S, H)
        conv_out_T = conv_out.transpose(1, 2).contiguous()  # (B, S, H)
        y = C_tensor * conv_out_T  # (B, S, H)

        # 6) Final out_proj via Triton linear
        y32 = y.contiguous().to(torch.float32)                 # (B, S, H)
        out_proj_weight32 = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_proj_bias32 = out_proj_bias.contiguous().to(torch.float32) if out_proj_bias is not None else torch.empty(1, device=device, dtype=torch.float32)
        output = torch.empty((B, S, H), device=device, dtype=x.dtype)  # match x dtype

        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y32, out_proj_weight32, out_proj_bias32, output,
            B, S, H,
            1 if out_proj_bias is not None else 0,
            y32.stride(0), y32.stride(1), y32.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
