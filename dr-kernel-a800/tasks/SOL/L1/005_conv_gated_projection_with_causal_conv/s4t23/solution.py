import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float32, input x: (B, S, H)
    W_ptr,         # *const float32, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float32, in_proj_bias: (I,)
    Out_ptr,       # *float32, output BCx: (B, S, I)
    B: tl.int32,   # batch size
    S: tl.int32,   # sequence length
    H: tl.int32,   # hidden size
    I: tl.int32,   # output channels = 3*H
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for w
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0)
            w_vals = tl.load(W_ptr + i * w_i_stride + h_offsets * w_h_stride, mask=h_mask, other=0.0)
            acc += tl.sum(x_vals * w_vals, axis=0)
        # add bias if provided
        b_i = tl.load(Bias_ptr + i)
        acc += b_i
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_pad_ptr,     # *const float32, input after padding: (B, H, S+pad)
    W_ptr,          # *const float32, conv_weight: (H, 1, 4)
    Bias_ptr,       # *const float32, conv_bias: (H,)
    Out_ptr,        # *float32, output conv_out: (B, H, S)
    B: tl.int32,    # batch size
    S: tl.int32,    # original sequence length
    H: tl.int32,    # hidden size
    PAD: tl.int32,  # padding = conv_kernel_size - 1 = 3
    # strides for Bx_pad
    bx_b_stride: tl.int32, bx_h_stride: tl.int32, bx_t_stride: tl.int32,
    # strides for W
    w_g_stride: tl.int32, w_k_stride: tl.int32,          # W is (H, 1, 4)
    # strides for Out
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
):
    # Grid: axis=0 over B*H (one program per (b, g))
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    bx_base = Bx_pad_ptr + b * bx_b_stride + g * bx_h_stride
    out_base = Out_ptr + b * out_b_stride + g * out_h_stride

    # We will write one output position per loop to keep things simple and avoid masks on t vectors
    for t in range(0, S):
        acc = tl.zeros((), dtype=tl.float32)
        # k = 0
        t_in = t + 0 - PAD
        if t_in >= 0:
            x0 = tl.load(bx_base + t_in * bx_t_stride)
            w0 = tl.load(W_ptr + g * w_g_stride + 0 * w_k_stride)
            acc += x0 * w0
        # k = 1
        t_in = t + 1 - PAD
        if t_in >= 0:
            x1 = tl.load(bx_base + t_in * bx_t_stride)
            w1 = tl.load(W_ptr + g * w_g_stride + 1 * w_k_stride)
            acc += x1 * w1
        # k = 2
        t_in = t + 2 - PAD
        if t_in >= 0:
            x2 = tl.load(bx_base + t_in * bx_t_stride)
            w2 = tl.load(W_ptr + g * w_g_stride + 2 * w_k_stride)
            acc += x2 * w2
        # k = 3
        t_in = t + 3 - PAD
        if t_in >= 0:
            x3 = tl.load(bx_base + t_in * bx_t_stride)
            w3 = tl.load(W_ptr + g * w_g_stride + 3 * w_k_stride)
            acc += x3 * w3
        # add bias
        b_g = tl.load(Bias_ptr + g)
        acc += b_g
        # store
        tl.store(out_base + t * out_s_stride, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,          # *const float32, input y: (B, S, H)
    Wout_ptr,       # *const float32, out_proj_weight: (H, H)
    Bout_ptr,       # *const float32, out_proj_bias: (H,)
    Out_ptr,        # *float32, output: (B, S, H)
    B: tl.int32,    # batch size
    S: tl.int32,    # sequence length
    H: tl.int32,    # hidden size
    # strides for Y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for Wout
    wout_hout_stride: tl.int32, wout_hin_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_HIN: tl.constexpr,
):
    # Grid: axis=0 over B*S (one program per (b, s))
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for h_in in range(0, H, BLOCK_HIN):
            h_offsets = h_in + tl.arange(0, BLOCK_HIN)
            h_mask = h_offsets < H
            y_vals = tl.load(y_base + h_offsets * y_h_stride, mask=h_mask, other=0.0)
            w_vals = tl.load(Wout_ptr + h_out * wout_hout_stride + h_offsets * wout_hin_stride, mask=h_mask, other=0.0)
            acc += tl.sum(y_vals * w_vals, axis=0)
        b_out = tl.load(Bout_ptr + h_out)
        acc += b_out
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters in this module; weights are passed at forward time.

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton implementation of the fused computation:
        1) x -> BCx via in_proj_weight/bias (linear), split into B_tensor, C_tensor, x_proj
        2) Bx = B_tensor * x_proj
        3) Grouped causal 1D conv on Bx with kernel_size=4 and groups=H
        4) y = C_tensor * conv_out
        5) Output = linear(y, out_proj_weight, out_proj_bias)
        """

        B, S, H = x.shape
        I = 3 * H
        K = conv_weight.shape[2]
        assert conv_weight.shape[0] == H and conv_weight.shape[1] == 1, "conv_weight must be (H, 1, K)"
        assert conv_bias is not None and conv_bias.shape[0] == H, "conv_bias must be shape (H,)"
        assert out_proj_weight.shape[0] == H and out_proj_weight.shape[1] == H, "out_proj_weight must be (H, H)"
        assert out_proj_bias is not None and out_proj_bias.shape[0] == H, "out_proj_bias must be shape (H,)"

        # Ensure all tensors are contiguous and float32 for Triton
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        # Step 1: in_proj linear
        BCx = torch.empty((B, S, I), device=x.device, dtype=torch.float32)
        # Launch Triton kernel
        # Strides for x: (B, S, H) => strides = (S*H, H, 1)
        x_b_stride = x.stride(0); x_s_stride = x.stride(1); x_h_stride = x.stride(2)
        # Strides for W: (I, H)
        w_i_stride = in_proj_weight.stride(0); w_h_stride = in_proj_weight.stride(1)
        # Strides for Out: (B, S, I)
        out_b_stride = BCx.stride(0); out_s_stride = BCx.stride(1); out_i_stride = BCx.stride(2)
        grid_in = (B * S,)
        BLOCK_H = 64
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, I,
            x_b_stride, x_s_stride, x_h_stride,
            w_i_stride, w_h_stride,
            out_b_stride, out_s_stride, out_i_stride,
            BLOCK_H=BLOCK_H,
            num_warps=4
        )

        # Now split BCx into B_tensor, C_tensor, x_proj_tensor along last dim
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2*H]
        x_proj_tensor = BCx[:, :, 2*H:]

        # Step 2: Elementwise gating in PyTorch (simple multiply, light op)
        Bx = B_tensor * x_proj_tensor  # (B, S, H), float32

        # Step 3: Grouped causal conv. Pad on host to avoid Triton mask complexity
        pad = K - 1  # kernel_size - 1
        Bx_padded = torch.nn.functional.pad(Bx, (pad, 0))  # (B, H, S+pad), float32

        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        # Launch Triton kernel (grid over B*H)
        # Strides for Bx_padded: (B, H, S+pad)
        bx_b_stride = Bx_padded.stride(0); bx_h_stride = Bx_padded.stride(1); bx_t_stride = Bx_padded.stride(2)
        # Strides for W: (H, 1, 4)
        w_g_stride = conv_weight.stride(0); w_k_stride = conv_weight.stride(2)  # conv_weight is (H, 1, 4), second dim is 1
        # Strides for Out: (B, H, S)
        out_b_stride_conv = conv_out.stride(0); out_h_stride_conv = conv_out.stride(1); out_s_stride_conv = conv_out.stride(2)
        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, S, H, pad,
            bx_b_stride, bx_h_stride, bx_t_stride,
            w_g_stride, w_k_stride,
            out_b_stride_conv, out_h_stride_conv, out_s_stride_conv,
            num_warps=4
        )

        # Step 4: Output gating (PyTorch elementwise)
        y = C_tensor * conv_out  # (B, S, H), float32

        # Step 5: Out-proj linear in Triton
        out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        # Strides for y: (B, S, H)
        y_b_stride = y.stride(0); y_s_stride = y.stride(1); y_h_stride = y.stride(2)
        # Strides for W_out: (H, H)
        wout_hout_stride = out_proj_weight.stride(0); wout_hin_stride = out_proj_weight.stride(1)
        # Strides for out: (B, S, H)
        out_b_stride_lin = out.stride(0); out_s_stride_lin = out.stride(1); out_h_stride_lin = out.stride(2)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y_b_stride, y_s_stride, y_h_stride,
            wout_hout_stride, wout_hin_stride,
            out_b_stride_lin, out_s_stride_lin, out_h_stride_lin,
            BLOCK_HIN=64,
            num_warps=4
        )

        return out

# If you want to run locally for sanity:
# m = ModelNew()
# x = torch.randn(2, 4096, 256, device='cuda', dtype=torch.float32)
# in_proj_weight = torch.randn(768, 256, device='cuda', dtype=torch.float32)
# in_proj_bias = torch.randn(768, device='cuda', dtype=torch.float32)
# conv_weight = torch.randn(256, 1, 4, device='cuda', dtype=torch.float32)  # (H, 1, K)
# conv_bias = torch.randn(256, device='cuda', dtype=torch.float32)
# out_proj_weight = torch.randn(256, 256, device='cuda', dtype=torch.float32)
# out_proj_bias = torch.randn(256, device='cuda', dtype=torch.float32)
# y = m(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)
# print(y.shape)  # should be (2, 4096, 256)


def run(*args):
    return ModelNew()(*args)
