import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const T, input x: (B, S, H)
    W_ptr,         # *const T, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const T, in_proj_bias: (I) or nullptr
    Out_ptr,       # *T, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    has_bias: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
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
            x_vals = tl.load(
                x_base + h_offsets * x_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)
            w_vals = tl.load(
                W_ptr + i * w_i_stride + h_offsets * w_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)
            acc += tl.sum(x_vals * w_vals, axis=0)
        if has_bias:
            bval = tl.load(Bias_ptr + i).to(tl.float32)
            acc += bval
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    In_ptr,        # *const T, input: (B, H, S)
    W_ptr,         # *const T, weight: (H, 1, 4)
    Bias_ptr,      # *const T, bias: (H) or nullptr
    Out_ptr,       # *T, output: (B, H, S)
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    K: tl.int32,   # kernel_size = 4
    # strides
    in_b_stride: tl.int32, in_h_stride: tl.int32, in_s_stride: tl.int32,
    w_g_stride: tl.int32, w_k_stride: tl.int32,
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
    has_bias: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # One program per (b, g)
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    in_base = In_ptr + b * in_b_stride + g * in_h_stride
    out_base = Out_ptr + b * out_b_stride + g * out_h_stride

    # Compute outputs for t = 0..S-1 in tiles of BLOCK_T
    for t_start in range(0, S, BLOCK_T):
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        # K is small (4), loop over k explicitly to match PyTorch conv semantics
        for k in range(0, K):
            in_t = t_offsets + k - 1  # causal padding
            valid_in = (in_t >= 0) & (in_t < S) & t_mask
            val = tl.load(in_base + in_t * in_s_stride, mask=valid_in, other=0.0)
            w_k = tl.load(W_ptr + g * w_g_stride + k * w_k_stride)
            acc += val * w_k
        if has_bias:
            bval = tl.load(Bias_ptr + g).to(tl.float32)
            acc += bval
        tl.store(out_base + t_offsets * out_s_stride, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const T, input y: (B, S, H)
    W_ptr,         # *const T, out_proj_weight: (H, H)
    Bias_ptr,      # *const T, out_proj_bias: (H) or nullptr
    Out_ptr,       # *T, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_h_out_stride: tl.int32, w_h_in_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    has_bias: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H
            y_vals = tl.load(
                y_base + h_in_offsets * y_h_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)
            w_vals = tl.load(
                W_ptr + h_out * w_h_out_stride + h_in_offsets * w_h_in_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)
            acc += tl.sum(y_vals * w_vals, axis=0)
        if has_bias:
            bval = tl.load(Bias_ptr + h_out).to(tl.float32)
            acc += bval
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Shapes
        B, S, H = x.shape
        I = 3 * H
        K = conv_weight.shape[2]  # should be 4

        # 1) in_proj: (B, S, H) -> (B, S, I)
        BCx = torch.empty((B, S, I), dtype=x.dtype, device=x.device)
        in_proj_linear_kernel[(B * S,)](
            x, in_proj_weight, (in_proj_bias if in_proj_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)),
            BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            (1 if in_proj_bias is not None else 0),
            BLOCK_H=64,
            num_warps=4
        )

        # 2) Split BCx into B, C, x_proj
        B_tensor = BCx[:, :, :H]            # (B, S, H)
        C_tensor = BCx[:, :, H:2*H]         # (B, S, H)
        x_proj = BCx[:, :, 2*H:]            # (B, S, H)

        # Elementwise gating: Bx = B * x_proj
        Bx = B_tensor * x_proj  # (B, S, H)

        # 3) Grouped causal conv on Bx_trans = (B, H, S)
        Bx_trans = Bx.transpose(1, 2).contiguous()  # (B, H, S)
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        # Launch grouped causal conv kernel: one program per (b, g)
        grouped_causal_conv1d_kernel[(B * H,)](
            Bx_trans, conv_weight, (conv_bias if conv_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)),
            conv_out,
            B, H, S, K,
            Bx_trans.stride(0), Bx_trans.stride(1), Bx_trans.stride(2),
            conv_weight.stride(0), conv_weight.stride(2),  # conv_weight is (H, 1, 4)
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            (1 if conv_bias is not None else 0),
            BLOCK_T=128,
            num_warps=4
        )

        # 4) Output gating: y = C * conv_out
        # conv_out is (B, H, S); C_tensor is (B, S, H); transpose C to (B, H, S)
        C_t = C_tensor.transpose(1, 2).contiguous()  # (B, H, S)
        y = C_t * conv_out  # elementwise multiply

        # 5) out_proj: y -> output (B, S, H)
        output = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        out_proj_linear_kernel[(B * S,)](
            y, out_proj_weight, (out_proj_bias if out_proj_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)),
            output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            (1 if out_proj_bias is not None else 0),
            BLOCK_H=64,
            num_warps=4
        )
        return output


def run(*args):
    return ModelNew()(*args)
