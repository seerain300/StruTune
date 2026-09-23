import torch
import triton
import triton.language as tl

@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I,)
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for w
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Each program handles one (b, s)
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
        # add bias
        bias_val = tl.load(Bias_ptr + i)
        acc += bias_val
        # store to Out[b, s, i]
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def gated_mul_kernel(
    A_ptr, B_ptr, Out_ptr,
    Bsz: tl.int32, S: tl.int32, H: tl.int32,
    a_b_stride: tl.int32, a_s_stride: tl.int32, a_h_stride: tl.int32,
    b_b_stride: tl.int32, b_s_stride: tl.int32, b_h_stride: tl.int32,
    o_b_stride: tl.int32, o_s_stride: tl.int32, o_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    a_ptr = A_ptr + b * a_b_stride + s * a_s_stride + h * a_h_stride
    b_ptr = B_ptr + b * b_b_stride + s * b_s_stride + h * b_h_stride

    a_val = tl.load(a_ptr)
    b_val = tl.load(b_ptr)
    out_val = a_val * b_val
    o_ptr = Out_ptr + b * o_b_stride + s * o_s_stride + h * o_h_stride
    tl.store(o_ptr, out_val)


@triton.jit
def pad1d_causal_kernel(
    In_ptr,        # *const float, input tensor to pad: (B, C, L) where C=H and L=S
    Out_ptr,       # *float, output padded tensor: (B, C, L + K - 1), K=4 (causal left pad)
    B: tl.int32, C: tl.int32, L: tl.int32, K: tl.constexpr,
    in_b_stride: tl.int32, in_c_stride: tl.int32, in_l_stride: tl.int32,
    out_b_stride: tl.int32, out_c_stride: tl.int32, out_l_stride: tl.int32,
):
    # Each program handles one (b, c)
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C
    # Process t in tiles along L
    BLOCK_T = 128
    for t in range(0, L, BLOCK_T):
        t_offsets = t + tl.arange(0, BLOCK_T)
        # out indices for this t tile are [t, t+1, ..., t+BLOCK_T-1]
        # input indices would be [t + 1 - k] with k in [0..K-1], but here we pad zeros on left.
        # We write Out[b, c, t_out] = (In[b, c, t_out + 1]) if t_out + 1 in [0, L), else 0.
        out_base = Out_ptr + b * out_b_stride + c * out_c_stride
        in_base = In_ptr + b * in_b_stride + c * in_c_stride

        # We need to shift t_offsets by +1 to map to input indices; handle out-of-range with zeros
        # Compute t_in = t_offsets + 1
        t_in = t_offsets + 1
        valid_in = (t_in >= 0) & (t_in < L)

        # Load input values for positions t_in, masked; other=0.0
        in_vals = tl.load(in_base + t_in * in_l_stride, mask=valid_in, other=0.0)
        # Store into Out at t_offsets
        tl.store(out_base + t_offsets * out_l_stride, in_vals, mask=(t_offsets < (L + K - 1)))


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *const float, input Bx padded: (B, C=H, L=S+K-1)
    W_ptr,         # *const float, conv_weight: (C=H, 1, K=4)
    Bias_ptr,      # *const float, conv_bias: (C=H,)
    Out_ptr,       # *float, output conv_out: (B, C=H, L_out=S)
    B: tl.int32, C: tl.int32, L: tl.int32, K: tl.constexpr,
    bx_b_stride: tl.int32, bx_c_stride: tl.int32, bx_l_stride: tl.int32,
    w_c_stride: tl.int32, w_k_stride: tl.int32,
    out_b_stride: tl.int32, out_c_stride: tl.int32, out_l_stride: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # Each program handles one (b, c)
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    bx_base = Bx_ptr + b * bx_b_stride + c * bx_c_stride
    out_base = Out_ptr + b * out_b_stride + c * out_c_stride

    # Loop over output positions t
    for t in range(0, L, BLOCK_T):
        t_offsets = t + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < L  # L_out == L (no extra padding in output)

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # K is constexpr (4); unroll
        for k in range(0, K):
            # input index = t + k - 1 (causal)
            t_in = t_offsets + k - 1
            valid_in = (t_in >= 0) & (t_in < L)
            vals = tl.load(bx_base + t_in * bx_l_stride, mask=valid_in & mask_t, other=0.0)
            # load weight for this c and k
            w_val = tl.load(W_ptr + c * w_c_stride + k * w_k_stride)
            acc += vals * w_val

        # add bias
        bias_val = tl.load(Bias_ptr + c)
        acc += bias_val

        # store to Out[b, c, t]
        tl.store(out_base + t_offsets * out_l_stride, acc, mask=mask_t)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_ptr,         # *const float, out_proj_weight: (H, H)
    Bias_ptr,      # *const float, out_proj_bias: (H,)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_hout_stride: tl.int32, w_hin_stride: tl.int32,
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
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H
            y_vals = tl.load(y_base + h_in_offsets * y_h_stride, mask=h_in_mask, other=0.0)
            w_vals = tl.load(W_ptr + h_out * w_hout_stride + h_in_offsets * w_hin_stride, mask=h_in_mask, other=0.0)
            acc += tl.sum(y_vals * w_vals, axis=0)
        # add bias
        bias_val = tl.load(Bias_ptr + h_out)
        acc += bias_val
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        Triton-optimized fused computation:
        1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, I), I=3*H
        2) Split: B = BCx[:, :, :H], x_proj = BCx[:, :, 2*H:], C = BCx[:, :, H:2*H]
        3) Gate: Bx = B * x_proj
        4) Pad: Bx_padded with causal left pad K=4
        5) Grouped causal conv: conv_out = conv(Bx_padded, conv_weight, conv_bias, groups=H)
        6) Gate: y = C * conv_out
        7) Out-proj: output = F.linear(y, out_proj_weight, out_proj_bias)
        All heavy ops (linear and conv) are Triton kernels; elementwise ops and slicing are lightweight PyTorch.
        """
        B, S, H = x.shape
        I = 3 * H  # in_proj outputs 3*H channels

        # 1) in_proj linear: BCx of shape (B, S, I)
        x_contig = x.contiguous()
        w_contig = in_proj_weight.contiguous()
        bias_in_contig = in_proj_bias.contiguous()

        # Allocate BCx with same dtype as x
        BCx = torch.empty((B, S, I), dtype=x.dtype, device=x.device)

        # Launch Triton kernel (grid over B*S)
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x_contig, w_contig, bias_in_contig, BCx,
            B, S, H, I,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
            w_contig.stride(0), w_contig.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Split BCx into B, C, x_proj
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2*H]
        x_proj_tensor = BCx[:, :, 2*H:]

        # 3) Bx = B * x_proj (elementwise gating)
        Bx = B_tensor * x_proj_tensor  # PyTorch elementwise; lightweight

        # 4) Pad Bx with causal left pad K=4
        K = 4
        Bx_padded = torch.empty((B, H, S + K - 1), dtype=x.dtype, device=x.device)

        # Launch pad kernel
        grid_pad = (B * H,)
        pad1d_causal_kernel[grid_pad](
            Bx, Bx_padded,
            B, H, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            num_warps=2,
        )

        # 5) Grouped causal conv1d: conv_out (B, H, S) with groups=H
        conv_weight_contig = conv_weight.contiguous()
        conv_bias_contig = conv_bias.contiguous()

        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)

        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_weight_contig, conv_bias_contig, conv_out,
            B, H, S, K,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight_contig.stride(0), conv_weight_contig.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_T=128,
            num_warps=4,
        )

        # 6) y = C * conv_out (elementwise gating)
        # y has shape (B, H, S)
        y = C_tensor.transpose(1, 2) * conv_out  # C_tensor: (B, S, H); conv_out: (B, H, S)
        y = y.transpose(1, 2)  # (B, S, H) for out-proj

        # 7) out_proj linear: output (B, S, H)
        out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)

        # Launch Triton kernel over B*S
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight.contiguous(), out_proj_bias.contiguous(), out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=64,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
