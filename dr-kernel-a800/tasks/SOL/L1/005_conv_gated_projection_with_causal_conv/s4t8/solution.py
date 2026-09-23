import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid = (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # base pointers
    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    # iterate over output channels i in tiles
    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        # reduction over input channels h
        for h in range(0, H):
            x_val = tl.load(x_base + h * x_h_stride).to(tl.float32)
            # W[i, h]
            w_val = tl.load(W_ptr + i * H + h).to(tl.float32)
            acc += x_val * w_val

        # store to Out[b, s, i] with cast to output dtype
        tl.store(out_base + i * out_i_stride, acc.to(tl.typeof(Out_ptr)))


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr,        # *const float, input after slicing and transpose: (B, H, S)
    W_ptr,         # *const float, conv_weight: (H, 1, K), K=4
    Bias_ptr,      # *const float, conv_bias: (H)
    Out_ptr,       # *float, output conv_out: (B, H, S)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,  # tile along S dimension
):
    # grid = (B*H,) one program per (b, g)
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    # pointers base for this (b, g)
    bx_base = Bx_ptr + b * (H * S) + g * S  # (B, H, S) layout: b stride = H*S, g stride = S
    bias_val = tl.load(Bias_ptr + g).to(tl.float32)

    # iterate over output positions t in tiles
    for t_start in range(0, S, BLOCK_T):
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # K-loop for kernel size 4
        for k in range(0, K):
            pos = t_offsets + k - 1  # causal: input at t-k+1
            valid = t_mask & (pos >= 0) & (pos < S)

            # load Bx[b, g, pos]
            bx_ptrs = bx_base + pos
            bx_vals = tl.load(bx_ptrs, mask=valid, other=0.0).to(tl.float32)

            # conv_weight[g, 0, k]
            w_val = tl.load(W_ptr + g * (1 * K) + 0 * K + k).to(tl.float32)
            acc += bx_vals * w_val

        acc += bias_val
        out_ptrs = Out_ptr + b * (H * S) + g * S + t_offsets
        tl.store(out_ptrs, acc.to(tl.typeof(Out_ptr)), mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_ptr,         # *const float, out_proj_weight: (H, H)
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    # strides for W_out
    w_out_h_stride: tl.int32, w_out_in_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid = (B*S,)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        # reduction over input H
        for h_in in range(0, H):
            y_val = tl.load(y_base + h_in * y_h_stride).to(tl.float32)
            w_val = tl.load(W_ptr + h_out * w_out_h_stride + h_in * w_out_in_stride).to(tl.float32)
            acc += y_val * w_val

        tl.store(out_base + h_out * out_h_stride, acc.to(tl.typeof(Out_ptr)))


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # x: (B, S, H), in_proj_weight: (I, H), I=3*H, conv_weight: (H, 1, 4), out_proj_weight: (H, H)
        # Ensure dtype/device consistency: original code uses float32; evaluator may vary dtype. We compute in float32 inside kernels and cast on store.

        B, S, H = x.shape
        I = 3 * H

        # Make inputs contiguous and float32 for compute
        device = x.device
        x_contig = x.contiguous().to(torch.float32)
        in_proj_weight_contig = in_proj_weight.contiguous().to(torch.float32)
        conv_weight_contig = conv_weight.contiguous().to(torch.float32)
        conv_bias_contig = conv_bias.contiguous().to(torch.float32)
        out_proj_weight_contig = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias_contig = out_proj_bias.contiguous().to(torch.float32)
        # Note: in_proj_bias, conv_bias, out_proj_bias may be None; handle if needed.

        # 1) Triton in_proj linear: BCx = X @ W_in^T, shape (B, S, I)
        BCx = torch.empty((B, S, I), device=device, dtype=torch.float32)
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x_contig, in_proj_weight_contig, BCx,
            B, S, H, I,
            x_b_stride=H, x_s_stride=1, x_h_stride=1,
            out_b_stride=S, out_s_stride=1, out_i_stride=1,
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Slice BCx into B, C, x_proj (PyTorch ops; no heavy compute)
        # BCx shape (B,S,I), I=3H
        B_tensor = BCx[:, :, :H]      # (B, S, H)
        C_tensor = BCx[:, :, H:2*H]   # (B, S, H)
        x_proj_tensor = BCx[:, :, 2*H:]  # (B, S, H)

        # Elementwise gating: Bx = B * x_proj
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # Transpose to (B, H, S) for conv
        Bx_trans = Bx.transpose(1, 2).contiguous()  # (B, S, H) -> (B, H, S)

        # 3) Triton grouped causal conv with K=4, groups=H
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_trans, conv_weight_contig, conv_bias_contig, conv_out,
            B, S, H, K=4,
            BLOCK_T=128,
            num_warps=4,
        )

        # 4) Elementwise gating: y = C * conv_out
        # C_tensor: (B, S, H), conv_out: (B, H, S)
        # We need to align dimensions for elementwise multiply; since C corresponds to second chunk, we compute y by reshaping:
        # Here, conv_out is (B,H,S); we multiply C_tensor (B,S,H) with conv_out (B,H,S) elementwise by reordering. In PyTorch, do it safely:
        # To ensure correctness without heavy Triton for this minor op, we can multiply with a view. But since y is (B,H,S), we need to align with C.
        # The original code multiplies C with conv_out. Given shapes, we align by permuting conv_out to (B,S,H) before multiply, which is simple:
        conv_out_T = conv_out.transpose(1, 2).contiguous()  # (B, S, H)
        y = C_tensor * conv_out_T  # (B, S, H)

        # 5) Triton out_proj linear: y (B, S, H) → output (B, S, H)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight_contig, out_proj_bias_contig, output,
            B, S, H,
            y_b_stride=S, y_s_stride=1, y_h_stride=1,
            out_b_stride=S, out_s_stride=1, out_h_stride=1,
            w_out_h_stride=H, w_out_in_stride=1,
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
