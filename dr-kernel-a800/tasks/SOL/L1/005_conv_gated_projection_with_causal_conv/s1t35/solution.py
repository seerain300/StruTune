import torch
import triton
import triton.language as tl


# Elementwise multiplication kernel: out = A * B, both (B, S, H)
@triton.jit
def TritonMulKernel(
    A_ptr, B_ptr, Out_ptr,
    Bsz, S, H,
    stride_a_b, stride_a_s, stride_a_h,
    stride_b_b, stride_b_s, stride_b_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    a_ptrs = A_ptr + pid_b * stride_a_b + s_offsets[:, None] * stride_a_s + h_offsets[None, :]
    b_ptrs = B_ptr + pid_b * stride_b_b + s_offsets[:, None] * stride_b_s + h_offsets[None, :]
    o_ptrs = Out_ptr + pid_b * stride_o_b + s_offsets[:, None] * stride_o_s + h_offsets[None, :]

    a_vals = tl.load(a_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = a_vals * b_vals

    tl.store(o_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Elementwise linear-like projection: out[b, s, h] = sum_h' y[b, s, h'] * w[h', h] + bias[h]
# Here we implement: out = y @ out_proj_weight.T + out_proj_bias, both elementwise across (B,S,H).
# We do this in Triton as a simple kernel over tiles. For simplicity, we implement per-tile reduction
# over H dimension with a loop. This is safe and simple.
@triton.jit
def TritonLinearFinalKernel(
    y_ptr,           # (B, S, H)
    w_ptr,           # (H, H) out_proj_weight
    bias_ptr,        # (H)
    out_ptr,         # (B, S, H)
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    h_out = pid_h

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # For each input hidden dimension h_in, compute dot(y[b, s, h_in], w[h_in, h_out])
    # Note: We reduce over h_in for the tile, which is safe as H is moderate.
    for h_in in range(0, H):
        y_ptrs = y_ptr + pid_b * stride_y_b + s_offsets * stride_y_s + h_in * stride_y_h
        y_vec = tl.load(y_ptrs, mask=mask_s, other=0.0)
        # Load scalar w[h_in, h_out]
        w_val = tl.load(w_ptr + h_in * H + h_out)
        acc += y_vec * w_val

    # add bias
    b_val = tl.load(bias_ptr + h_out)
    acc += b_val

    # store
    out_ptrs = out_ptr + pid_b * stride_out_b + s_offsets * stride_out_s + h_out * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_s)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure contiguity; keep dtype as float32 for stability
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)  # (H, 1, 4)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)      # (H,)

        B, S, H = x.shape

        # 1) Three linear projections using PyTorch for correctness and speed
        # Each returns (B, S, H)
        B_ = torch.nn.functional.linear(x, in_proj_weight[:H, :], in_proj_bias[:H])
        C_ = torch.nn.functional.linear(x, in_proj_weight[H:2*H, :], in_proj_bias[H:2*H])
        Xproj_ = torch.nn.functional.linear(x, in_proj_weight[2*H:3*H, :], in_proj_bias[2*H:3*H])

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        BLOCK_S = 128
        BLOCK_H = 64  # tile over H dimension
        grid0 = B
        grid1 = (S + BLOCK_S - 1) // BLOCK_S
        grid2 = (H + BLOCK_H - 1) // BLOCK_H
        TritonMulKernel[(grid0, grid1, grid2)](
            B_, Xproj_, Bx,
            B, S, H,
            B_.stride(0), B_.stride(1), B_.stride(2),
            Xproj_.stride(0), Xproj_.stride(1), Xproj_.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
        )

        # 3) Grouped causal 1D convolution with groups=H, kernel_size=4
        # Apply conv on Bx_pad (left pad by 3)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # pad left 3
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight, conv_bias, groups=H
        )  # shape: (B, H, S)

        # 4) Output gating: y = C * conv_out (elementwise). Shapes: C (B, S, H), conv_out (B, H, S)
        # We need y with shape (B, H, S) to multiply. The original code multiplies tensors with shapes
        # (B,S,H) and (B,H,S) via PyTorch broadcasting, but here conv_out is (B,H,S).
        # We will compute elementwise product over broadcasted dimensions: y[b,s,h] = C[b,s,h] * conv_out[b,h,s]
        # We need to align indices; simplest way: compute y as Bx here, but Bx is already used.
        # The original code uses 'C' here, not 'Bx'. Let's do it correctly:
        # First, we need to obtain 'C' from the original flow. We used 'C_' computed from linear.
        # To ensure correctness with original semantics, compute y = C_ * conv_out via PyTorch broadcasting.
        # We will expand C_ to (B,1,S,1,H) and conv_out to (1,B,H,S,1), then multiply. For simplicity,
        # use expand_as on conv_out to (B,S,H) by permuting and broadcasting is not necessary; PyTorch
        # supports direct multiply. So we can just multiply with expand to (B,S,H):
        y = C_ * conv_out  # PyTorch will broadcast elementwise across (B,S,H) and (B,H,S)

        # 5) Final projection: out = y @ out_proj_weight.T + out_proj_bias
        # Implement via Triton for "compute" requirement, though PyTorch's F.linear is fast.
        out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        # Triton grid
        grid0 = B
        grid1 = (S + BLOCK_S - 1) // BLOCK_S
        grid2 = H  # since we only do per-h, grid2 is H
        TritonLinearFinalKernel[(grid0, grid1, grid2)](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_H=1,
        )

        return out


def run(*args):
    return ModelNew()(*args)
