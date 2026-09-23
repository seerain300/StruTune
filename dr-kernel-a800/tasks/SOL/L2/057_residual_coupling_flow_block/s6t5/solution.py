import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time = axes_and_scalars["time"]
    channels = 192
    hidden_channels = 192
    half_channels = 96
    kernel_size = 5

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv1d(out_c, in_c, k):
        fan_in = in_c * k
        return torch.randn(out_c, in_c, k, device=device, generator=g) * math.sqrt(2.0 / fan_in)

    inputs = {
        "x": torch.randn(batch_size, channels, time, device=device, generator=g),
        # Binary mask
        "x_mask": torch.ones(batch_size, 1, time, device=device),
        "reverse": False,
    }

    # 4 transforms x 3 convs each
    for i in range(4):
        # conv0: hidden_channels out, half_channels in
        inputs[f"transform_{i}_conv0_weight"] = kaiming_conv1d(hidden_channels, half_channels, kernel_size)
        inputs[f"transform_{i}_conv0_bias"] = torch.randn(hidden_channels, device=device, generator=g)
        # conv1: hidden_channels out, hidden_channels in
        inputs[f"transform_{i}_conv1_weight"] = kaiming_conv1d(hidden_channels, hidden_channels, kernel_size)
        inputs[f"transform_{i}_conv1_bias"] = torch.randn(hidden_channels, device=device, generator=g)
        # conv2: half_channels out, hidden_channels in
        inputs[f"transform_{i}_conv2_weight"] = kaiming_conv1d(half_channels, hidden_channels, kernel_size)
        inputs[f"transform_{i}_conv2_bias"] = torch.randn(half_channels, device=device, generator=g)

    return inputs


# Triton kernels: conv1d with K=5, padding=2 (valid conv => T_out = T_in - 1)
@triton.jit
def conv1d_k5_p2(
    x_ptr,          # *f32, [B, C_in, T_in]
    w_ptr,          # *f32, [C_out, C_in, 5]
    bias_ptr,       # *f32, [C_out]
    y_ptr,          # *f32, [B, C_out, T_out] where T_out = T_in - 1
    B, C_in, T_in, C_out, T_out,
    # strides
    x_bs, x_cs, x_ts,
    w_os, w_cs, w_ks,
    y_bs, y_cs, y_ts,
    BLOCK_T: tl.constexpr,
):
    pid_bc = tl.program_id(0)  # over B * C_out
    b = pid_bc // C_out
    co = pid_bc % C_out
    # time tile
    t_start = tl.program_id(1) * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T_out

    # accumulate over input channels and kernel taps
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    # loop over input channels
    for ci in range(0, C_in):
        # loop over kernel taps k in [0..4]
        for k in range(0, 5):
            t_src = t_offsets + 2 - k  # padding=2
            x_off = b * x_bs + ci * x_cs + t_src * x_ts
            w_off = co * w_os + ci * w_cs + k * w_ks
            x_val = tl.load(x_ptr + x_off, mask=t_mask, other=0.0)
            w_val = tl.load(w_ptr + w_off)  # scalar
            acc += x_val * w_val
    # add bias
    b_val = tl.load(bias_ptr + co)
    acc = acc + b_val
    # store
    y_off = b * y_bs + co * y_cs + t_offsets * y_ts
    tl.store(y_ptr + y_off, acc, mask=t_mask)


@triton.jit
def add_bias(y_ptr, bias_ptr, B, C, T, y_bs, y_cs, y_ts):
    pid_bc = tl.program_id(0)  # over B * C
    b = pid_bc // C
    c = pid_bc % C
    # vector across T
    t_offsets = tl.arange(0, 128)  # BLOCK_T is 128 in launch
    t_mask = t_offsets < T
    y_off = b * y_bs + c * y_cs + t_offsets * y_ts
    y_val = tl.load(y_ptr + y_off, mask=t_mask, other=0.0)
    b_val = tl.load(bias_ptr + c)
    y_val = y_val + b_val
    tl.store(y_ptr + y_off, y_val, mask=t_mask)


@triton.jit
def relu_kernel(y_ptr, B, C, T, y_bs, y_cs, y_ts):
    pid_bc = tl.program_id(0)  # over B * C
    b = pid_bc // C
    c = pid_bc % C
    t_offsets = tl.arange(0, 128)
    t_mask = t_offsets < T
    y_off = b * y_bs + c * y_cs + t_offsets * y_ts
    y_val = tl.load(y_ptr + y_off, mask=t_mask, other=0.0)
    y_val = tl.maximum(y_val, 0.0)
    tl.store(y_ptr + y_off, y_val, mask=t_mask)


@triton.jit
def mul_mask(y_ptr, mask_ptr, B, C, T, y_bs, y_cs, y_ts, mask_bs, mask_cs, mask_ts):
    pid_bc = tl.program_id(0)  # over B * C
    b = pid_bc // C
    c = pid_bc % C
    t_offsets = tl.arange(0, 128)
    t_mask = t_offsets < T
    y_off = b * y_bs + c * y_cs + t_offsets * y_ts
    y_val = tl.load(y_ptr + y_off, mask=t_mask, other=0.0)
    # mask has shape [B, 1, T], we only need time. It's broadcast across channel.
    m_off = b * mask_bs + 0 * mask_cs + t_offsets * mask_ts
    m_val = tl.load(mask_ptr + m_off, mask=t_mask, other=1.0)
    y_val = y_val * m_val
    tl.store(y_ptr + y_off, y_val, mask=t_mask)


@triton.jit
def add_or_sub(y_ptr, addend_ptr, B, C, T, op_add, y_bs, y_cs, y_ts):
    # addend_ptr points to tensor of shape [B, C, T]
    pid_bc = tl.program_id(0)  # over B * C
    b = pid_bc // C
    c = pid_bc % C
    t_offsets = tl.arange(0, 128)
    t_mask = t_offsets < T
    y_off = b * y_bs + c * y_cs + t_offsets * y_ts
    y_val = tl.load(y_ptr + y_off, mask=t_mask, other=0.0)
    add_off = b * y_bs + c * y_cs + t_offsets * y_ts  # same layout as y
    add_val = tl.load(addend_ptr + add_off, mask=t_mask, other=0.0)
    if op_add:
        y_val = y_val + add_val
    else:
        y_val = y_val - add_val
    tl.store(y_ptr + y_off, y_val, mask=t_mask)


@triton.jit
def copy_to(out_ptr, src_ptr, B, C, T, out_bs, out_cs, out_ts, src_bs, src_cs, src_ts, op_add: tl.constexpr):
    # copy src to out, optionally add src to out (op_add=True) or subtract (op_add=False)
    pid_bc = tl.program_id(0)  # over B * C
    b = pid_bc // C
    c = pid_bc % C
    t_offsets = tl.arange(0, 128)
    t_mask = t_offsets < T
    out_off = b * out_bs + c * out_cs + t_offsets * out_ts
    src_off = b * src_bs + c * src_cs + t_offsets * src_ts
    out_val = tl.load(out_ptr + out_off, mask=t_mask, other=0.0)
    src_val = tl.load(src_ptr + src_off, mask=t_mask, other=0.0)
    if op_add:
        out_val = out_val + src_val
    else:
        out_val = out_val - src_val
    tl.store(out_ptr + out_off, out_val, mask=t_mask)


@triton.jit
def copy_concat_halves(out_ptr, x0_ptr, x1_ptr, B, C_half, T,
                        out_bs, out_cs, out_ts,
                        x0_bs, x0_cs, x0_ts,  # x0: [B, C_half, T]
                        x1_bs, x1_cs, x1_ts,  # x1_after: [B, C_half, T] (T may differ)
                        op_add: tl.constexpr):
    # out: [B, 2*C_half, T_out]
    # x0: [B, C_half, T], x1: [B, C_half, T]
    # We write:
    #   out[:, :C_half, :] = x0
    #   out[:, C_half:, :] = x1_after (+ or - depending on op_add)
    pid_b = tl.program_id(0)  # over B
    t_offsets = tl.arange(0, 128)
    t_mask = t_offsets < T

    # first half: channels 0..C_half-1
    for ch in range(0, C_half):
        out_off = pid_b * out_bs + ch * out_cs + t_offsets * out_ts
        x0_off = pid_b * x0_bs + ch * x0_cs + t_offsets * x0_ts
        x0_val = tl.load(x0_ptr + x0_off, mask=t_mask, other=0.0)
        tl.store(out_ptr + out_off, x0_val, mask=t_mask)

    # second half: channels C_half..2*C_half-1
    for ch in range(0, C_half):
        src_ch = ch  # we are copying x1's channels into out's second half
        out_ch = C_half + src_ch
        out_off = pid_b * out_bs + out_ch * out_cs + t_offsets * out_ts
        x1_off = pid_b * x1_bs + src_ch * x1_cs + t_offsets * x1_ts
        x1_val = tl.load(x1_ptr + x1_off, mask=t_mask, other=0.0)
        if op_add:
            out_val = tl.load(out_ptr + out_off, mask=t_mask, other=0.0) + x1_val
        else:
            out_val = tl.load(out_ptr + out_off, mask=t_mask, other=0.0) - x1_val
        tl.store(out_ptr + out_off, out_val, mask=t_mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        """
        Triton-optimized forward:
        - Inputs: x [B, 192, T], x_mask [B, 1, T], and 4 sets of weights/bias for transforms.
        - For each transform, compute 3 convs (K=5, padding=2), apply ReLU after each conv+bias,
          multiply by mask, and add to the second half x1. Finally, concatenate halves into output.
        - All math is done via Triton kernels; no torch ops in forward.
        """
        # Extract arguments: positions match get_inputs(..., *args)
        x = args[0]  # [B, 192, T]
        x_mask = args[1]  # [B, 1, T]
        reverse = args[2]  # bool
        # Build the list of transforms' params
        transforms = []
        for i in range(4):
            conv0_w = args[3 + 6 * i]  # weight for conv0
            conv0_b = args[4 + 6 * i]
            conv1_w = args[5 + 6 * i]
            conv1_b = args[6 + 6 * i]
            conv2_w = args[7 + 6 * i]
            conv2_b = args[8 + 6 * i]
            transforms.append((conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b))

        B, C, T = x.shape
        half_channels = C // 2  # 96
        # We'll produce final output y_out [B, 192, T - 12], because each conv reduces T by 1.
        # Initialize y_out with zeros (final output after all 4 transforms added)
        T_final = T - 12
        y_out = torch.zeros((B, C, T_final), device=x.device, dtype=x.dtype)

        # Process each transform sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Prepare input halves
            x0 = x[:, :half_channels, :].contiguous()
            x1 = x[:, half_channels:, :].contiguous()

            # After conv2, output has T dimension T - 3 (since conv0/1 each reduce by 1)
            # We'll update x1 with +h2 (forward) or -h2 (reverse) after applying mask and ReLU
            # conv0: x0 [B, 96, T] -> y0 [B, 192, T-1]
            Bx = B
            C_in0 = half_channels  # 96
            C_out0 = conv0_w.shape[0]  # 192
            T_in0 = T
            T_out0 = T_in0 - 1  # valid conv with padding=2 and K=5

            y0 = torch.empty((Bx, C_out0, T_out0), device=x.device, dtype=x.dtype)

            # Launch conv1d kernel for conv0
            grid0 = (Bx * C_out0, triton.cdiv(T_out0, 128))
            conv1d_k5_p2[grid0](
                x0, conv0_w, conv0_b, y0,
                Bx, C_in0, T_in0, C_out0, T_out0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_T=128,
            )

            # conv1: y0 [B, 192, T-1] -> y1 [B, 192, T-2]
            C_in1 = conv1_w.shape[1]  # 192
            C_out1 = conv1_w.shape[0]  # 192
            T_in1 = T_out0  # T - 1
            T_out1 = T_in1 - 1  # T - 2

            y1 = torch.empty((Bx, C_out1, T_out1), device=x.device, dtype=x.dtype)

            grid1 = (Bx * C_out1, triton.cdiv(T_out1, 128))
            conv1d_k5_p2[grid1](
                y0, conv1_w, conv1_b, y1,
                Bx, C_in1, T_in1, C_out1, T_out1,
                y0.stride(0), y0.stride(1), y0.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_T=128,
            )

            # conv2: y1 [B, 192, T-2] -> h2 [B, 96, T-3]
            C_in2 = conv2_w.shape[1]  # 192
            C_out2 = conv2_w.shape[0]  # 96
            T_in2 = T_out1  # T - 2
            T_out2 = T_in2 - 1  # T - 3

            h2 = torch.empty((Bx, C_out2, T_out2), device=x.device, dtype=x.dtype)

            grid2 = (Bx * C_out2, triton.cdiv(T_out2, 128))
            conv1d_k5_p2[grid2](
                y1, conv2_w, conv2_b, h2,
                Bx, C_in2, T_in2, C_out2, T_out2,
                y1.stride(0), y1.stride(1), y1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_T=128,
            )

            # Apply ReLU and mask on h2
            # ReLU
            grid_relu = (Bx * C_out2, triton.cdiv(T_out2, 128))
            relu_kernel[grid_relu](h2, Bx, C_out2, T_out2, h2.stride(0), h2.stride(1), h2.stride(2))

            # Multiply by mask (broadcast across channels)
            # x_mask shape [B, 1, T]; we only need time dimension
            grid_mask = (Bx * C_out2, triton.cdiv(T_out2, 128))
            mul_mask[grid_mask](h2, x_mask, Bx, C_out2, T_out2, h2.stride(0), h2.stride(1), h2.stride(2),
                                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2))

            # Now update x1: forward adds, reverse subtracts
            # x1 has shape [B, 96, T], we will modify it in place by adding/sub h2 (shape [B, 96, T-3])
            # We need to align time dimensions for add_or_sub. Since each conv reduces time by 1, after 3 convs,
            # h2 has T_out2 = T - 3. x1 has T, so we will add/sub only on the first T_out2 time steps.
            # To do that, we can create a zero tensor of same shape as x1, add/sub h2 (padded with zeros), or
            # directly use copy_to with addend=h2 (we'll use copy_to with addend=h2 and op_add depending on reverse).
            # However, since h2 length is T_out2, we should operate on the first T_out2 time positions only.
            # But the original code applies transform to the second half and concatenates; here we update x1's entire length T,
            # effectively adding/sub for the first T_out2 positions. Given T >= T_out2 in provided workloads, this matches.
            # For safety and correctness, we'll zero-initialize a delta tensor of shape x1 and add/sub h2 into it
            # only on first T_out2 positions; for remaining positions, keep zeros. But since we concatenate halves,
            # and final output is produced separately, we can simply add/sub x1 with h2 by copying h2 into x1 for the
            # first T_out2 positions. However, to keep the final output y_out correct (we are not concatenating here),
            # we update x1 directly by creating a new tensor x1_after = x1 + h2 (or - h2).
            # But y_out is the final output after 4 transforms added. We will compute x1_after and then add it to y_out
            # after all transforms. To do that, we need to know x1_after shape. However, forward's signature expects us to
            # return the final x (i.e., y_out), not update an external x. So, instead, we will compute x1_after and use it
            # to update y_out via copy_concat_halves in the end loop. For now, we compute h2 and its masked ReLU, and
            # produce x1_after separately for each transform.
            # For simplicity and correctness, we'll allocate x1_after as zeros of shape [B, 96, T], and add/sub h2
            # across first T_out2 positions. Given T_final will be T - 12 and each transform reduces output time
            # and final output channels are unchanged, we should not be updating x1 here. Instead, we will compute
            # x1_after and then copy to y_out via concat after all transforms.

            # Create x1_after = x1 (+ or -) h2 on first T_out2 positions
            x1_after = torch.zeros_like(x1, device=x.device, dtype=x.dtype)
            # We need to add/sub h2 into x1_after[:, :, :T_out2]
            # Launch add_or_sub kernel across B, C_half, and T_out2
            grid_add = (B, half_channels, triton.cdiv(T_out2, 128))
            add_or_sub[grid_add](x1_after, h2, B, half_channels, T_out2, not reverse, x1_after.stride(0), x1_after.stride(1), x1_after.stride(2))

            # Concatenate x0 and x1_after into a temporary output of shape [B, 192, T - 12]
            # We perform this in-place into y_out: first half channels [B, 96, T-3], second half [B, 96, T-3] (from x1_after)
            # However, since y_out is zeros [B, 192, T-12], we can directly copy into y_out by mapping:
            # y_out[:, :96, :T-3] = x0
            # y_out[:, 96:, :T-3] = x1_after (first T-3 positions)
            T_minus_three = T_out2
            grid_copy = (B, 1)
            copy_concat_halves[grid_copy](y_out, x0, x1_after, B, half_channels, T_minus_three,
                                          y_out.stride(0), y_out.stride(1), y_out.stride(2),
                                          x0.stride(0), x0.stride(1), x0.stride(2),
                                          x1_after.stride(0), x1_after.stride(1), x1_after.stride(2),
                                          op_add=True)

        # After processing all 4 transforms, y_out is the final output as per original: concatenated halves with +h
        # No further torch operations; return y_out
        return y_out


def run(*args):
    return ModelNew()(*args)
