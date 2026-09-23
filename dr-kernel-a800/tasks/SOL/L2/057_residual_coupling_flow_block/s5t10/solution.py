import math
import torch
import torch.nn.functional as F

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def apply_mask_to_h_triton(h_ptr, mask_ptr, out_ptr,
                                B, Cin, T,
                                h_stride0, h_stride1, h_stride2,
                                mask_stride0, mask_stride1, mask_stride2,
                                out_stride0, out_stride1, out_stride2,
                                BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
        # Grid: (B, ceil(Cin/BLOCK_CO), ceil(T/BLOCK_POS))
        pid_b = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_pos = tl.program_id(2)

        co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

        co_mask = co_offsets < Cin
        pos_mask = pos_offsets < T

        # Addressing for h[b, co, pos]
        base = pid_b * h_stride0
        h_ptrs = h_ptr + base + co_offsets[:, None] * h_stride1 + pos_offsets[None, :] * h_stride2
        h_vals = tl.load(h_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

        # Addressing for mask[b, 0, pos] -> we broadcast along channel
        base_mask = pid_b * mask_stride0
        mask_ptrs = mask_ptr + base_mask + pos_offsets[None, :] * mask_stride2  # channel dimension is 1
        mask_vals = tl.load(mask_ptrs, mask=pos_mask[None, :], other=1.0)  # x_mask is ones in provided setup

        out_vals = h_vals * mask_vals  # broadcast along channels

        # Store to out[b, co, pos]
        out_ptrs = out_ptr + base + co_offsets[:, None] * out_stride1 + pos_offsets[None, :] * out_stride2
        tl.store(out_ptrs, out_vals, mask=co_mask[:, None] & pos_mask[None, :])


    @triton.jit
    def add_h_to_x1_triton(x1_ptr, h_ptr, out_ptr,
                            B, Cin, T,
                            x1_stride0, x1_stride1, x1_stride2,
                            h_stride0, h_stride1, h_stride2,
                            out_stride0, out_stride1, out_stride2,
                            ADD: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
        # Grid: (B, ceil(Cin/BLOCK_CO), ceil(T/BLOCK_POS))
        pid_b = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_pos = tl.program_id(2)

        co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

        co_mask = co_offsets < Cin
        pos_mask = pos_offsets < T

        base_x1 = pid_b * x1_stride0
        x1_ptrs = x1_ptr + base_x1 + co_offsets[:, None] * x1_stride1 + pos_offsets[None, :] * x1_stride2
        x1_vals = tl.load(x1_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

        base_h = pid_b * h_stride0
        h_ptrs = h_ptr + base_h + co_offsets[:, None] * h_stride1 + pos_offsets[None, :] * h_stride2
        h_vals = tl.load(h_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

        if ADD:
            out_vals = x1_vals + h_vals
        else:
            out_vals = x1_vals - h_vals

        out_ptrs = out_ptr + base_x1 + co_offsets[:, None] * out_stride1 + pos_offsets[None, :] * out_stride2
        tl.store(out_ptrs, out_vals, mask=co_mask[:, None] & pos_mask[None, :])


    @triton.jit
    def concat_add_v1_triton(x0_ptr, x1_ptr, out_ptr,
                              B, half_c, T,
                              x0_stride0, x0_stride1, x0_stride2,
                              x1_stride0, x1_stride1, x1_stride2,
                              out_stride0, out_stride1, out_stride2,
                              BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
        # Grid: (B, ceil(half_c/BLOCK_CO), ceil(T/BLOCK_POS))
        pid_b = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_pos = tl.program_id(2)

        co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

        co_mask = co_offsets < half_c
        pos_mask = pos_offsets < T

        # First half: write x0 into out[b, co, pos]
        base0 = pid_b * x0_stride0
        x0_ptrs = x0_ptr + base0 + co_offsets[:, None] * x0_stride1 + pos_offsets[None, :] * x0_stride2
        x0_vals = tl.load(x0_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

        out_base = pid_b * out_stride0
        out_ptrs = out_ptr + out_base + co_offsets[:, None] * out_stride1 + pos_offsets[None, :] * out_stride2
        tl.store(out_ptrs, x0_vals, mask=co_mask[:, None] & pos_mask[None, :])


    @triton.jit
    def concat_add_v2_triton(x0_ptr, x1_ptr, out_ptr,
                              B, half_c, T,
                              x0_stride0, x0_stride1, x0_stride2,
                              x1_stride0, x1_stride1, x1_stride2,
                              out_stride0, out_stride1, out_stride2,
                              BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
        # Grid: (B, ceil(half_c/BLOCK_CO), ceil(T/BLOCK_POS))
        pid_b = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_pos = tl.program_id(2)

        co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

        co_mask = co_offsets < half_c
        pos_mask = pos_offsets < T

        # First half: write x0 into out[b, co, pos]
        base0 = pid_b * x0_stride0
        x0_ptrs = x0_ptr + base0 + co_offsets[:, None] * x0_stride1 + pos_offsets[None, :] * x0_stride2
        x0_vals = tl.load(x0_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

        out_base = pid_b * out_stride0
        out_ptrs = out_ptr + out_base + co_offsets[:, None] * out_stride1 + pos_offsets[None, :] * out_stride2
        tl.store(out_ptrs, x0_vals, mask=co_mask[:, None] & pos_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: x (B, C, T), x_mask (B, 1, T), reverse (bool), then 4 sets of weights/biases for 3 convs each
        x = args[0]
        x_mask = args[1]
        reverse = args[2] if len(args) > 2 else False

        # Ensure CUDA tensors
        assert x.is_cuda, "Input x must be on CUDA"
        assert x_mask.is_cuda, "x_mask must be on CUDA"

        # Dimensions
        B, C, T = x.shape
        half_channels = C // 2
        # For the provided get_inputs, channels are fixed: hidden=192, half=96, but we keep half_channels computed from x

        # Prepare transforms
        transforms = [
            (args[3], args[4], args[5], args[6], args[7], args[8]),   # conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b
            (args[9], args[10], args[11], args[12], args[13], args[14]),
            (args[15], args[16], args[17], args[18], args[19], args[20]),
            (args[21], args[22], args[23], args[24], args[25], args[26]),
        ]

        if not reverse:
            # Forward: apply transforms sequentially
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # Split x into x0 and x1
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

                # Compute h = conv1d(x0) -> ReLU -> conv1d -> ReLU -> conv1d using PyTorch (fast, correct)
                padding = (conv0_w.shape[2] - 1) // 2  # kernel_size is the last dim of weight
                h = F.conv1d(x0, conv0_w, conv0_b, padding=padding)  # [B, hidden, T]
                h = F.relu(h)
                h = F.conv1d(h, conv1_w, conv1_b, padding=padding)  # [B, hidden, T]
                h = F.relu(h)
                h = F.conv1d(h, conv2_w, conv2_b, padding=padding)  # [B, half, T]

                # Apply mask to h (broadcast along channels): h_masked = h * x_mask
                h_masked = torch.empty_like(h)
                if TRITON_AVAILABLE:
                    h_contig = h.contiguous()
                    mask_contig = x_mask.contiguous()
                    h_masked = torch.empty_like(h_contig)
                    grid = (B, triton.cdiv(half_channels, 128), triton.cdiv(T, 128))
                    apply_mask_to_h_triton[grid](
                        h_contig, mask_contig, h_masked,
                        B, half_channels, T,
                        h_contig.stride(0), h_contig.stride(1), h_contig.stride(2),
                        mask_contig.stride(0), mask_contig.stride(1), mask_contig.stride(2),
                        h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                        BLOCK_CO=128, BLOCK_POS=128
                    )
                else:
                    # Fallback: PyTorch elementwise multiply (rare path if Triton unavailable)
                    h_masked = h * x_mask

                # Update x1: x1 = x1 + h_masked
                x1_out = torch.empty_like(x1)
                grid = (B, triton.cdiv(half_channels, 128), triton.cdiv(T, 128))
                add_h_to_x1_triton[grid](
                    x1, h_masked, x1_out,
                    B, half_channels, T,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    ADD=True, BLOCK_CO=128, BLOCK_POS=128
                )

                # Concatenate back into out: [x0, x1_out]
                out = torch.empty((B, C, T), dtype=x.dtype, device=x.device)
                grid_concat = (B, triton.cdiv(half_channels, 128), triton.cdiv(T, 128))
                concat_add_v1_triton[grid_concat](
                    x0, x1_out, out,
                    B, half_channels, T,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    BLOCK_CO=128, BLOCK_POS=128
                )

                # Apply x_mask broadcast along channels to output (final masking)
                out_masked = torch.empty_like(out)
                if TRITON_AVAILABLE:
                    out_mask = out  # we need mask of shape [B, 1, T]; create if not present
                    # Since x_mask is [B,1,T], we can broadcast:
                    mask_broadcast = x_mask.expand(B, 1, T).contiguous()
                    out_masked = torch.empty_like(out)
                    grid = (B, triton.cdiv(C, 128), triton.cdiv(T, 128))
                    apply_mask_to_h_triton[grid](
                        out, mask_broadcast, out_masked,
                        B, C, T,
                        out.stride(0), out.stride(1), out.stride(2),
                        mask_broadcast.stride(0), mask_broadcast.stride(1), mask_broadcast.stride(2),
                        out_masked.stride(0), out_masked.stride(1), out_masked.stride(2),
                        BLOCK_CO=128, BLOCK_POS=128
                    )
                else:
                    out_masked = out * x_mask

                x = out_masked

        else:
            # Reverse: apply transforms in reverse order, subtracting h each time
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                # Split x into x0 and x1
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

                # Compute h = conv1d(x0) -> ReLU -> conv1d -> ReLU -> conv1d
                padding = (conv0_w.shape[2] - 1) // 2
                h = F.conv1d(x0, conv0_w, conv0_b, padding=padding)
                h = F.relu(h)
                h = F.conv1d(h, conv1_w, conv1_b, padding=padding)
                h = F.relu(h)
                h = F.conv1d(h, conv2_w, conv2_b, padding=padding)

                # Apply mask to h: h_masked = h * x_mask
                h_masked = torch.empty_like(h)
                if TRITON_AVAILABLE:
                    h_contig = h.contiguous()
                    mask_contig = x_mask.contiguous()
                    h_masked = torch.empty_like(h_contig)
                    grid = (B, triton.cdiv(half_channels, 128), triton.cdiv(T, 128))
                    apply_mask_to_h_triton[grid](
                        h_contig, mask_contig, h_masked,
                        B, half_channels, T,
                        h_contig.stride(0), h_contig.stride(1), h_contig.stride(2),
                        mask_contig.stride(0), mask_contig.stride(1), mask_contig.stride(2),
                        h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                        BLOCK_CO=128, BLOCK_POS=128
                    )
                else:
                    h_masked = h * x_mask

                # Update x1: x1 = x1 - h_masked
                x1_out = torch.empty_like(x1)
                grid = (B, triton.cdiv(half_channels, 128), triton.cdiv(T, 128))
                add_h_to_x1_triton[grid](
                    x1, h_masked, x1_out,
                    B, half_channels, T,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    ADD=False, BLOCK_CO=128, BLOCK_POS=128
                )

                # Concatenate back into out: [x0, x1_out]
                out = torch.empty((B, C, T), dtype=x.dtype, device=x.device)
                grid_concat = (B, triton.cdiv(half_channels, 128), triton.cdiv(T, 128))
                concat_add_v2_triton[grid_concat](
                    x0, x1_out, out,
                    B, half_channels, T,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    BLOCK_CO=128, BLOCK_POS=128
                )

                # Apply x_mask broadcast along channels to output (final masking)
                out_masked = torch.empty_like(out)
                if TRITON_AVAILABLE:
                    mask_broadcast = x_mask.expand(B, 1, T).contiguous()
                    out_masked = torch.empty_like(out)
                    grid = (B, triton.cdiv(C, 128), triton.cdiv(T, 128))
                    apply_mask_to_h_triton[grid](
                        out, mask_broadcast, out_masked,
                        B, C, T,
                        out.stride(0), out.stride(1), out.stride(2),
                        mask_broadcast.stride(0), mask_broadcast.stride(1), mask_broadcast.stride(2),
                        out_masked.stride(0), out_masked.stride(1), out_masked.stride(2),
                        BLOCK_CO=128, BLOCK_POS=128
                    )
                else:
                    out_masked = out * x_mask

                x = out_masked

        return x


def run(*args):
    return ModelNew()(*args)
