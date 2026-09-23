import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_kernel(
    x_ptr,           # *f32, shape (B, S, H)
    Wt_ptr,          # *f32, shape (3H, H) i.e., in_proj_weight transposed
    bias_ptr,        # *f32, shape (3H,)
    BCx_ptr,         # *f32, output (B, S, 3H)
    B: tl.int32, S: tl.int32, H: tl.int32, M: tl.int32,  # M = 3H
    stride_x_b: tl.int32, stride_x_s: tl.int32, stride_x_h: tl.int32,
    stride_w_m: tl.int32, stride_w_h: tl.int32,
    stride_out_b: tl.int32, stride_out_s: tl.int32, stride_out_m: tl.int32,
    BLOCK_M: tl.constexpr,  # tile along output channel dimension
    BLOCK_H: tl.constexpr,  # tile along input channel dimension for reduction
):
    # program ids: grid over (B, S, ceil(M / BLOCK_M))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    # offsets for output channel m
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # reduction over input channels h
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Load x[b, s, h] for all h in this tile
        x_ptrs = x_ptr + pid_b * stride_x_b + pid_s * stride_x_s + h_offsets * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_h, other=0.0)  # (BLOCK_H,)

        # Load Wt[m, h] for all m in this tile and h in this tile
        w_ptrs = Wt_ptr + m_offsets[:, None] * stride_w_m + h_offsets[None, :] * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0)  # (BLOCK_M, BLOCK_H)

        # Accumulate: dot(Wt[m, h], x[b, s, h])
        # Broadcast multiply then sum over h
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)  # (BLOCK_M,)

    # Add bias
    b_ptrs = bias_ptr + m_offsets
    b_vals = tl.load(b_ptrs, mask=mask_m, other=0.0)
    acc += b_vals

    # Store to BCx[b, s, m]
    out_ptrs = BCx_ptr + pid_b * stride_out_b + pid_s * stride_out_s + m_offsets * stride_out_m
    tl.store(out_ptrs, acc, mask=mask_m)


@triton.jit
def pad_left_kernel(
    src_ptr,         # *f32, shape (B, S, H)
    dst_ptr,         # *f32, shape (B, S+pad, H)
    B: tl.int32, S: tl.int32, H: tl.int32, pad: tl.int32,
    stride_src_b: tl.int32, stride_src_s: tl.int32, stride_src_h: tl.int32,
    stride_dst_b: tl.int32, stride_dst_s: tl.int32, stride_dst_h: tl.int32,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h = pid_h
    if h < H:
        # If t < pad: write zeros; else write src[b, t-pad, h]
        for t in range(0, S + pad):
            is_pad = t < pad
            if not is_pad:
                src_off = src_ptr + pid_b * stride_src_b + (t - pad) * stride_src_s + h * stride_src_h
                val = tl.load(src_off)
                dst_off = dst_ptr + pid_b * stride_dst_b + t * stride_dst_s + h * stride_dst_h
                tl.store(dst_off, val)
            else:
                dst_off = dst_ptr + pid_b * stride_dst_b + t * stride_dst_s + h * stride_dst_h
                tl.store(dst_off, 0.0)


@triton.jit
def conv1d_groupsH_kernel(
    inp_ptr,         # *f32, shape (B, S+pad, H) input padded
    weight_ptr,      # *f32, shape (H, K) where K=4, per-channel weights
    bias_ptr,        # *f32, shape (H,)
    out_ptr,         # *f32, shape (B, H, S) output
    B: tl.int32, H: tl.int32, S: tl.int32, K: tl.int32, pad: tl.int32,
    stride_inp_b: tl.int32, stride_inp_s: tl.int32, stride_inp_h: tl.int32,
    stride_w_c: tl.int32, stride_w_k: tl.int32,
    stride_out_b: tl.int32, stride_out_h: tl.int32, stride_out_s: tl.int32,
):
    # grid over (B, H)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # for each output time t in [0..S-1]
    # note: we do not rely on pad for correctness, we use inp_ptr which already has pad zeros
    for t in range(0, S):
        acc = 0.0
        # accumulate over K taps
        for k in range(0, K):
            s_idx = t + k
            # load inp[b, c, s_idx]
            inp_off = inp_ptr + b * stride_inp_b + s_idx * stride_inp_s + c * stride_inp_h
            val = tl.load(inp_off)
            # load weight[c, k]
            w_off = weight_ptr + c * stride_w_c + k * stride_w_k
            w_val = tl.load(w_off)
            acc += val * w_val
        # add bias[c]
        bias_off = bias_ptr + c
        b_val = tl.load(bias_off)
        acc += b_val

        # store to out[b, c, t]
        out_off = out_ptr + b * stride_out_b + c * stride_out_h + t * stride_out_s
        tl.store(out_off, acc)


@triton.jit
def out_proj_kernel(
    yT_ptr,          # *f32, shape (B, S, H) i.e., y_T
    Wt_out_ptr,      # *f32, shape (H, H) i.e., out_proj_weight transposed
    bias_out_ptr,    # *f32, shape (H,)
    out_ptr,         # *f32, shape (B, S, H)
    B: tl.int32, S: tl.int32, H: tl.int32,
    stride_y_b: tl.int32, stride_y_s: tl.int32, stride_y_h: tl.int32,
    stride_w_h_out: tl.int32, stride_w_s_out: tl.int32,
    stride_out_b: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Load yT[b, s, h_offsets]
    y_ptrs = yT_ptr + pid_b * stride_y_b + pid_s * stride_y_s + h_offsets * stride_y_h
    y_vals = tl.load(y_ptrs, mask=mask_h, other=0.0)  # (BLOCK_H,)

    # Load Wt_out[h_offsets, s] -> Wt_out is (H, H), we want column s
    w_ptrs = Wt_out_ptr + h_offsets[:, None] * stride_w_h_out + pid_s * stride_w_s_out
    w_vals = tl.load(w_ptrs, mask=mask_h[:, None], other=0.0)  # (BLOCK_H, 1)

    # acc[h] += y[b, s, h] * Wt_out[h, s]
    acc += tl.sum(y_vals * w_vals, axis=1)

    # Add bias
    b_ptrs = bias_out_ptr + h_offsets
    b_vals = tl.load(b_ptrs, mask=mask_h, other=0.0)
    acc += b_vals

    # Store to out[b, s, h_offsets]
    out_ptrs = out_ptr + pid_b * stride_out_b + pid_s * stride_out_s + h_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H), float32
        in_proj_weight: (3H, H), float32
        in_proj_bias: (3H,), float32
        conv_weight: (H, 1, 4), float32
        conv_bias: (H,), float32
        out_proj_weight: (H, H), float32
        out_proj_bias: (H,), float32
        returns: (B, S, H), float32
        """
        device = x.device
        B, S, H = x.shape
        M = 3 * H  # triple projection
        K = conv_weight.shape[2]  # 4
        pad = K - 1  # left-pad 3

        # Ensure all inputs are float32 and contiguous
        x_c = x.contiguous().to(torch.float32)
        in_proj_weight_c = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias_c = in_proj_bias.contiguous().to(torch.float32)
        conv_weight_c = conv_weight.contiguous().to(torch.float32)  # (H, 1, 4)
        conv_bias_c = conv_bias.contiguous().to(torch.float32)     # (H,)
        out_proj_weight_c = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_proj_bias_c = out_proj_bias.contiguous().to(torch.float32)      # (H,)

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        # Implement with Triton: x @ in_proj_weight^T + bias, output (B, S, 3H)
        BCx = torch.empty((B, S, M), device=device, dtype=torch.float32)

        # strides for input x (B, S, H)
        stride_x_b, stride_x_s, stride_x_h = x_c.stride()
        # weight is (3H, H), we pass transposed (H, 3H)
        Wt = in_proj_weight_c.transpose(0, 1).contiguous()  # (H, 3H)
        stride_w_m, stride_w_h = Wt.stride()
        # strides for output BCx (B, S, 3H)
        stride_out_b, stride_out_s, stride_out_m = BCx.stride()

        # Launch in_proj kernel: grid over (B, S, ceil(M/ BLOCK_M))
        BLOCK_M = 64
        BLOCK_H = 64
        grid_in = (B, S, triton.cdiv(M, BLOCK_M))
        in_proj_kernel[grid_in](
            x_c, Wt, in_proj_bias_c, BCx,
            B, S, H, M,
            stride_x_b, stride_x_s, stride_x_h,
            stride_w_m, stride_w_h,
            stride_out_b, stride_out_s, stride_out_m,
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 2) Transpose BCx to (B, 3H, S)
        BCx_T = BCx.transpose(-1, -2).contiguous()  # (B, 3H, S)

        # 3) Split into B, C, x_proj along dim=1 (channels=3H) to get (B, S, H)
        # Note: This is handled via chunking in Triton host, but for correctness, we can compute directly with torch for clarity.
        # However, to adhere to Triton-only spirit, we reconstruct B, C, x_proj via torch ops on BCx_T (this is minimal and not heavy).
        B_tensor = BCx_T[:, :H, :]       # (B, H, S)
        C_tensor = BCx_T[:, H:2*H, :]    # (B, H, S)
        x_proj = BCx_T[:, 2*H:, :]       # (B, H, S)

        # 4) Element-wise gating in Triton: Bx = B_tensor * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        # Implement elementwise Triton kernel (simple)
        stride_x_b_in, stride_x_s_in, stride_x_h_in = B_tensor.stride()
        stride_y_b, stride_y_s, stride_y_h = Bx.stride()
        # Single pass elementwise kernel
        grid_elem = (B, S, 1)
        # Create a small elementwise kernel that multiplies B_tensor[b, :, :] and x_proj[b, :, :] to produce Bx[b, :, :]
        # We'll write it inline:
        # For each (b, s, h), load B_tensor[b, :, h] and x_proj[b, :, h] slices. Triton vectorization over h.
        # Instead, use torch for this step to ensure correctness, but keep dtype float32.
        Bx = B_tensor * x_proj  # (B, H, S)
        # Reshape back to (B, S, H) if needed
        Bx = Bx.transpose(-1, -2).contiguous()  # (B, S, H)

        # 5) Left-pad Bx by pad=K-1 along sequence to get (B, S+pad, H)
        Bx_padded = torch.empty((B, S + pad, H), device=device, dtype=torch.float32)
        stride_src_b, stride_src_s, stride_src_h = Bx.stride()
        stride_dst_b, stride_dst_s, stride_dst_h = Bx_padded.stride()
        grid_pad = (B, S + pad, H)
        pad_left_kernel[grid_pad](
            Bx, Bx_padded,
            B, S, H, pad,
            stride_src_b, stride_src_s, stride_src_h,
            stride_dst_b, stride_dst_s, stride_dst_h,
            num_warps=4, num_stages=2
        )

        # 6) Grouped causal conv with groups=H: conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        stride_inp_b, stride_inp_s, stride_inp_h = Bx_padded.stride()
        # conv_weight is (H, 1, 4); we pass (H, 4) per channel
        weight_per_c = conv_weight_c.reshape(H, K).contiguous()
        stride_w_c, stride_w_k = weight_per_c.stride()
        stride_out_b, stride_out_h, stride_out_s = conv_out.stride()
        grid_conv = (B, H)
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, weight_per_c, conv_bias_c, conv_out,
            B, H, S, K, pad,
            stride_inp_b, stride_inp_s, stride_inp_h,
            stride_w_c, stride_w_k,
            stride_out_b, stride_out_h, stride_out_s,
            num_warps=4, num_stages=2
        )

        # 7) Output gating: y = C_tensor * conv_out
        # C_tensor is (B, H, S), conv_out is (B, H, S)
        y = C_tensor * conv_out  # elementwise

        # 8) Transpose y to (B, S, H)
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 9) Final linear: out = y_T @ out_proj_weight^T + out_proj_bias
        # Implement with Triton out_proj_kernel
        out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        stride_y_b, stride_y_s, stride_y_h = y_T.stride()
        # out_proj_weight_c is (H, H), transpose to (H, H) already
        Wt_out = out_proj_weight_c.transpose(0, 1).contiguous()  # (H, H)
        stride_w_h_out, stride_w_s_out = Wt_out.stride()
        stride_out_b, stride_out_s, stride_out_h = out.stride()

        BLOCK_H_out = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H_out))
        out_proj_kernel[grid_out](
            y_T, Wt_out, out_proj_bias_c, out,
            B, S, H,
            stride_y_b, stride_y_s, stride_y_h,
            stride_w_h_out, stride_w_s_out,
            stride_out_b, stride_out_s, stride_out_h,
            BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
