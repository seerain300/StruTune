import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_kernel(
    x_ptr,        # *f32, input [B, S, H]
    W_ptr,        # *f32, in_proj_weight [3H, H]
    b_ptr,        # *f32, in_proj_bias [3H]
    out_ptr,      # *f32, output [B, S, 3H]
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    m_block = tl.program_id(2)

    m_start = m_block * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < (3 * H)

    # Accumulator for this (b, s, m_block)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Sum over input features h
    for h in range(0, H):
        # x[b, s, h] linear index: ((b * S) + s) * H + h
        x_idx = ((b * S) + s) * H + h
        # Weight for each m: W[m, h] at index m_offsets * H + h
        w_vals = tl.load(W_ptr + m_offsets * H + h, mask=mask_m, other=0.0)
        x_val = tl.load(x_ptr + x_idx)
        acc += w_vals * x_val

    # Add bias
    b_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += b_vals

    # Store to out[b, s, m_offsets]
    out_idx = ((b * S) * (3 * H)) + (s * (3 * H)) + m_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_m)


@triton.jit
def elementwise_gate_kernel(
    A_ptr,  # *f32, tensor A (B, S, H)
    B_ptr,  # *f32, tensor B (B, S, H)
    C_ptr,  # *f32, output C = A * B (B, S, H)
    B_size: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    m_block = tl.program_id(2)

    m_start = m_block * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < H

    A_idx = ((b * S) * H) + s * H + m_offsets
    B_idx = A_idx  # same indexing
    C_idx = ((b * S) * H) + s * H + m_offsets

    a = tl.load(A_ptr + A_idx, mask=mask_m, other=0.0)
    b_vals = tl.load(B_ptr + B_idx, mask=mask_m, other=0.0)
    c = a * b_vals
    tl.store(C_ptr + C_idx, c, mask=mask_m)


@triton.jit
def pad_left_kernel(
    inp_ptr,      # *f32, input [B, S, H]
    out_ptr,      # *f32, output [B, S_padded, H]
    pad_left: tl.constexpr, B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    m_block = tl.program_id(2)

    m_start = m_block * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < H

    # If t < pad_left, write zeros; else write inp[b, t - pad_left, :]
    is_valid = t >= pad_left
    # Compute source index for valid case: ((b * (S - pad_left)) + (t - pad_left)) * H + m_offsets
    # We need S_out = S + pad_left; here S is original seq_len
    # Note: this kernel assumes S_out is known at host; we pass S and compute t_src accordingly
    t_src = t - pad_left
    # Compute base index
    base = ((b * (S - pad_left)) + t_src) * H
    src_idx = base + m_offsets

    # For invalid t, load zeros
    a = tl.load(inp_ptr + src_idx, mask=mask_m & is_valid, other=0.0)
    out_idx = ((b * (S + pad_left)) + t) * H + m_offsets
    tl.store(out_ptr + out_idx, a, mask=mask_m)


@triton.jit
def conv1d_groupsH_kernel(
    inp_ptr,      # *f32, input [B, H, S_padded]
    W_ptr,        # *f32, conv_weight [H, 1, 4] but passed as [H, K] contiguous
    bias_ptr,     # *f32, conv_bias [H]
    out_ptr,      # *f32, output [B, H, S]
    B: tl.constexpr, H: tl.constexpr, S_padded: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    # Each program handles one (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over output sequence positions with BLOCK_T tiles
    for t_start in range(0, S, BLOCK_T):
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < S

        # Accumulate convolution over K taps
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Load weights for this channel c (scalar per k)
        for k in range(0, K):
            w_k = tl.load(W_ptr + c * K + k)
            # inp[b, c, t + k] for all t in this tile
            pos = t_offsets + k  # sequence positions
            mask_in = mask_t & (pos < S_padded)
            # Linear index for inp: ((b * H) + c) * S_padded + pos
            base_inp = ((b * H) + c) * S_padded
            vals = tl.load(inp_ptr + base_inp + pos, mask=mask_in, other=0.0)
            acc += vals * w_k

        # Add bias
        bias_c = tl.load(bias_ptr + c)
        acc += bias_c

        # Store to out[b, c, t] = acc
        out_base = ((b * H) + c) * S
        tl.store(out_ptr + out_base + t_offsets, acc, mask=mask_t)


@triton.jit
def out_gate_kernel(
    C_ptr,        # *f32, tensor C (B, H, S)
    conv_ptr,     # *f32, conv_out (B, H, S)
    out_ptr,      # *f32, gated output (B, H, S)
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    m_block = tl.program_id(2)

    m_start = m_block * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < H  # conv_out and C have H channels

    # Indices
    base_C = ((b * H) + c) * S
    base_conv = ((b * H) + c) * S
    out_base = ((b * H) + c) * S

    C_idx = base_C + m_offsets
    conv_idx = base_conv + m_offsets
    out_idx = out_base + m_offsets

    c_vals = tl.load(C_ptr + C_idx, mask=mask_m, other=0.0)
    conv_vals = tl.load(conv_ptr + conv_idx, mask=mask_m, other=0.0)
    out_vals = c_vals * conv_vals

    tl.store(out_ptr + out_idx, out_vals, mask=mask_m)


@triton.jit
def out_proj_kernel(
    y_ptr,        # *f32, y_T (B, S, H)
    W_ptr,        # *f32, out_proj_weight (H, H)
    b_ptr,        # *f32, out_proj_bias (H)
    out_ptr,      # *f32, output (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    BLOCK_H: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)

    h_start = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Accumulate over h2 in H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for h2 in range(0, H):
        # Load y[b, s, h2]
        y_idx = ((b * S) + s) * H + h2
        y_val = tl.load(y_ptr + y_idx)

        # Load W[h2, h_offsets] where W is (H, H) row-major: W[h2, h_offsets] at indices h2*H + h_offsets
        w_idx = h2 * H + h_offsets
        w_vals = tl.load(W_ptr + w_idx, mask=mask_h, other=0.0)
        acc += w_vals * y_val

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals

    # Store to output[b, s, h_offsets]
    out_idx = ((b * S) + s) * H + h_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        device = x.device
        dtype = torch.float32

        B, S, H = x.shape
        K = conv_weight.shape[2]  # conv_kernel_size = 4
        pad_left = K - 1  # 3

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        # x is (B, S, H), in_proj_weight is (3H, H)
        in_proj_weight_c = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias_c = in_proj_bias.to(torch.float32).contiguous()
        BCx = torch.empty((B, S, 3 * H), device=device, dtype=torch.float32)

        BLOCK_M_in = 64
        grid_in = (B, S, triton.cdiv(3 * H, BLOCK_M_in))
        in_proj_kernel[grid_in](
            x.contiguous(), in_proj_weight_c, in_proj_bias_c, BCx,
            B=B, S=S, H=H, BLOCK_M=BLOCK_M_in,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B, C, x_proj: along dim=1 (channels), each size H
        # Note: BCx shape (B, S, 3H) -> chunk size=H along dim=1
        B_t = BCx[:, :, :H]
        C_t = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # 3) Element-wise gating: Bx = B_t * x_proj (B, S, H)
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        BLOCK_M_gate = 128
        grid_gate = (B, S, triton.cdiv(H, BLOCK_M_gate))
        elementwise_gate_kernel[grid_gate](
            B_t, x_proj, Bx,
            B=B, S=S, H=H, BLOCK_M=BLOCK_M_gate,
            num_warps=4, num_stages=2
        )

        # 4) Pad left by pad_left = 3 to make causal conv valid
        Bx_padded = torch.empty((B, S + pad_left, H), device=device, dtype=torch.float32)
        BLOCK_M_pad = 128
        grid_pad = (B, S + pad_left, triton.cdiv(H, BLOCK_M_pad))
        pad_left_kernel[grid_pad](
            Bx, Bx_padded, pad_left,
            B=B, S=S, H=H, BLOCK_M=BLOCK_M_pad,
            num_warps=4, num_stages=2
        )

        # 5) Grouped causal 1D convolution with groups=H (conv_weight: (H, 1, 4), conv_bias: (H,))
        conv_weight_c = conv_weight.to(torch.float32).contiguous()  # (H, 1, 4)
        conv_bias_c = conv_bias.to(torch.float32).contiguous()
        # conv weight for kernel is (H, K) contiguous
        conv_weight_K = conv_weight_c.reshape(H, K).contiguous()
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)

        BLOCK_T_conv = 64
        grid_conv = (B, H)
        conv1d_groupsH_kernel[grid_conv](
            Bx_padded, conv_weight_K, conv_bias_c, conv_out,
            B=B, H=H, S_padded=S + pad_left, S=S, K=K, BLOCK_T=BLOCK_T_conv,
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_t * conv_out, elementwise
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        BLOCK_M_gate2 = 64
        grid_gate2 = (B, H, triton.cdiv(S, BLOCK_M_gate2))
        out_gate_kernel[grid_gate2](
            C_t, conv_out, y,
            B=B, H=H, S=S, BLOCK_M=BLOCK_M_gate2,
            num_warps=4, num_stages=2
        )

        # 7) Transpose back to (B, S, H) for final linear projection
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 8) Final linear projection with out_proj
        out_proj_weight_c = out_proj_weight.to(torch.float32).contiguous()  # (H, H)
        out_proj_bias_c = out_proj_bias.to(torch.float32).contiguous()     # (H,)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        BLOCK_H_out = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H_out))
        out_proj_kernel[grid_out](
            y_T, out_proj_weight_c, out_proj_bias_c, output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H_out,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
