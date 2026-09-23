import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_kernel(
    x_ptr,                # *float32, input x: (B, S, H), contiguous
    in_proj_w_ptr,        # *float32, in_proj_weight: (3H, H), contiguous
    in_proj_b_ptr,        # *float32, in_proj_bias: (3H,), contiguous
    bcx_ptr,              # *float32, output BCx: (B, S, 3H), contiguous
    B: tl.constexpr,      # batch size
    S: tl.constexpr,      # sequence length
    H: tl.constexpr,      # hidden size
    M: tl.constexpr,      # M = 3H
    BLOCK_M: tl.constexpr # tile for M dimension
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    m_block = tl.program_id(2)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # Accumulator for BCx[b, s, m_offsets]
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over h in [0..H) — H is constexpr, so this is a static loop
    for h in range(0, H):
        # x[b, s, h] -> base index = ((b * S) + s) * H + h
        x_idx = ((b * S) + s) * H + h
        x_val = tl.load(x_ptr + x_idx, mask=True, other=0.0)  # scalar
        # in_proj_w[m, h] -> index = m_offsets * H + h
        w_idx = m_offsets * H + h
        w_val = tl.load(in_proj_w_ptr + w_idx, mask=mask_m, other=0.0)
        acc += x_val * w_val  # broadcast multiply

    # Add bias
    bias = tl.load(in_proj_b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += bias

    # Store to BCx[b, s, m_offsets] -> index = ((b * S + s) * M) + m_offsets
    out_idx = ((b * S + s) * M) + m_offsets
    tl.store(bcx_ptr + out_idx, acc, mask=mask_m)


@triton.jit
def pad_left_kernel(
    bx_ptr,               # *float32, input Bx: (B, S, H), contiguous
    bx_pad_ptr,           # *float32, output Bx_padded: (B, S + pad_left, H), contiguous
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    pad_left: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # t in [0..S + pad_left)
    c = tl.program_id(2)  # channel index (unused here since input has H channels)
    # We launch grid=(B, S + pad_left, H). For each (b, t, h), either write 0 or copy from Bx.
    h = c  # since c is actually the channel index, which equals h
    if t < pad_left:
        # pad left: write zeros
        tl.store(bx_pad_ptr + ((b * (S + pad_left)) + t) * H + h, 0.0)
    else:
        src_t = t - pad_left
        src_idx = ((b * S) + src_t) * H + h
        val = tl.load(bx_ptr + src_idx)
        tl.store(bx_pad_ptr + ((b * (S + pad_left)) + t) * H + h, val)


@triton.jit
def out_proj_kernel(
    y_ptr,                # *float32, input y: (B, S, H), contiguous
    out_w_ptr,            # *float32, out_proj_weight: (H, H), contiguous
    out_b_ptr,            # *float32, out_proj_bias: (H,), contiguous
    out_ptr,              # *float32, output: (B, S, H), contiguous
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Accumulate output[b, s, h_offsets]
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # y[b, s, h_offsets] -> idx = ((b * S) + s) * H + h_offsets
    y_idx = ((b * S) + s) * H + h_offsets
    y_vals = tl.load(y_ptr + y_idx, mask=mask_h, other=0.0)

    # out_w[h_offsets, h_offsets] -> idx = h_offsets[:, None] * H + h_offsets[None, :]
    for h2 in range(0, H):
        w_idx = h_offsets * H + h2  # vector of length BLOCK_H
        w_vals = tl.load(out_w_ptr + w_idx, mask=mask_h, other=0.0)
        acc += y_vals[:, None] * w_vals[None, :]

    # Add bias
    b_vals = tl.load(out_b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += b_vals[None, :]

    # Store to out[b, s, h_offsets]
    out_idx = ((b * S) + s) * H + h_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_h)


@triton.jit
def conv_groupsH_kernel(
    bx_pad_ptr,           # *float32, Bx_padded: (B, S_padded, H), contiguous
    conv_w_ptr,           # *float32, conv_weight: (H, 1, 4), contiguous (we pass as (H, 4))
    conv_b_ptr,           # *float32, conv_bias: (H,), contiguous
    conv_out_ptr,         # *float32, output conv_out: (B, H, S), contiguous
    B: tl.constexpr,
    H: tl.constexpr,
    S_padded: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr        # kernel_size (here 4)
):
    # Grid: (B, H, S). Each program handles one (b, c, t).
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)

    # Compute conv_out[b, c, t] = sum_{k=0..K-1} bx_pad[b, c, t + k] * conv_w[c, k] + conv_b[c]
    acc = 0.0
    # Note: we rely on bx_pad_ptr zeroing for t+k >= S_padded via host pad; here S_padded is S+pad_left
    # but our grid only runs t in [0..S-1], so t+k < S_padded is always true for k in [0..K-1].
    for k in range(0, K):
        src_t = t + k
        val = tl.load(bx_pad_ptr + ((b * S_padded) + src_t) * H + c)
        w_k = tl.load(conv_w_ptr + c * K + k)  # conv_w is flattened to (H, K)
        acc += val * w_k

    bias_c = tl.load(conv_b_ptr + c)
    acc += bias_c

    # Store conv_out[b, c, t] -> index = ((b * H + c) * S) + t
    out_idx = ((b * H + c) * S) + t
    tl.store(conv_out_ptr + out_idx, acc)


def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    """
    Triton-only fused computation:
    1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
    2) Split: B = BCx[:, :, :H], C = BCx[:, :, H:2H], x_proj = BCx[:, :, 2H:3H]
    3) Gating: Bx = B * x_proj
    4) Pad left by K-1 (K=4 -> pad=3)
    5) Grouped causal conv with groups=H, kernel_size=4
    6) Output gating: y = C * conv_out
    7) Final linear: y -> (B, S, H)
    """
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    device = x.device
    dtype = torch.float32

    # Ensure contiguity and dtype
    x_c = x.contiguous().to(dtype)
    B, S, H = x_c.shape
    M = 3 * H

    # 1) in_proj: BCx
    BCx = torch.empty((B, S, M), device=device, dtype=dtype)
    BLOCK_M = 64
    grid_in = (B, S, triton.cdiv(M, BLOCK_M))
    in_proj_kernel[grid_in](
        x_c, in_proj_weight.contiguous().to(dtype), in_proj_bias.contiguous().to(dtype),
        BCx,
        B=B, S=S, H=H, M=M, BLOCK_M=BLOCK_M,
        num_warps=4, num_stages=2
    )

    # 2) Split
    B_t = BCx[:, :, :H].contiguous()   # (B, S, H)
    C_t = BCx[:, :, H:2 * H].contiguous()  # (B, S, H)
    x_proj = BCx[:, :, 2 * H:].contiguous()  # (B, S, H)

    # 3) Gating
    Bx = (B_t * x_proj).contiguous()   # (B, S, H)

    # 4) Pad left by 3
    pad_left = 3
    S_padded = S + pad_left
    Bx_pad = torch.empty((B, S_padded, H), device=device, dtype=dtype)
    grid_pad = (B, S_padded, H)
    pad_left_kernel[grid_pad](
        Bx, Bx_pad,
        B=B, S=S, H=H, pad_left=pad_left,
        num_warps=2, num_stages=1
    )

    # 5) Grouped causal conv (groups=H, kernel_size=4), output (B, H, S)
    conv_w_flat = conv_weight.contiguous().to(dtype).reshape(H, conv_weight.shape[2])  # (H, 4) since kernel_size=4
    conv_out = torch.empty((B, H, S), device=device, dtype=dtype)
    grid_conv = (B, H, S)
    conv_groupsH_kernel[grid_conv](
        Bx_pad, conv_w_flat, conv_bias.contiguous().to(dtype), conv_out,
        B=B, H=H, S_padded=S_padded, S=S, K=4,
        num_warps=2, num_stages=1
    )

    # 6) Output gating: y = C_t * conv_out
    # conv_out: (B, H, S), C_t: (B, S, H)
    # We need to align dims for elementwise multiply. Trick: transpose conv_out to (B, S, H) then multiply.
    conv_out_T = conv_out.transpose(1, 2).contiguous()  # (B, S, H)
    y = (C_t * conv_out_T).contiguous()  # (B, S, H)

    # 7) Final linear: y -> (B, S, H) using out_proj_weight (H, H), out_proj_bias (H,)
    output = torch.empty((B, S, H), device=device, dtype=dtype)
    BLOCK_H = 64
    grid_out = (B, S, triton.cdiv(H, BLOCK_H))
    out_proj_kernel[grid_out](
        y, out_proj_weight.contiguous().to(dtype), out_proj_bias.contiguous().to(dtype), output,
        B=B, S=S, H=H, BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=2
    )

    return output


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        return run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)


def run(*args):
    return ModelNew()(*args)
