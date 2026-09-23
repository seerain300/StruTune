import torch
import triton
import triton.language as tl


# in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
# x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H,), BCx: (B, S, 3H)
@triton.jit
def in_proj_kernel(
    x_ptr,                  # *float32, (B, S, H)
    W_ptr,                  # *float32, (M_total, H) with M_total=3*H
    b_ptr,                  # *float32, (M_total,)
    BCx_ptr,                # *float32, output (B, S, M_total)
    B: tl.constexpr,        # int
    S: tl.constexpr,        # int
    H: tl.constexpr,        # int
    M_total: tl.constexpr,  # int (3*H)
    BLOCK_M: tl.constexpr,  # tile size along M_total
):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    m_chunk = tl.program_id(2)

    m_start = m_chunk * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)  # vector [BLOCK_M]
    mask_m = m_offsets < M_total

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # loop over H (compile-time constant -> unrolled)
    for h_i in range(0, H):
        x_val = tl.load(x_ptr + b_id * (S * H) + s_id * H + h_i)
        w_ptr = W_ptr + m_offsets * H + h_i
        w_vals = tl.load(w_ptr, mask=mask_m, other=0.0)
        acc += x_val * w_vals

    bias_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)
    acc += bias_vals

    bcx_ptr = BCx_ptr + b_id * (S * M_total) + s_id * M_total + m_offsets
    tl.store(bcx_ptr, acc, mask=mask_m)


# Grouped causal conv with groups=H: conv_out[b, c, t] = sum_{k=0..3} Bx_padded[b, c, t+k] * W[c, k] + bias[c]
# Bx_pad: (B, S_pad, H), W: (H, 4), b_bias: (H,), conv_out: (B, H, S)
@triton.jit
def conv_groupsH_kernel(
    Bx_pad_ptr,        # *float32, (B, S_pad, H)
    W_ptr,             # *float32, (H, 4)
    b_bias_ptr,        # *float32, (H,)
    conv_out_ptr,      # *float32, (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
):
    b_id = tl.program_id(0)
    c_id = tl.program_id(1)  # channel c in [0..H-1]

    for t in range(0, S):
        acc = tl.zeros((), dtype=tl.float32)
        # static taps k=0..K-1
        for k in range(0, K):
            idx = t + k  # valid t in [0..S-1], idx in [k..S+k-1]
            val = tl.load(Bx_pad_ptr + b_id * ((S + K - 1) * H) + idx * H + c_id)
            w_k = tl.load(W_ptr + c_id * K + k)
            acc += val * w_k
        bias = tl.load(b_bias_ptr + c_id)
        acc += bias

        out_ptr = conv_out_ptr + b_id * (H * S) + c_id * S + t
        tl.store(out_ptr, acc)


# Final linear projection: out = y @ out_proj_weight^T + out_proj_bias
# y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
@triton.jit
def out_proj_kernel(
    y_ptr,                # *float32, (B, S, H)
    W_ptr,                # *float32, (H, H)
    b_ptr,                # *float32, (H,)
    out_ptr,              # *float32, (B, S, H)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    BLOCK_H: tl.constexpr  # tile over H for output h-dim
):
    b_id = tl.program_id(0)
    s_id = tl.program_id(1)
    h_chunk = tl.program_id(2)
    h_offsets = h_chunk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for h2 in range(0, H):
        y_val = tl.load(y_ptr + b_id * (S * H) + s_id * H + h2)
        w_vec = tl.load(W_ptr + h2 * H + h_offsets, mask=mask_h, other=0.0)
        acc += y_val * w_vec

    bias = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bias

    out_ptr_vec = out_ptr + b_id * (S * H) + s_id * H + h_offsets
    tl.store(out_ptr_vec, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the original pipeline:
        1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        2) Split BCx -> B, C, x_proj
        3) Bx = B * x_proj
        4) conv: grouped causal 1D conv with groups=H, kernel_size=4 (left-pad by 3)
        5) y = C * conv_out
        6) out_proj: final linear y -> (B, S, H)

        Note: We implement in_proj, conv, and out_proj with Triton. Elementwise gating is done with torch to
        keep code simple and reliable. The heavy ops (GEMM and conv) are Triton.
        """
        assert x.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        device = x.device
        dtype = torch.float32

        B, S, H = x.shape
        M_total = 3 * H

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        x_c = x.contiguous()
        W_c = in_proj_weight.contiguous()
        b_c = in_proj_bias.contiguous()
        BCx = torch.empty((B, S, M_total), device=device, dtype=dtype)

        BLOCK_M = 64
        grid_in = (B, S, triton.cdiv(M_total, BLOCK_M))
        in_proj_kernel[grid_in](
            x_c, W_c, b_c, BCx,
            B=B, S=S, H=H, M_total=M_total, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) Split BCx -> B, C, x_proj along last dim
        B_t = BCx[:, :, :H].contiguous()         # (B, S, H)
        C_t = BCx[:, :, H:2*H].contiguous()      # (B, S, H)
        x_proj = BCx[:, :, 2*H:].contiguous()    # (B, S, H)

        # 3) Elementwise gating: Bx = B_t * x_proj (use torch for simplicity)
        Bx = B_t * x_proj  # (B, S, H)

        # 4) Left-pad along sequence by K-1 = 3 to enforce causal conv
        K = 4
        pad_left = K - 1
        S_pad = S + pad_left
        Bx_pad = torch.empty((B, S_pad, H), device=device, dtype=dtype)

        # Manually fill Bx_pad with left pad
        # Copy Bx into Bx_pad[..., pad_left:]
        if S > 0:
            Bx_pad[:, pad_left:] = Bx

        # 5) Grouped conv: conv_out (B, H, S)
        # conv_weight: (H, 1, 4) -> flatten to (H, 4)
        W_conv = conv_weight.contiguous().view(H, 4)  # (H, 4)
        b_conv = conv_bias.contiguous()               # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=dtype)

        grid_conv = (B, H)
        conv_groupsH_kernel[grid_conv](Bx_pad, W_conv, b_conv, conv_out, B=B, S=S, H=H, K=K, num_warps=4, num_stages=2)

        # 6) Output gating: y = C_t * conv_out, elementwise
        y = C_t * conv_out  # (B, H, S)

        # 7) Transpose y to (B, S, H)
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # 8) Final linear projection using Triton
        W_out = out_proj_weight.contiguous()    # (H, H)
        b_out = out_proj_bias.contiguous()      # (H,)
        output = torch.empty((B, S, H), device=device, dtype=dtype)

        BLOCK_H = 64
        grid_out = (B, S, triton.cdiv(H, BLOCK_H))
        out_proj_kernel[grid_out](y_T, W_out, b_out, output, B=B, S=S, H=H, BLOCK_H=BLOCK_H, num_warps=4, num_stages=2)

        return output


def run(*args):
    return ModelNew()(*args)
