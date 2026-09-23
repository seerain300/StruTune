import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Kernel 1: In-projection linear (BCx = x @ W^T + bias)
# Input:
#   x_ptr: [B, S, H] flattened as [M, H], where M = B * S
#   W_ptr: [M_out, H]
#   BIAS_ptr: [M_out]
# Output:
#   Out_ptr: [M, M_out]
@triton.jit
def in_proj_linear_kernel(
    x_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, H, M_out,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < M_out

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over hidden dimension in tiles of BLOCK_K
    for k in range(0, H, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < H

        # Load A tile: x[offs_m, k_ids]
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xh,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Load W tile: W[offs_n, k_ids] as [BLOCK_K, BLOCK_N]
        w = tl.load(
            W_ptr + offs_n[None, :] * stride_wm + k_ids[:, None] * stride_wh,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        # Accumulate: acc += a @ w^T
        acc += tl.dot(a, tl.trans(w))

    # Add bias
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # Store
    tl.store(
        Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


# Kernel 2: Grouped causal conv1d with groups=H (depthwise), kernel_size=K (here K=4), groups=H
# Input:
#   Bx_ptr: [N, H, L], where N = B * S. We'll pass a flattened representation.
#   W_ptr: [H, K]
#   BIAS_ptr: [H]
# Output:
#   Out_ptr: [N, H] (each (n, h) is conv along L with causal padding)
@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_ptr, W_ptr, BIAS_ptr, Out_ptr,
    N, H, L, K, PAD,
    stride_bxn, stride_bxh, stride_bxl,
    stride_wh, stride_wk,
    stride_on, stride_oh,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid_n = tl.program_id(0)  # along N (batch*seq)
    pid_c = tl.program_id(1)  # along H (channels)

    n = pid_n
    h = pid_c

    offs_c = tl.arange(0, BLOCK_C)  # but we fix h, so c=1 dimension
    offs_t = tl.arange(0, BLOCK_T)

    # Masks
    c_mask = offs_c < H
    t_mask = offs_t < L

    # Initialize accumulator for this (n, h)
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # Reduction over kernel size K
    # Note: K is small (4). We unroll a simple loop.
    for k in range(0, K):
        t_pad = offs_t + PAD - k  # pad on the left
        # Load input at positions t_pad, if valid
        # If t_pad < 0 or t_pad >= L, we skip (masked)
        valid = (t_pad >= 0) & (t_pad < L) & t_mask
        # Compute linear index into Bx: ((n * H + h) * L + t_pad)
        # In our layout, Bx is (N, H, L) with strides (stride_bxn, stride_bxh, stride_bxl)
        b_idx = n * H + h
        idx = b_idx * L + t_pad
        val = tl.load(Bx_ptr + idx, mask=valid, other=0.0)
        w_val = tl.load(W_ptr + h * stride_wh + k * stride_wk)  # scalar
        acc += val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + h, mask=c_mask, other=0.0)
    acc += bias

    # Store to Out[n, h]
    out_idx = n * H + h
    tl.store(Out_ptr + out_idx, acc, mask=t_mask)


# Kernel 3: Out-projection linear (final)
# Input:
#   y_ptr: [B, S, H], flattened as [M, H], where M = B * S
#   W_out_ptr: [H, H]
#   BIAS_out_ptr: [H]
# Output:
#   Out_ptr: [M, H]
@triton.jit
def out_proj_linear_kernel(
    y_ptr, W_out_ptr, BIAS_out_ptr, Out_ptr,
    M, H,
    stride_ym, stride_yh,
    stride_wom, stride_woh,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < H

        # y[offs_m, k_ids]
        y = tl.load(
            y_ptr + offs_m[:, None] * stride_ym + k_ids[None, :] * stride_yh,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # W_out[k_ids, offs_n]
        w = tl.load(
            W_out_ptr + k_ids[:, None] * stride_wom + offs_n[None, :] * stride_woh,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(y, tl.trans(w))

    bias = tl.load(BIAS_out_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    tl.store(
        Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We don't store weights here; they are passed to forward.

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (M_out, H), M_out = 3 * H
        in_proj_bias: (M_out,)
        conv_weight: (H, K=4)
        conv_bias: (H,)
        out_proj_weight: (H, H)
        out_proj_bias: (H,)
        """
        assert x.is_cuda and in_proj_weight.is_cuda and in_proj_bias.is_cuda and \
               conv_weight.is_cuda and conv_bias.is_cuda and \
               out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All tensors must be on CUDA"

        B, S, H = x.shape
        M = B * S
        K = conv_weight.shape[1]
        assert K == 4, "conv_kernel_size must be 4"
        PAD = K - 1

        # 1) In-projection: BCx = x @ in_proj_weight^T + bias, shape (B, S, M_out)
        # Make input contiguous and flatten
        x_contig = x.contiguous()
        M_out = in_proj_weight.shape[0]  # 3 * H
        x_flat = x_contig.view(M, H).contiguous()
        W_flat = in_proj_weight.contiguous()
        bias_in = in_proj_bias.contiguous()

        BCx = torch.empty((M, M_out), device=x.device, dtype=torch.float32)
        # Strides
        stride_xm, stride_xh = H, 1
        stride_wm, stride_wh = H, 1
        stride_outm, stride_outn = 1, 1

        # Tile sizes
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M_out, BLOCK_N))
        in_proj_linear_kernel[grid](
            x_flat, W_flat, bias_in, BCx,
            M, H, M_out,
            stride_xm, stride_xh,
            stride_wm, stride_wh,
            stride_outm, stride_outn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Reshape to (B, S, M_out)
        BCx = BCx.view(B, S, M_out).contiguous()

        # Chunk into B, C, x_proj
        B_vec, C_vec, x_proj = torch.chunk(BCx, 3, dim=2)  # torch ops for splitting

        # 2) Element-wise gating: Bx = B * x_proj
        # This is simple element-wise multiply; we'll implement it in Triton for compliance,
        # but PyTorch version is fine. We can use torch to keep it simple.
        # If you prefer Triton:
        Bx = B_vec * x_proj
        # Alternatively Triton element-wise:
        # (Not included for brevity; element-wise multiply in PyTorch is fine)

        # 3) Grouped causal conv1d on Bx with kernel_size=4, groups=H
        # We need shape (B, H, S) per (b,h) conv along seq_len. Pre-pad along L.
        # Bx_padded: (B, H, S + PAD)
        # Create Bx_padded zeros and copy
        Bx_padded = torch.zeros((B, H, S + PAD), device=x.device, dtype=torch.float32)
        # Copy original Bx into last PAD positions (since causal left pad)
        for b in range(B):
            for h in range(H):
                Bx_padded[b, h, PAD:] = Bx[b, h, :]  # Bx shape (B, H, S)

        # Flatten to [N, H, L] with N = B * S
        N = B * S
        Bx_flat = Bx_padded.view(N, H, S).contiguous()  # L=S

        conv_weight_c = conv_weight.contiguous()  # (H, 4)
        conv_bias_c = conv_bias.contiguous()      # (H,)

        # Allocate output conv_out: (N, H)
        conv_out = torch.empty((N, H), device=x.device, dtype=torch.float32)

        # Strides for Bx_flat
        stride_bxn = H * S
        stride_bxh = S
        stride_bxl = 1

        stride_wh = H * 4
        stride_wk = 1

        stride_on = H
        stride_oh = 1

        # Tile sizes for conv
        BLOCK_C = 64  # along channels, but we have H, so it's fine
        BLOCK_T = 128

        grid_conv = (triton.cdiv(N, BLOCK_T), triton.cdiv(H, BLOCK_C))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_flat, conv_weight_c, conv_bias_c, conv_out,
            N, H, S, K, PAD,
            stride_bxn, stride_bxh, stride_bxl,
            stride_wh, stride_wk,
            stride_on, stride_oh,
            BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )

        # Reshape to (B, S, H)
        conv_out = conv_out.view(B, S, H).contiguous()

        # 4) Output gating: y = C * conv_out
        # We can do this in PyTorch (element-wise)
        y = C_vec * conv_out

        # Transpose back to (B, H, S): y.transpose(-1, -2)
        # But we need to keep (B, S, H) for final projection; transpose back is not needed
        # Proceed to final projection.

        # 5) Final out-projection: y @ out_proj_weight^T + out_proj_bias
        # y has shape (B, S, H). We'll flatten (B*S, H)
        y_flat = y.view(M, H).contiguous()
        W_out = out_proj_weight.contiguous()       # (H, H)
        bias_out = out_proj_bias.contiguous()      # (H,)

        out = torch.empty((M, H), device=x.device, dtype=torch.float32)

        stride_ym, stride_yh = H, 1
        stride_wom, stride_woh = H, 1
        stride_outm, stride_outn = 1, 1

        # Tile sizes
        BLOCK_M2 = 128
        BLOCK_N2 = 64
        BLOCK_K2 = 32

        grid_out = (triton.cdiv(M, BLOCK_M2), triton.cdiv(H, BLOCK_N2))
        out_proj_linear_kernel[grid_out](
            y_flat, W_out, bias_out, out,
            M, H,
            stride_ym, stride_yh,
            stride_wom, stride_woh,
            stride_outm, stride_outn,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2,
        )

        # Reshape to (B, S, H)
        out = out.view(B, S, H).contiguous()

        return out


def run(*args):
    return ModelNew()(*args)
