import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_OUT,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # X_ptr: [M, H]
    # W_ptr: [M_OUT, H]
    # OUT_ptr: [M, M_OUT]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Accumulate X[:, :H] @ W[:M_OUT, :H]^T
    for k in range(0, H):
        # X tile: [BLOCK_M]
        x_ptrs = X_ptr + (offs_m * stride_xm + k * stride_xn)
        x_vals = tl.load(x_ptrs, mask=m_mask, other=0.0)  # [BLOCK_M]
        # W tile: [BLOCK_N]
        w_ptrs = W_ptr + (offs_n * stride_wm + k * stride_wn)
        w_vals = tl.load(w_ptrs, mask=n_mask, other=0.0)  # [BLOCK_N]
        # Outer product
        acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc += bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def multiply_chunk_kernel(
    IN_ptr, SCALE_ptr, OUT_ptr,
    M, H,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Elementwise: OUT[b, s, h] = IN[b, s, h] * SCALE[other] for a fixed other index
    # Here we implement elementwise multiply for a 3D tensor by iterating over M dimension.
    # We choose grid over (M, H tiles). For each (b,s), we process H in tiles.
    pid_m = tl.program_id(0)  # over M=B*S
    pid_n = tl.program_id(1)  # over H tiles

    # Compute b, s from pid_m
    b = pid_m // S
    s = pid_m % S

    offs_h = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    h_mask = offs_h < H

    # Pointers to IN[b, s, :] and OUT[b, s, :]
    in_ptrs = IN_ptr + (b * stride_im + s * stride_in + offs_h * stride_in)
    out_ptrs = OUT_ptr + (b * stride_om + s * stride_on + offs_h * stride_on)

    # Load scale (vector) SCALE_ptr has shape [H]
    scale_ptrs = SCALE_ptr + offs_h
    scale = tl.load(scale_ptrs, mask=h_mask, other=1.0)

    # Load input tile and multiply
    in_vals = tl.load(in_ptrs, mask=h_mask, other=0.0)
    out_vals = in_vals * scale

    # Store
    tl.store(out_ptrs, out_vals, mask=h_mask)


@triton.jit
def final_proj_kernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_im, stride_in,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # IN: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < IN_H
        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Load W block: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # acc += x @ w^T
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc += bias_vals[None, :]

    # Store
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self, S: int):
        super().__init__()
        self.S = S  # keep for potential future use; not used directly

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
        Use Triton for in_proj, gating, final projection; keep PyTorch for grouped causal conv1d.
        x: (B, S, H)
        in_proj_weight: (3*H, H)
        conv_weight: (H, 1, 4)
        out_proj_weight: (H, H)
        """
        B, S, H = x.shape
        M_out = in_proj_weight.shape[0]
        # Ensure CUDA tensors
        assert x.is_cuda, "Input x must be on CUDA for Triton kernels"
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda and out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All tensors must be CUDA for Triton kernels"

        # 1) in_proj: x -> (B, S, 3H) via in_proj_weight
        x_flat = x.reshape(B * S, H).contiguous()
        y_flat = torch.empty((B * S, M_out), device=x.device, dtype=x.dtype)

        grid_in = (triton.cdiv(B * S, 128), triton.cdiv(M_out, 64))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            B * S, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        # Reshape to (B, S, 3H)
        BCx = y_flat.view(B, S, 3 * H)
        B_tensor = BCx[:, :, :H].contiguous()         # (B, S, H)
        C_tensor = BCx[:, :, H : 2 * H].contiguous()  # (B, S, H)
        x_proj = BCx[:, :, 2 * H : 3 * H].contiguous()  # (B, S, H)

        # 2) gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        grid_g = (B * S, triton.cdiv(H, 128))
        multiply_chunk_kernel[grid_g](
            B_tensor, x_proj, Bx,
            B * S, H,
            B_tensor.stride(0), B_tensor.stride(2),
            Bx.stride(0), Bx.stride(2),
            BLOCK_M=1, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # 3) grouped causal conv1d on Bx with kernel_size=4, groups=H
        # Using PyTorch to ensure correctness
        conv_out = torch.nn.functional.conv1d(
            Bx, conv_weight, conv_bias, bias=conv_bias, stride=1, padding=0, dilation=1, groups=H
        )  # (B, H, S)

        # 4) Output gating: y = C * conv_out, shape (B, H, S)
        y_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)
        # Implement elementwise multiply in Triton with grid over (B, H, S) tiles
        grid_go = (B, H, triton.cdiv(S, 128))
        multiply_chunk_kernel[grid_go](
            C_tensor, conv_out, y_out,
            B * H * S, S,
            C_tensor.stride(0), C_tensor.stride(2),
            y_out.stride(0), y_out.stride(2),
            BLOCK_M=1, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # 5) Final projection: y_out (B, H, S) -> (B, S, H)
        y_out_flat = y_out.transpose(1, 2).contiguous().view(B * S, H)
        out_flat = torch.empty((B * S, H), device=x.device, dtype=x.dtype)

        grid_out = (triton.cdiv(B * S, 128), triton.cdiv(H, 64))
        final_proj_kernel[grid_out](
            y_out_flat, out_proj_weight, out_proj_bias, out_flat,
            B * S, H, H,
            y_out_flat.stride(0), y_out_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        return out_flat.view(B, S, H)


def run(*args):
    return ModelNew()(*args)
