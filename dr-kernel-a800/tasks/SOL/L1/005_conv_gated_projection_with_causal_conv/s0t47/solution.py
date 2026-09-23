import torch
import triton
import triton.language as tl


@triton.jit
def out_proj_kernel(
    y_ptr,            # input y: (B, S, H), we read y_T = y.transpose(-1, -2) -> (B, H, S)
    out_w_ptr,        # out_proj_weight: (H, H), we read as weight matrix (H rows, H cols)
    out_b_ptr,        # out_proj_bias: (H,)
    out_ptr,          # output: (B, S, H)
    B: tl.constexpr,  # batch size
    S: tl.constexpr,  # seq_len
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # tile size along H for accumulation
):
    """
    Compute output[b, s, h] = sum_{h2=0..H-1} y_T[b, h, s] * out_w[h, h2] + out_b[h]
    where y_T is (B, H, S).
    Grid: (B, S, ceil(H/BLOCK_H))
    """
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_block = tl.program_id(2)
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H

    # Accumulator for output[b, s, offs_h]
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Read y_T[b, :, s] -> vector of length H
    # y_T layout is (B, H, S) contiguous: strides (H*S, S, 1)
    # Address: y_ptr + b*(H*S) + h*s + s
    y_vec = tl.zeros([BLOCK_H], dtype=tl.float32)
    for h in range(0, H):
        # Load y_T[b, h, s] = y_ptr[b*(H*S) + h*s + s]
        y_val = tl.load(y_ptr + b * (H * S) + h * S + s, mask=mask, other=0.0)
        y_vec[h] = y_val

    # For each h2 in H, accumulate dot: acc += y_vec[h2] * out_w[h, h2] + out_b[h]
    for h2 in range(0, H):
        # weight[h, h2] scalar
        w_val = tl.load(out_w_ptr + h * H + h2)
        # bias[h]
        b_val = tl.load(out_b_ptr + h)
        # acc += y_vec[h2] * w_val
        acc += y_vec[h2] * w_val + b_val

    # Store to output[b, s, h] = acc[h] for each h in offs_h
    for i in range(0, BLOCK_H):
        oh = offs_h[i]
        if mask[i]:
            tl.store(out_ptr + b * (S * H) + s * H + oh, acc[i])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements the original pipeline, using Triton for the final linear projection,
        and torch for in_proj, gating, conv, and output gating to ensure correctness
        and avoid Triton runtime issues. We still return the final output tensor.
        """
        device = x.device
        dtype = x.dtype

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        # in_proj_weight: (3H, H), in_proj_bias: (3H,)
        BCx = torch.nn.functional.linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)
        B, S, M = BCx.shape
        H = int(M // 3)
        assert M == 3 * H, "in_proj_weight must have in_features=hidden_size"

        # 2) Split BCx into B, C, x_proj along last dim
        B_t = BCx[:, :, :H]            # (B, S, H)
        C_t = BCx[:, :, H:(2 * H)]     # (B, S, H)
        x_proj = BCx[:, :, (2 * H):]   # (B, S, H)

        # 3) Element-wise gating: Bx = B_t * x_proj
        Bx = B_t * x_proj  # (B, S, H)

        # 4) Left-pad for causal conv by pad_left = conv_weight.shape[2] - 1 (here 4-1=3)
        K = conv_weight.shape[2]
        pad_left = K - 1
        S_padded = S + pad_left
        Bx_padded = torch.nn.functional.pad(Bx, (pad_left, 0))  # pad on left along sequence
        # Reshape to (B, H, S_padded) for conv1d
        Bx_padded = Bx_padded.transpose(1, 2).transpose(0, 1)  # (H, S_padded, B)
        # But conv1d expects (N, C, L); here N=B, C=H, L=S_padded
        # We'll permute back after conv
        # However torch.nn.functional.conv1d expects (N, C, L). We need (B, H, S_padded).
        # So:
        Bx_padded = Bx_padded.transpose(0, 2).transpose(1, 2).contiguous()  # (B, H, S_padded)

        # 5) Grouped causal conv: conv_weight: (H, 1, K), conv_bias: (H,)
        # F.conv1d expects input (N, C, L) = (B, H, S_padded), weight (C_out, C_in/groups, K) with groups=H
        # Here C_in = H, C_out = H, groups = H, so each channel convolves itself.
        conv_out = torch.nn.functional.conv1d(Bx_padded, conv_weight, conv_bias, groups=H)  # (B, H, S)

        # 6) Output gating: y = C_t * conv_out
        y = C_t * conv_out  # (B, H, S)

        # 7) Final linear projection using Triton: output = linear(y, out_proj_weight, out_proj_bias)
        # y is (B, H, S). We need y_T = (B, S, H) for our Triton kernel.
        y_T = y.transpose(1, 2).contiguous()  # (B, S, H), float32

        # Prepare output tensor
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Launch Triton kernel: compute out[b, s, h] = sum_h2 y_T[b, h, s] * out_w[h, h2] + out_b[h]
        # Grid: (B, S, ceil(H/BLOCK_H))
        BLOCK_H = 64
        grid = (B, S, triton.cdiv(H, BLOCK_H))
        out_proj_kernel[grid](
            y_T, out_proj_weight.contiguous(), out_proj_bias.contiguous(), output,
            B=B, S=S, H=H, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # Cast to original dtype if needed
        if output.dtype != dtype:
            output = output.to(dtype)
        return output


def run(*args):
    return ModelNew()(*args)
