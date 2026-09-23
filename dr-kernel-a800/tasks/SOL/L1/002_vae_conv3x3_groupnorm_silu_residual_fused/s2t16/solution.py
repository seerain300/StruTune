import torch
import triton
import triton.language as tl


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, H, W, C_OUT,
    BLOCK_W: tl.constexpr,  # tile size along W
):
    # Grid: (N, C_OUT, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    h = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Accumulator for output row
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C):
        # Accumulate over 3x3 neighborhood with padding=1
        for dh in range(-1, 1):
            for dw in range(-1, 1):
                h2 = h + dh
                in_idx = ((pid_n * C + ci) * H + h2) * W + (w_offsets + dw)
                # mask for padding
                mask = (h2 >= 0) & (h2 < H) & ((w_offsets + dw) >= 0) & ((w_offsets + dw) < W) & mask_w
                x_val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
                w_idx = pid_co * (C * 9) + ci * 9 + (1 + dh) * 3 + (1 + dw)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # store acc to out[n, co, h, w_offsets]
    out_idx = ((pid_n * C_OUT + pid_co) * H + h) * W + w_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_w)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We won't keep parameters; this is just to mirror the original run function signature.
        # In a real scenario, you'd inject conv weights and norms via constructor or forward.

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Ensure float32 and contiguous
        x_in = x.contiguous().float()
        conv1_weight = conv1_weight.contiguous().float()
        conv2_weight = conv2_weight.contiguous().float()
        norm1_weight = norm1_weight.contiguous().float()
        norm1_bias = norm1_bias.contiguous().float()
        norm2_weight = norm2_weight.contiguous().float()
        norm2_bias = norm2_bias.contiguous().float()

        N, C, H, W = x_in.shape

        # First convolution in Triton: out1 = conv3x3(x_in)
        out1 = torch.empty((N, C, H, W), dtype=torch.float32, device=x_in.device)
        conv3x3_nchw_kernel[(N, C, H, triton.cdiv(W, 64))](
            x_in, conv1_weight, out1,
            N, C, H, W, C,
            BLOCK_W=64,
        )

        # GroupNorm + SiLU using PyTorch (robust and correct)
        out1 = torch.nn.functional.group_norm(out1, 32, weight=norm1_weight, bias=norm1_bias, eps=eps)
        out1 = torch.nn.functional.silu(out1)

        # Second convolution in Triton: out2 = conv3x3(out1)
        out2 = torch.empty((N, C, H, W), dtype=torch.float32, device=x_in.device)
        conv3x3_nchw_kernel[(N, C, H, triton.cdiv(W, 64))](
            out1, conv2_weight, out2,
            N, C, H, W, C,
            BLOCK_W=64,
        )

        # GroupNorm + SiLU using PyTorch
        out2 = torch.nn.functional.group_norm(out2, 32, weight=norm2_weight, bias=norm2_bias, eps=eps)
        out2 = torch.nn.functional.silu(out2)

        # Residual connection
        out = out2 + x_in

        return out


def run(*args):
    return ModelNew()(*args)
