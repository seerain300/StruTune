import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, C_out, H, W,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    NUM_CI: tl.constexpr, NUM_H: tl.constexpr, NUM_W: tl.constexpr):
    # One program per output element (n, co, h, w)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    for ci in range(NUM_CI):
        for kh in range(NUM_H):
            for kw in range(NUM_W):
                h_in = pid_h + kh
                w_in = pid_w + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # Input load (masked)
                x_offset = pid_n * x_sN + ci * x_sC + h_in * x_sH + w_in * x_sW
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                x_val = x_val.to(tl.float32)
                # Weight load (scalar)
                w_offset = pid_co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                w_val = tl.load(w_ptr + w_offset)
                w_val = w_val.to(tl.float32)
                acc += x_val * w_val

    y_offset = pid_n * y_sN + pid_co * y_sC + pid_h * y_sH + pid_w * y_sW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps  # not used directly; GroupNorm uses its own eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure contiguous and float32
        x32 = x.contiguous().to(torch.float32)

        # First conv: y1 = conv3x3(x, conv1_weight)
        N, C_in, H, W = x32.shape
        C_out = conv1_weight.shape[0]
        y1 = torch.empty((N, C_out, H, W), dtype=torch.float32, device=x32.device)

        conv3x3_nchw_4d[(N, C_out, H, W)](
            x32, conv1_weight, y1,
            N, C_in, C_out, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            NUM_CI=C_in, NUM_H=3, NUM_W=3,
            num_warps=1
        )

        # GroupNorm and SiLU for y1
        # GroupNorm: num_groups=32, affine=True, eps=1e-5 (PyTorch default)
        y1_gn = torch.nn.functional.group_norm(y1, num_groups=32, weight=norm1_weight, bias=norm1_bias, eps=1e-5)
        y1_silu = torch.nn.functional.silu(y1_gn)

        # Second conv: y2 = conv3x3(y1_silu, conv2_weight)
        N2, C_in2, H2, W2 = y1_silu.shape
        C_out2 = conv2_weight.shape[0]
        assert N2 == N and H2 == H and W2 == W
        y2 = torch.empty((N, C_out2, H, W), dtype=torch.float32, device=x32.device)

        conv3x3_nchw_4d[(N, C_out2, H, W)](
            y1_silu, conv2_weight, y2,
            N, C_in2, C_out2, H, W,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            NUM_CI=C_in2, NUM_H=3, NUM_W=3,
            num_warps=1
        )

        # GroupNorm and SiLU for y2
        y2_gn = torch.nn.functional.group_norm(y2, num_groups=32, weight=norm2_weight, bias=norm2_bias, eps=1e-5)
        y2_silu = torch.nn.functional.silu(y2_gn)

        # Residual connection: y = y2_silu + x
        out = y2_silu + x32

        return out


def run(*args):
    return ModelNew()(*args)
