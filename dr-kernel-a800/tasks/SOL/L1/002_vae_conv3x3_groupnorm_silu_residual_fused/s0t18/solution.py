import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nobias_single(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights [C_out, C_in, 3, 3] flattened to [C_out, C_in, 9]
    out_ptr,      # *float, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # program ids: one program per (n, co, h_out, w_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    h_out = pid_h
    w_out = pid_w

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 kernel positions
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh
                w_in = w_out - 1 + kw
                # ensure indices are within input bounds
                in_h = (h_in >= 0) & (h_in < H)
                in_w = (w_in >= 0) & (w_in < W)
                in_bounds = in_h & in_w
                # base pointer for x at (n, ci, h_in, w_in)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                # guarded load: use 0.0 when out of bounds
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # store result at (n, co, h_out, w_out)
    ptr_out = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
    tl.store(ptr_out, acc)


@triton.jit
def group_norm_two_pass(
    out_ptr,         # *const float, input tensor after conv, shape [B, C, H, W]
    gamma_ptr,       # *const float, per-channel gamma (weight) [C]
    beta_ptr,        # *const float, per-channel beta (bias) [C]
    out_norm_ptr,    # *float, output normalized + affine [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # First pass: compute sum and sum of squares per (n, group)
    group_size = C // num_groups
    for n in tl.static_range(B):
        for group in tl.static_range(num_groups):
            sum_val = tl.zeros((), dtype=tl.float32)
            sum_sq = tl.zeros((), dtype=tl.float32)
            # iterate over channels in this group and all spatial positions
            for c_off in tl.static_range(C):
                if (c_off % num_groups) == group:
                    for h in tl.static_range(H):
                        for w in tl.static_range(W):
                            ptr = out_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            x_val = tl.load(ptr).to(tl.float32)
                            sum_val += x_val
                            sum_sq += x_val * x_val
            mean = sum_val / (H * W)
            var = sum_sq / (H * W) - mean * mean
            rstd = 1.0 / tl.sqrt(var + eps)
            # Second pass: normalize and apply affine gamma/beta
            for c_off in tl.static_range(C):
                if (c_off % num_groups) == group:
                    gamma = tl.load(gamma_ptr + c_off).to(tl.float32)
                    beta = tl.load(beta_ptr + c_off).to(tl.float32)
                    for h in tl.static_range(H):
                        for w in tl.static_range(W):
                            in_ptr = out_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            x_val = tl.load(in_ptr).to(tl.float32)
                            y = (x_val - mean) * rstd
                            out_val = y * gamma + beta
                            out_ptr_cur = out_norm_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            tl.store(out_ptr_cur, out_val)


@triton.jit
def silu_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual(x_ptr, residual_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offs, mask=mask, other=0.0)
    y = x + res
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, C_in: int, C_out: int, H: int, W: int, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Ensure all tensors are on same device and dtype float32
        device = x.device
        dtype = torch.float32
        x = x.to(device=device, dtype=dtype).contiguous()

        # First convolution: y1 = conv3x3(x, conv1_weight, bias=None)
        y1 = torch.empty((x.shape[0], self.C_out, self.H, self.W), device=device, dtype=dtype)
        grid_conv1 = (x.shape[0], self.C_out, self.H, self.W)
        conv3x3_nobias_single[grid_conv1](
            x, conv1_weight.to(device=device, dtype=dtype).contiguous(),
            y1,
            B=x.shape[0], C_in=self.C_in, H=self.H, W=self.W, C_out=self.C_out,
            x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
            out_stride_n=y1.stride(0), out_stride_c=y1.stride(1), out_stride_h=y1.stride(2), out_stride_w=y1.stride(3),
        )

        # GroupNorm1
        y1_gn = torch.empty_like(y1)
        grid_gn1 = (x.shape[0], self.num_groups)
        group_norm_two_pass[grid_gn1](
            y1, norm1_weight.to(device=device, dtype=dtype).contiguous(),
            norm1_bias.to(device=device, dtype=dtype).contiguous(),
            y1_gn,
            B=x.shape[0], C=self.C_out, H=self.H, W=self.W, num_groups=self.num_groups, eps=self.eps,
            out_stride_n=y1.stride(0), out_stride_c=y1.stride(1), out_stride_h=y1.stride(2), out_stride_w=y1.stride(3),
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_gn)
        N1 = y1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](y1_gn, y1_silu, N1, BLOCK=1024)

        # Second convolution: y2_pre = conv3x3(y1_silu, conv2_weight, bias=None)
        y2_pre = torch.empty((x.shape[0], self.C_out, self.H, self.W), device=device, dtype=dtype)
        grid_conv2 = (x.shape[0], self.C_out, self.H, self.W)
        conv3x3_nobias_single[grid_conv2](
            y1_silu, conv2_weight.to(device=device, dtype=dtype).contiguous(),
            y2_pre,
            B=x.shape[0], C_in=self.C_out, H=self.H, W=self.W, C_out=self.C_out,
            x_stride_n=y1_silu.stride(0), x_stride_c=y1_silu.stride(1), x_stride_h=y1_silu.stride(2), x_stride_w=y1_silu.stride(3),
            out_stride_n=y2_pre.stride(0), out_stride_c=y2_pre.stride(1), out_stride_h=y2_pre.stride(2), out_stride_w=y2_pre.stride(3),
        )

        # GroupNorm2
        y2_gn = torch.empty_like(y2_pre)
        grid_gn2 = (x.shape[0], self.num_groups)
        group_norm_two_pass[grid_gn2](
            y2_pre, norm2_weight.to(device=device, dtype=dtype).contiguous(),
            norm2_bias.to(device=device, dtype=dtype).contiguous(),
            y2_gn,
            B=x.shape[0], C=self.C_out, H=self.H, W=self.W, num_groups=self.num_groups, eps=self.eps,
            out_stride_n=y2_pre.stride(0), out_stride_c=y2_pre.stride(1), out_stride_h=y2_pre.stride(2), out_stride_w=y2_pre.stride(3),
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_gn)
        N2 = y2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](y2_gn, y2_silu, N2, BLOCK=1024)

        # Add residual x
        out = torch.empty_like(y2_silu)
        Nfinal = y2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual[grid_add](y2_silu, x, out, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
