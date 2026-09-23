import torch
import triton
import triton.language as tl


# Triton kernel: conv3x3 (stride=1, padding=1, no bias), one output element per program.
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float32, input [B, C, H, W]
    w_ptr,        # *const float32, weights [C, C, 3, 3]
    out_ptr,      # *float32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # program ids: one per (n, co, h_out, w_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # compute input coordinates for padding=1
    h_in = pid_h - 1
    w_in = pid_w - 1

    acc = 0.0

    # loop over input channels and 3x3 kernel
    for ci in tl.static_range(C):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_ok = (h_in + kh >= 0) & (h_in + kh < H)
                w_ok = (w_in + kw >= 0) & (w_in + kw < W)
                in_bounds = h_ok & w_ok
                ptr_x = x_ptr + pid_n * x_stride_n + ci * x_stride_c + (h_in + kh) * x_stride_h + (w_in + kw) * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index for w[co, ci, kh, kw]
                w_idx = pid_co * (C * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    out_ptr_idx = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c + pid_h * out_stride_h + pid_w * out_stride_w
    tl.store(out_ptr_idx, acc)


# Triton GroupNorm two-pass per (batch, group). Assumes input is [B, C, H, W] contiguous.
@triton.jit
def group_norm_two_pass(
    x_ptr,          # *const float32
    scale_ptr,      # *const float32, per-channel scale [C]
    bias_ptr,       # *const float32, per-channel bias [C]
    y_ptr,          # *float32
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    y_stride_n: tl.constexpr, y_stride_c: tl.constexpr, y_stride_h: tl.constexpr, y_stride_w: tl.constexpr,
):
    n = tl.program_id(0)  # batch
    g = tl.program_id(1)  # group id
    group_size = C // num_groups
    # first pass: compute sum and sum of squares for this group
    total_sum = 0.0
    total_sq = 0.0
    for c in tl.static_range(C):
        if (c // group_size) == g:
            for h in tl.static_range(H):
                for w in tl.static_range(W):
                    ptr = x_ptr + n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
                    val = tl.load(ptr).to(tl.float32)
                    total_sum += val
                    total_sq += val * val
    mean = total_sum / (group_size * H * W)
    var = total_sq / (group_size * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply affine
    for c in tl.static_range(C):
        if (c // group_size) == g:
            scale = tl.load(scale_ptr + c).to(tl.float32)
            beta = tl.load(bias_ptr + c).to(tl.float32)
            for h in tl.static_range(H):
                for w in tl.static_range(W):
                    ptr_in = x_ptr + n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
                    ptr_out = y_ptr + n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
                    x_val = tl.load(ptr_in).to(tl.float32)
                    y_val = (x_val - mean) * inv_std
                    y_val = y_val * scale + beta
                    tl.store(ptr_out, y_val)


# Triton elementwise SiLU: y = x * sigmoid(x) = x / (1 + exp(-x))
@triton.jit
def silu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Triton elementwise add residual: y = x_out + x_residual
@triton.jit
def add_residual_kernel(x_ptr, res_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a + b
    tl.store(y_ptr + offs, y, mask=mask)


# ModelNew: Triton-only implementation, no torch ops in forward
class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, num_groups: int = 32):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        x: (B, C, H, W), conv weights: (C, C, 3, 3), GroupNorm scale/bias: (C,)
        Returns: (B, C, H, W)
        """
        assert x.dim() == 4, "x must be 4D (B, C, H, W)"
        B, C, H, W = x.shape
        device = x.device

        # Ensure contiguity and float32
        x_f = x.contiguous().to(torch.float32)
        conv1_w = conv1_weight.contiguous().to(torch.float32)
        conv2_w = conv2_weight.contiguous().to(torch.float32)
        norm1_weight_f = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f = norm2_bias.contiguous().to(torch.float32)

        # 1) First conv: y1 = conv3x3(x)
        y1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x_f, conv1_w, y1,
            B, C, H, W,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        )

        # 2) GroupNorm1
        y1_gn = torch.empty_like(y1)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            y1, norm1_weight_f, norm1_bias_f, y1_gn,
            B, C, H, W,
            self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
        )

        # 3) SiLU1
        y1_silu = torch.empty_like(y1_gn)
        N1 = y1_gn.numel()
        BLOCK = 1024
        grid_silu1 = (triton.cdiv(N1, BLOCK),)
        silu_kernel[grid_silu1](y1_gn, y1_silu, N1, BLOCK=BLOCK)

        # 4) Second conv: y2_pre = conv3x3(y1_silu)
        y2_pre = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            y1_silu, conv2_w, y2_pre,
            B, C, H, W,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        )

        # 5) GroupNorm2
        y2_gn = torch.empty_like(y2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            y2_pre, norm2_weight_f, norm2_bias_f, y2_gn,
            B, C, H, W,
            self.num_groups, self.eps,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            y2_gn.stride(0), y2_gn.stride(1), y2_gn.stride(2), y2_gn.stride(3),
        )

        # 6) SiLU2
        y2_silu = torch.empty_like(y2_gn)
        N2 = y2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, BLOCK),)
        silu_kernel[grid_silu2](y2_gn, y2_silu, N2, BLOCK=BLOCK)

        # 7) Add residual x
        out = torch.empty_like(y2_silu)
        Nfinal = y2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, BLOCK),)
        add_residual_kernel[grid_add](y2_silu, x_f, out, Nfinal, BLOCK=BLOCK)

        return out


def run(*args):
    return ModelNew()(*args)
