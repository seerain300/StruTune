import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nobias_single_elem_const(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights [C_out, C_in, 3, 3], flattened to [C_out, C_in, 9]
    out_ptr,      # *float, output [B, C_out, H, W]
    B: tl.constexpr,
    C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = pid_h - 1 + kh  # padding=1
                w_in = pid_w - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # base pointer for x at (n, ci, h_in, w_in)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # store result at (n, co, h, w)
    ptr_out = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c + pid_h * out_stride_h + pid_w * out_stride_w
    tl.store(ptr_out, acc)


@triton.jit
def group_norm_two_pass_const(
    out_ptr,         # *const float, input tensor after conv, shape [B, C, H, W]
    gamma_ptr,       # *const float, per-channel gamma (weight) [C]
    beta_ptr,        # *const float, per-channel beta (bias) [C]
    out_norm_ptr,    # *float, output normalized + affine [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # Pass 1: compute sum and sum of squares per (n, group)
    group_size = C // num_groups
    for n in tl.static_range(B):
        for g in tl.static_range(num_groups):
            sum_val = tl.zeros((), dtype=tl.float32)
            sum_sq = tl.zeros((), dtype=tl.float32)
            # iterate over channels in this group and all spatial positions
            for c_off in tl.static_range(C):
                if (c_off % num_groups) == g:
                    for h in tl.static_range(H):
                        for w in tl.static_range(W):
                            ptr = out_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            x_val = tl.load(ptr).to(tl.float32)
                            sum_val += x_val
                            sum_sq += x_val * x_val
            mean = sum_val / (group_size * H * W)
            var = sum_sq / (group_size * H * W) - mean * mean
            rstd = 1.0 / tl.sqrt(var + eps)
            # Pass 2: normalize and apply affine, then store
            for c_off in tl.static_range(C):
                if (c_off % num_groups) == g:
                    gamma = tl.load(gamma_ptr + c_off).to(tl.float32)
                    beta = tl.load(beta_ptr + c_off).to(tl.float32)
                    for h in tl.static_range(H):
                        for w in tl.static_range(W):
                            ptr_in = out_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            ptr_out = out_norm_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            x_val = tl.load(ptr_in).to(tl.float32)
                            y = (x_val - mean) * rstd
                            y = y * gamma + beta
                            tl.store(ptr_out, y)


@triton.jit
def silu_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Elementwise SiLU over flattened buffers
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(in_ptr, res_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Elementwise addition: out = in + residual
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = a + b
    tl.store(out_ptr + offs, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-5, num_groups=32, H: int = None, W: int = None, C_in: int = None, C_out: int = None):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups
        # These are expected to match typical conv block sizes; host will pass them to ensure consistency
        self.H = H
        self.W = W
        self.C_in = C_in
        self.C_out = C_out

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
    ):
        # Ensure device and dtype compatibility; cast to float32 for Triton
        device = x.device
        x_f = x.to(torch.float32)
        conv1_weight_f = conv1_weight.to(torch.float32)  # shape (C_out, C_in, 3, 3)
        conv2_weight_f = conv2_weight.to(torch.float32)  # shape (C_out, C_out, 3, 3)
        norm1_weight_f = norm1_weight.to(torch.float32)
        norm1_bias_f = norm1_bias.to(torch.float32)
        norm2_weight_f = norm2_weight.to(torch.float32)
        norm2_bias_f = norm2_bias.to(torch.float32)

        # Shapes: x: (B, C_in, H, W)
        B, C_in, H, W = x_f.shape
        C_out = conv1_weight_f.shape[0]
        assert self.C_in is None or self.C_in == C_in, f"Input channels mismatch: expected {self.C_in}, got {C_in}"
        assert self.C_out is None or self.C_out == C_out, f"Output channels mismatch: expected {self.C_out}, got {C_out}"
        assert self.H is None or self.H == H, f"Height mismatch: expected {self.H}, got {H}"
        assert self.W is None or self.W == W, f"Width mismatch: expected {self.W}, got {W}"

        # 1) First conv: out1_pre = conv3x3(x)
        out1_pre = torch.empty((B, C_out, H, W), device=device, dtype=torch.float32)

        grid_conv1 = (B, C_out, H, W)
        conv3x3_nobias_single_elem_const[grid_conv1](
            x_f, conv1_weight_f, out1_pre,
            B, self.C_in if self.C_in is not None else C_in, self.H if self.H is not None else H,
            self.W if self.W is not None else W, self.C_out if self.C_out is not None else C_out,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            out1_pre.stride(0), out1_pre.stride(1), out1_pre.stride(2), out1_pre.stride(3),
        )

        # 2) GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1_pre)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass_const[grid_gn1](
            out1_pre, norm1_weight_f, norm1_bias_f, out1_gn,
            B, self.C_out if self.C_out is not None else C_out, self.H if self.H is not None else H,
            self.W if self.W is not None else W, self.num_groups, self.eps,
            out1_pre.stride(0), out1_pre.stride(1), out1_pre.stride(2), out1_pre.stride(3),
        )

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # 4) Second conv: out2_pre = conv3x3(out1_silu)
        out2_pre = torch.empty((B, C_out, H, W), device=device, dtype=torch.float32)

        grid_conv2 = (B, C_out, H, W)
        conv3x3_nobias_single_elem_const[grid_conv2](
            out1_silu, conv2_weight_f, out2_pre,
            B, self.C_out if self.C_out is not None else C_out, self.H if self.H is not None else H,
            self.W if self.W is not None else W, self.C_out if self.C_out is not None else C_out,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # 5) GroupNorm2 (num_groups=32)
        out2_gn = torch.empty_like(out2_pre)

        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass_const[grid_gn2](
            out2_pre, norm2_weight_f, norm2_bias_f, out2_gn,
            B, self.C_out if self.C_out is not None else C_out, self.H if self.H is not None else H,
            self.W if self.W is not None else W, self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # 7) Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x_f, out, Nfinal, BLOCK=1024)

        return out


# Optional: helper function to run from external code
def run_triton_only(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
):
    # If shapes are known (typical in evaluation), pass them to ModelNew for const kernel compilation
    H, W = x.shape[2], x.shape[3]
    C_in = x.shape[1]
    # The reference uses conv weights (C, C, 3, 3) and C_out equals C_in
    C_out = conv1_weight.shape[0]
    model = ModelNew(eps=1e-5, num_groups=32, H=H, W=W, C_in=C_in, C_out=C_out).to(x.device)
    x_f = x.to(torch.float32)
    w1_f = conv1_weight.to(torch.float32)
    w2_f = conv2_weight.to(torch.float32)
    g1_w = norm1_weight.to(torch.float32)
    g1_b = norm1_bias.to(torch.float32)
    g2_w = norm2_weight.to(torch.float32)
    g2_b = norm2_bias.to(torch.float32)
    return model(x_f, w1_f, g1_w, g1_b, w2_f, g2_w, g2_b)


def run(*args):
    return ModelNew()(*args)
