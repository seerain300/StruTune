import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,            # *f32, shape (B, C_in, H, W)
    w_ptr,            # *f32, shape (C_in, C_out, 3, 3)
    y_ptr,            # *f32, shape (B, C_out, H, W)
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    # Each program computes one output element y[n, c_out, h_out, w_out]
    pid = tl.program_id(0)
    total = B * C_out * H * W
    n = pid // (C_out * H * W)
    rem = pid % (C_out * H * W)
    c_out = rem // (H * W)
    h_out = rem // W
    w_out = rem % W

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks and 3x3 kernel window
    for ic_base in range(0, C_in, BLOCK_IN):
        for kh in range(3):
            for kw in range(3):
                ih = h_out + kh - 1
                iw = w_out + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                for ic in range(BLOCK_IN):
                    ic_full = ic_base + ic
                    x_off = (n * C_in + ic_full) * H * W + ih * W + iw
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    w_off = ic_full * (3 * 3) + kh * 3 + kw
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    y_off = (n * C_out + c_out) * H * W + h_out * W + w_out
    tl.store(y_ptr + y_off, acc)


@triton.jit
def groupnorm_affine_kernel(
    in_ptr,           # *f32, shape (B, C, H*W), contiguous
    out_ptr,          # *f32, shape (B, C, H*W), contiguous
    weight_ptr,       # *f32, shape (C,)
    bias_ptr,         # *f32, shape (C,)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    NUM_GROUPS: tl.constexpr, EPS: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    GROUP_SIZE = C // NUM_GROUPS
    c0 = g * GROUP_SIZE

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    hw = H * W
    for ic in range(GROUP_SIZE):
        c = c0 + ic
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            h = offs // W
            w = offs % W
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
            sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / (GROUP_SIZE * hw)
    var = sum_sq / (GROUP_SIZE * hw) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    for ic in range(GROUP_SIZE):
        c = c0 + ic
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            h = offs // W
            w = offs % W
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            y = (x - mean) * inv_std
            scale = tl.load(weight_ptr + c, mask=True, other=1.0)
            bias = tl.load(bias_ptr + c, mask=True, other=0.0)
            y = y * scale + bias
            out_offs = (n * C + c) * hw + offs
            tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def silu_kernel(in_ptr, out_ptr, TOTAL_ELEMS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, TOTAL_ELEMS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = a + b
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        # Tunable kernel parameters
        self.block_in = 16     # chunk over input channels in conv
        self.block_hw = 1024   # chunk over H*W for groupnorm
        self.silu_block = 4096 # vectorization for elementwise SiLU

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure inputs are contiguous and float32
        x = x.contiguous().to(torch.float32)
        device = x.device

        # First conv: NCHW, stride=1, padding=1, no bias
        B, C_in, H, W = x.shape
        assert C_in % self.num_groups == 0, f"Input channels {C_in} must be divisible by num_groups {self.num_groups}"
        C_out1 = conv1_weight.shape[1]
        assert conv1_weight.shape[0] == C_in, "conv1_weight C_in must match input C_in"
        assert conv1_weight.shape[2] == 3 and conv1_weight.shape[3] == 3, "conv1_weight must be 3x3"
        H_out1, W_out1 = H, W
        out1 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=torch.float32)

        total_out1 = B * C_out1 * H_out1 * W_out1
        grid1 = (total_out1,)
        conv3x3_nchw_fp32[grid1](
            x, conv1_weight,
            out1,
            B, C_in, C_out1, H, W,
            BLOCK_IN=self.block_in,
        )

        # GroupNorm 1 with affine
        in_flat1 = out1.view(B, C_out1, H_out1 * W_out1).contiguous()
        gn_out1 = torch.empty_like(in_flat1, device=device, dtype=torch.float32)
        grid_gn1 = (B, self.num_groups)
        groupnorm_affine_kernel[grid_gn1](
            in_flat1, gn_out1,
            norm1_weight, norm1_bias,
            B, C_out1, H_out1, W_out1, self.num_groups, self.eps, BLOCK_HW=self.block_hw
        )

        # SiLU 1
        silu_out1 = torch.empty_like(gn_out1, device=device, dtype=torch.float32)
        total1 = C_out1 * H_out1 * W_out1
        grid_silu1 = (triton.cdiv(total1, self.silu_block),)
        silu_kernel[grid_silu1](gn_out1, silu_out1, total1, BLOCK=self.silu_block)

        # Second conv: NCHW, stride=1, padding=1, no bias
        B2, C_in2, H2, W2 = silu_out1.shape  # B2 == B_out1 == B, C_in2 == C_out1, H2 == H_out1, W2 == W_out1
        C_out2 = conv2_weight.shape[1]
        assert conv2_weight.shape[0] == C_in2, "conv2_weight C_in must match first conv output channels"
        assert conv2_weight.shape[2] == 3 and conv2_weight.shape[3] == 3, "conv2_weight must be 3x3"

        out2 = torch.empty((B, C_out2, H2, W2), device=device, dtype=torch.float32)
        total_out2 = B * C_out2 * H2 * W2
        grid2 = (total_out2,)
        conv3x3_nchw_fp32[grid2](
            silu_out1.view(B, C_in2, H2, W2), conv2_weight,
            out2,
            B, C_in2, C_out2, H2, W2,
            BLOCK_IN=self.block_in,
        )

        # GroupNorm 2 with affine
        in_flat2 = out2.view(B, C_out2, H2 * W2).contiguous()
        gn_out2 = torch.empty_like(in_flat2, device=device, dtype=torch.float32)
        grid_gn2 = (B, self.num_groups)
        groupnorm_affine_kernel[grid_gn2](
            in_flat2, gn_out2,
            norm2_weight, norm2_bias,
            B, C_out2, H2, W2, self.num_groups, self.eps, BLOCK_HW=self.block_hw
        )

        # SiLU 2
        silu_out2 = torch.empty_like(gn_out2, device=device, dtype=torch.float32)
        total2 = C_out2 * H2 * W2
        grid_silu2 = (triton.cdiv(total2, self.silu_block),)
        silu_kernel[grid_silu2](gn_out2, silu_out2, total2, BLOCK=self.silu_block)

        # Residual addition: add original input x (cast to fp32) to the final output
        # Note: original code adds x (input) to final out. We add silu_out2 (final processed tensor) with x.
        # To ensure the kernel is actually used, we perform the addition here.
        x_flat = x.view(B, C_in, H * W).contiguous()
        final_flat = torch.empty_like(x_flat, device=device, dtype=torch.float32)
        total_res = B * C_in * H * W
        grid_add = (triton.cdiv(total_res, self.silu_block),)
        add_residual_kernel[grid_add](x_flat, silu_out2.view(B, C_out2, H2 * W2), final_flat, total_res, BLOCK=self.silu_block)

        # Reshape back to (B, C_in, H, W)
        return final_flat.view(B, C_in, H, W)


def run(*args):
    return ModelNew()(*args)
