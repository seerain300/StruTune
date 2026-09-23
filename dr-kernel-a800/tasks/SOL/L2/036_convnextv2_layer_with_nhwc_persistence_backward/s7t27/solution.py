import torch
import triton
import triton.language as tl


# =========================
# Triton kernels: init
# =========================
@triton.jit
def normal_fill_kernel(OUT_ptr, N, MEAN, STD, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Generate normal via box-muller in Triton
    u = tl.rand(offsets)  # uniform in [0,1)
    v = tl.rand(offsets)  # uniform in [0,1)
    z = tl.sqrt(-2.0 * tl.log(1.0 - u)) * tl.sign(2.0 * v - 1.0)
    val = MEAN + STD * z
    tl.store(OUT_ptr + offsets, val, mask=mask)


@triton.jit
def ones_fill_kernel(OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    val = 1.0
    tl.store(OUT_ptr + offsets, val, mask=mask)


@triton.jit
def drop_mask_kernel(OUT_ptr, N, DROP_PROB, BLOCK: tl.constexpr, seed: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # LCG RNG for reproducibility
    s = (seed * offsets + 1013904223)
    rnd = (s >> 32) * 1.0 / 4294967296.0
    keep = rnd > DROP_PROB
    val = tl.where(keep, 1.0, 0.0)
    tl.store(OUT_ptr + offsets, val, mask=mask)


# =========================
# Triton kernels: conv (depthwise) forward
# =========================
@triton.jit
def conv2d_depthwise_forward_kernel(
    X_ptr,       # input: *float32, (B, C, H, W)
    W_ptr,       # weight: *float32, (C, 1, 7, 7)
    Y_ptr,       # output: *float32, (B, C, H, W)
    B, C, H, W,  # int32
    BLOCK: tl.constexpr
):
    # Grid: (B, C). Each program handles one (b, c) plane over H*W
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    # Initialize accumulator
    acc = tl.zeros((), dtype=tl.float32)
    # Output index base for (b, c)
    for h_out in range(H):
        for w_out in range(W):
            base_out = pid_b * C * H * W + pid_c * H * W + h_out * W + w_out
            # Sum over 7x7 neighborhood with padding=3. Since padding=3, input index i = h_out + kh - 3, j = w_out + kw - 3
            for kh in range(7):
                for kw in range(7):
                    i = h_out + kh - 3
                    j = w_out + kw - 3
                    if (i >= 0) and (i < H) and (j >= 0) and (j < W):
                        x_index = pid_b * C * H * W + pid_c * H * W + i * W + j
                        xval = tl.load(X_ptr + x_index)
                        w_index = pid_c * (1 * 7 * 7) + kh * 7 + kw
                        wval = tl.load(W_ptr + w_index)
                        acc += xval * wval
            tl.store(Y_ptr + base_out, acc)


# =========================
# Triton kernels: permute NCHW -> NHWC: out[b, h, w, c] = x[b, c, h, w]
# =========================
@triton.jit
def permute_bchw_to_bhwc_kernel(
    IN_ptr,      # *float32, (B, C, H, W)
    OUT_ptr,     # *float32, (B, H, W, C)
    B, C, H, W,  # int32
    BLOCK: tl.constexpr
):
    pid_b = tl.program_id(0)
    c = tl.program_id(1)  # since grid is (B, C, H, W), we treat C as another dim
    h = tl.program_id(2)
    w = tl.program_id(3)
    base_in = pid_b * C * H * W + c * H * W + h * W + w
    val = tl.load(IN_ptr + base_in)
    idx_out = pid_b * H * W * C + h * W * C + w * C + c
    tl.store(OUT_ptr + idx_out, val)


# =========================
# Triton kernels: LayerNorm (reduce sum and sumsq across channels, then normalize)
# =========================
@triton.jit
def layernorm_reduce_sum_sumsq_kernel(
    X_ptr,        # *float32, (B, H, W, C) contiguous
    SUM_ptr,      # *float32, (B*H*W,)
    SUMSQ_ptr,    # *float32, (B*H*W,)
    B, H, W, C,   # int32
    BLOCK_C: tl.constexpr
):
    pid = tl.program_id(0)  # one program per (b,h,w)
    HW = H * W
    b = pid // HW
    hw = pid % HW
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)
    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        base = b * H * W * C + hw * C + c_offsets
        vals = tl.load(X_ptr + base, mask=c_mask, other=0.0)
        # sum and sumsq over this block
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)
    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def layernorm_forward_kernel(
    X_ptr,        # *float32, (B, H, W, C)
    SUM_ptr,      # *float32, (B*H*W,)
    SUMSQ_ptr,    # *float32, (B*H*W,)
    Y_ptr,        # *float32, (B, H, W, C)
    B, H, W, C,   # int32
    EPS,          # float32
    BLOCK_C: tl.constexpr
):
    pid = tl.program_id(0)  # one program per (b,h,w)
    HW = H * W
    b = pid // HW
    hw = pid % HW
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    mean = sum_val / C
    var = sumsq_val / C - mean * mean
    inv_std = tl.rsqrt(var + EPS)
    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        base = b * H * W * C + hw * C + c_offsets
        x = tl.load(X_ptr + base, mask=c_mask, other=0.0)
        y = (x - mean) * inv_std
        tl.store(Y_ptr + base, y, mask=c_mask)


# =========================
# Triton kernels: GELU (forward)
# =========================
@triton.jit
def gelu_forward_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    sqrt_2_over_pi = 0.7978845608028654
    c = 0.044715
    inner = sqrt_2_over_pi * (x + c * x * x * x)
    t = tl.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    tl.store(Y_ptr + offsets, y, mask=mask)


# =========================
# Triton kernels: GRN forward (per (b,h,w): global L2 norm over channels, scale x_gelu)
# =========================
@triton.jit
def grn_forward_kernel(
    X_ptr,        # *float32, (B, H, W, C)
    Y_ptr,        # *float32, (B, H, W, C)
    B, H, W, C,   # int32
    BLOCK_C: tl.constexpr
):
    pid = tl.program_id(0)  # one program per (b,h,w)
    HW = H * W
    b = pid // HW
    hw = pid % HW
    # First compute norm across C
    norm = tl.zeros((), dtype=tl.float32)
    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        base = b * H * W * C + hw * C + c_offsets
        x = tl.load(X_ptr + base, mask=c_mask, other=0.0)
        norm += tl.sum(x * x, axis=0)
    norm = tl.sqrt(norm)
    eps = 1e-6
    denom = norm + eps
    inv_denom = 1.0 / denom
    # Apply scaling: Y = X * inv_denom
    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        base = b * H * W * C + hw * C + c_offsets
        x = tl.load(X_ptr + base, mask=c_mask, other=0.0)
        y = x * inv_denom
        tl.store(Y_ptr + base, y, mask=c_mask)


# =========================
# Triton kernels: elementwise scale
# =========================
@triton.jit
def elem_scale_kernel(X_ptr, SCALE_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    scale = tl.load(SCALE_ptr)  # scalar
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = x * scale
    tl.store(OUT_ptr + offsets, y, mask=mask)


# =========================
# ModelNew: forward launches all Triton kernels
# =========================
class ModelNew(torch.nn.Module):
    def __init__(self, B: int, H: int, W: int, C: int = 128, eps: float = 1e-6, drop_path_prob: float = 0.1):
        super().__init__()
        self.B = B
        self.H = H
        self.W = W
        self.C = C
        self.eps = eps
        self.drop_path_prob = drop_path_prob
        self.device = torch.device("cuda")  # Triton requires CUDA

    def forward(self):
        B = self.B
        H = self.H
        W = self.W
        C = self.C

        # Allocate all tensors on CUDA
        # dwconv_weight: (C, 1, 7, 7), init N(0, 1/sqrt(49))
        dwconv_weight = torch.empty((C, 1, 7, 7), device=self.device, dtype=torch.float32)
        normal_fill_kernel[(triton.cdiv(C * 1 * 7 * 7, 1024),)](
            dwconv_weight, C * 1 * 7 * 7, 0.0, (1.0 / 49.0) ** 0.5, BLOCK=1024
        )

        # layernorm_weight: (C,) init N(1, 0.01)
        layernorm_weight = torch.empty((C,), device=self.device, dtype=torch.float32)
        normal_fill_kernel[(triton.cdiv(C, 1024),)](layernorm_weight, C, 1.0, 0.01, BLOCK=1024)

        # pwconv1_weight: (4C, C), init N(0, sqrt(2/C))
        C4 = C * 4
        pwconv1_weight = torch.empty((C4, C), device=self.device, dtype=torch.float32)
        normal_fill_kernel[(triton.cdiv(C4 * C, 1024),)](pwconv1_weight, C4 * C, 0.0, (2.0 / C) ** 0.5, BLOCK=1024)

        # grn_weight: (1,


def run(*args):
    return ModelNew()(*args)
