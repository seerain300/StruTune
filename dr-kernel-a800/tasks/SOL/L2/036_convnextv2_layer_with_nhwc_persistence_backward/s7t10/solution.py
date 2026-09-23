import torch
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================
@triton.jit
def conv2d_depthwise_forward_kernel(
    X_ptr,       # input: (B, C, H, W)
    W_ptr,       # weight: (C, 1, 7, 7)
    Y_ptr,       # output: (B, C, H, W)
    B, C, H, W,  # dims
    BLOCK_HW: tl.constexpr
):
    # Each program handles one (b, c) and tiles over H*W
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    # Iterate over tiles of spatial HW
    for start in range(0, H * W, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        hw_mask = offs < (H * W)
        h_idx = offs // W
        w_idx = offs % W

        acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

        # Unrolled 7x7 depthwise conv with padding=3
        for kh in range(0, 7):
            ih = h_idx + kh - 3
            valid_h = (ih >= 0) & (ih < H) & hw_mask
            for kw in range(0, 7):
                iw = w_idx + kw - 3
                valid = valid_h & (iw >= 0) & (iw < W)
                ptr = X_ptr + pid_b * (C * H * W) + pid_c * (H * W) + ih * W + iw
                x_val = tl.load(ptr, mask=valid, other=0.0)
                w_ptr = W_ptr + pid_c * (1 * 7 * 7) + kh * 7 + kw
                w_val = tl.load(w_ptr)
                acc += x_val * w_val

        # Store accumulated result
        y_ptr = Y_ptr + pid_b * (C * H * W) + pid_c * (H * W) + h_idx * W + w_idx
        tl.store(y_ptr, acc, mask=hw_mask)


@triton.jit
def permute_bchw_to_bhwc_kernel(X_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    # X_ptr: (B, C, H, W) row-major, OUT_ptr: (B, H, W, C)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    x_ptr = X_ptr + pid_b * (C * H * W) + pid_c * (H * W) + pid_h * W + pid_w
    out_ptr = OUT_ptr + pid_b * (H * W * C) + pid_h * (W * C) + pid_w * C + pid_c
    val = tl.load(x_ptr)
    tl.store(out_ptr, val)


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


@triton.jit
def matmul_forward_kernel(A_ptr, B_ptr, OUT_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # OUT[M, N] = A[M, K] @ B[K, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    out_ptrs = OUT_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def grn_forward_kernel(G_ptr, MEAN_ptr, NORM_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    # G_ptr: (B,1,1,C4) we reduce across C4 for each (b)
    pid_b = tl.program_id(0)
    total_sum = 0.0
    total_sumsq = 0.0
    for c in range(0, 4096, BLOCK):
        offs = c + tl.arange(0, BLOCK)
        mask = offs < N
        g = tl.load(G_ptr + pid_b * N + offs, mask=mask, other=0.0)
        total_sum += tl.sum(g, axis=0)
        total_sumsq += tl.sum(g * g, axis=0)
    mean = total_sum / N
    tl.store(MEAN_ptr + pid_b, mean)
    norm = 1.0 / tl.sqrt(mean + 1e-6)  # norm_features
    tl.store(NORM_ptr + pid_b, norm)
    # Scale: OUT = G * norm
    for c in range(0, 4096, BLOCK):
        offs = c + tl.arange(0, BLOCK)
        mask = offs < N
        g = tl.load(G_ptr + pid_b * N + offs, mask=mask, other=0.0)
        out = g * norm
        tl.store(OUT_ptr + pid_b * N + offs, out, mask=mask)


# =========================
# ModelNew: Triton forward
# =========================
class ModelNew(torch.nn.Module):
    def __init__(self, axes_and_scalars: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.B = axes_and_scalars["B"]
        self.H = axes_and_scalars["H"]
        self.W = axes_and_scalars["W"]
        self.C = 128
        self.C4 = self.C * 4
        self.eps = 1e-6
        self.drop_path_prob = 0.1

    def forward(self):
        # Allocate and initialize tensors using Triton
        # dwconv_weight: (C,1,7,7) ~ N(0, 1/sqrt(49))
        dwconv_weight = torch.empty((self.C, 1, 7, 7), device=self.device, dtype=torch.float32)
        Nw = self.C * 1 * 7 * 7
        normal_fill_kernel[(triton.cdiv(Nw, 1024),)](dwconv_weight, Nw, 0.0, (1.0 / 49) ** 0.5, BLOCK=1024)

        # layernorm_weight: (C,) ~


def run(*args):
    return ModelNew()(*args)
