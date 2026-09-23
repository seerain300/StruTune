import torch
import triton
import triton.language as tl


# =========================
# Triton kernels: init / random fill
# =========================
@triton.jit
def normal_fill_kernel(OUT_ptr, N, MEAN, STD, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Generate normal via box-muller transform
    u = tl.rand(offsets)  # uniform in [0,1)
    v = tl.rand(offsets)  # uniform in [0,1)
    z = tl.sqrt(-2.0 * tl.log(1.0 - u)) * tl.sign(2.0 * v - 1.0)  # N(0,1)
    val = MEAN + STD * z
    tl.store(OUT_ptr + offsets, val, mask=mask)


@triton.jit
def drop_mask_kernel(OUT_ptr, N, DROP_PROB, BLOCK: tl.constexpr, seed: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Simple LCG RNG for reproducibility
    s = seed * offsets + 1013904223
    rnd = (s >> 32) * 1.0 / 4294967296.0
    keep = rnd > DROP_PROB
    val = tl.where(keep, 1.0, 0.0)
    tl.store(OUT_ptr + offsets, val, mask=mask)


# =========================
# Triton elementwise: GELU forward
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
# Triton LayerNorm forward (across last dim C): per (N,H,W)
# We compute per-(b,h,w) mean and inv_std across C, then normalize in-place.
# Note: This kernel expects X to be (B,H,W,C) contiguous, and writes OUT with same layout.
# =========================
@triton.jit
def layer_norm_forward_kernel(
    X_ptr,        # (B, H, W, C) contiguous
    MEAN_ptr,     # (B, H, W)
    INV_STD_ptr,  # (B, H, W)
    OUT_ptr,      # (B, H, W, C) contiguous
    B, H, W, C,
    BLOCK_C: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // (H * W)
    rem = pid % (H * W)
    h = rem // W
    w = rem % W

    # Compute mean across channels
    sum_val = 0.0
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        base = (b * H + h) * W + w
        x = tl.load(X_ptr + base + offs_c, mask=mask_c, other=0.0)
        sum_val += tl.sum(x, axis=0)

    mean = sum_val / C
    mean_store_index = b * (H * W) + h * W + w
    tl.store(MEAN_ptr + mean_store_index, mean)

    # Compute variance across channels
    var_sum = 0.0
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        base = (b * H + h) * W + w
        x = tl.load(X_ptr + base + offs_c, mask=mask_c, other=0.0)
        diff = x - mean
        var_sum += tl.sum(diff * diff, axis=0)

    var = var_sum / C
    inv_std = 1.0 / tl.sqrt(var + 1e-6)
    tl.store(INV_STD_ptr + mean_store_index, inv_std)

    # Normalize and store
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        base_in = (b * H + h) * W + w
        x = tl.load(X_ptr + base_in + offs_c, mask=mask_c, other=0.0)
        y = (x - mean) * inv_std
        base_out = (b * H + h) * W + w + offs_c
        tl.store(OUT_ptr + base_out, y, mask=mask_c)


# =========================
# Triton matmul: (M,K) @ (K,N) -> (M,N), forward
# =========================
@triton.jit
def matmul_forward_kernel(
    A_ptr,         # (M, K) row-major
    B_ptr,         # (K, N) row-major
    OUT_ptr,       # (M, N) row-major
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :], mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :], mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    tl.store(OUT_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# =========================
# ModelNew.forward: Triton-enabled forward
# =========================
class ModelNew(torch.nn.Module):
    def __init__(self, B: int, H: int, W: int, axes_and_scalars: dict):
        super().__init__()
        self.B = B
        self.H = H
        self.W = W
        self.C = 128
        self.C4 = self.C * 4
        self.eps = 1e-6
        self.drop_path_prob = axes_and_scalars.get("drop_path_prob", 0.1)
        self.device = torch.device("cuda")

    def forward(self):
        # 1) Initialize weights and inputs via Triton random fill
        # dwconv_weight: (C,1,7,7) ~ N(0, 1/sqrt(49))
        dwconv_weight = torch.empty((self.C, 1, 7, 7), device=self.device, dtype=torch.float32)
        N = self.C * 1 * 7 * 7
        normal_fill_kernel[(triton.cdiv(N, 1024),)](dwconv_weight, N, 0.0, (1.0 / 49.0) ** 0.5, BLOCK=1024)

        # layernorm_weight: (C) ~ N(1, 0.01)
        layernorm_weight = torch.empty((self.C,), device=self.device, dtype=torch.float32)
        ones_fill_kernel[(self.C,)](layernorm_weight, self.C, BLOCK=1, MEAN=1.0, STD=0.01)  # trivial ones_fill variant

        # pwconv1_weight: (4C, C) ~ N(0, sqrt(2/C))
        pwconv1_weight = torch.empty((self.C4, self.C), device=self.device, dtype=torch.float32)
        N1 = self.C4 * self.C
        normal_fill_kernel[(triton.cdiv(N1, 1024),)](pwconv1_weight, N1, 0.0, (2.0 / self.C) ** 0.5, BLOCK=1024)

        # grn_weight: (1,1,1,4C) small normal
        grn_weight = torch.empty((1, 1, 1, self.C4), device=self.device, dtype=torch.float32)
        normal_fill_kernel[(self.C4,)](grn_weight, self.C4, 0.0, 0.01, BLOCK=1)

        # pwconv2_weight: (C, 4C) ~ N(0, sqrt(2/4C))
        pwconv2_weight = torch.empty((self.C, self.C4), device=self.device, dtype=torch.float32)
        N2 = self.C * self.C4
        normal_fill_kernel[(triton.cdiv(N2, 1024),)](pwconv2_weight, N2, 0.0, (2.0 / self.C4) ** 0.5, BLOCK=1024)

        # 2) Residual and grad_output
        residual = torch.empty((self.B, self.C, self.H, self.W), device=self.device, dtype=torch.float32)
        normal_fill_kernel[(triton.cdiv(self.B * self.C * self.H * self.W, 1024),)](
            residual, self.B * self.C * self.H * self.W, 0.0, 0.1, BLOCK=1024
        )

        grad_output = torch.empty((self.B, self.C, self.H, self.W), device=self.device, dtype=torch.float32)
        normal_fill_kernel[(triton.cdiv(self.B * self.C * self.H * self.W, 1024),)](
            grad_output, self.B * self.C * self.H * self.W, 0.0, 1.0, BLOCK=1024
        )

        # 3) Drop mask: (B,1,1,1)
        drop_mask = torch.empty((self.B, 1, 1, 1), device=self.device, dtype=torch.float32)
        drop_mask_kernel[(self.B,)](drop_mask, self.B, self.drop_path_prob, BLOCK=1, seed=1234)

        # 4) Depthwise conv: use PyTorch to avoid Triton OOB. Padding


def run(*args):
    return ModelNew()(*args)
