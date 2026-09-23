import torch
import triton
import triton.language as tl


# Triton matmul kernel: C = A @ B
# A: [M, K], B: [K, N], C: [M, N]
# Load A, B as their native dtype, cast to float32, compute tl.dot in float32, store float32.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(C, A, B, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 32)

    acc = tl.zeros((128, 128), dtype=tl.float32)

    for k in range(0, K, 32):
        a_ptrs = A + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = B + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # cast to fp32 for dot
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # cast to fp32 for dot

        acc += tl.dot(a, b)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton pointwise SiLU: Y = X * sigmoid(X), compute in float32, store float32
@triton.jit
def _silu_kernel(Y, X, M, N, stride_xm, stride_xn, stride_ym, stride_yn):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptrs, y, mask=mask)


# Triton elementwise multiply: Z = A * B, compute in float32, store float32
@triton.jit
def _mul_kernel(Z, A, B, M, N, stride_am, stride_an, stride_bm, stride_bn, stride_zm, stride_zn):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = B + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    z_ptrs = Z + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)
    z = a * b
    tl.store(z_ptrs, z, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 1408):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def forward(self, hidden_states: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
        """
        Triton-only forward:
        - shared_gate_output = hidden_states @ gate_weight   -> [M, N1]
        - shared_up_output    = hidden_states @ up_weight    -> [M, N1]
        - shared_activated    = silu(shared_gate_output) * shared_up_output -> [M, N1]
        Returns in bfloat16.
        """
        # Ensure contiguous tensors
        hidden = hidden_states.contiguous()   # [M, K]
        gate_w = gate_weight.contiguous()     # [N1, K]
        up_w = up_weight.contiguous()         # [N1, K]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N1 = gate_w.shape[0]  # intermediate_size

        # 1) Compute shared_gate_output = hidden @ gate_w -> [M, N1], float32
        shared_gate_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid1 = (triton.cdiv(M, 128), triton.cdiv(N1, 128))
        _matmul_kernel[grid1](
            shared_gate_output, hidden, gate_w,
            M, N1, K,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            shared_gate_output.stride(0), shared_gate_output.stride(1),
        )

        # 2) Compute shared_up_output = hidden @ up_w -> [M, N1], float32
        shared_up_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid2 = (triton.cdiv(M, 128), triton.cdiv(N1, 128))
        _matmul_kernel[grid2](
            shared_up_output, hidden, up_w,
            M, N1, K,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            shared_up_output.stride(0), shared_up_output.stride(1),
        )

        # 3) Compute silu(shared_gate_output) -> [M, N1], float32
        silu_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid3 = (triton.cdiv(M, 128), triton.cdiv(N1, 128))
        _silu_kernel[grid3](
            silu_output, shared_gate_output,
            M, N1, shared_gate_output.stride(0), shared_gate_output.stride(1),
            silu_output.stride(0), silu_output.stride(1),
        )

        # 4) Compute activated = silu_output * shared_up_output -> [M, N1], float32
        activated = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid4 = (triton.cdiv(M, 128), triton.cdiv(N1, 128))
        _mul_kernel[grid4](
            activated, silu_output, shared_up_output,
            M, N1, silu_output.stride(0), silu_output.stride(1),
            shared_up_output.stride(0), shared_up_output.stride(1),
            activated.stride(0), activated.stride(1),
        )

        # Return in bfloat16 to match typical input dtype
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
