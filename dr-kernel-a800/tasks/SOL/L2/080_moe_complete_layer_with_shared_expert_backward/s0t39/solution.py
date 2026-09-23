import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(out_ptr, a_ptr, b_ptr, M, N, K,
                   a_stride_m, a_stride_k,
                   b_stride_k, b_stride_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C[M, N] = A[M, K] @ B[K, N]
    All inputs/outputs are float32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + (offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (offs_k[:, None] * b_stride_k + offs_n[None, :] * b_stride_n)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = out_ptr + (offs_m[:, None] * 0 + offs_n[None, :] * 0)  # stride(0)=M, stride(1)=N
    # We need actual strides: out strides are (out.stride(0), out.stride(1))
    c_ptrs = out_ptr + (offs_m[:, None] * out_ptr.stride(0) + offs_n[None, :] * out_ptr.stride(1))
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _silu_kernel(out_ptr, x_ptr, M, N,
                 out_stride_m, out_stride_n,
                 x_stride_m, x_stride_n,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Elementwise: out = x * sigmoid(x) over a 2D tensor [M, N], float32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + (offs_m[:, None] * x_stride_m + offs_n[None, :] * x_stride_n)
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s

    out_ptrs = out_ptr + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)
    tl.store(out_ptrs, y, mask=mask)


@triton.jit
def _mul_kernel(out_ptr, a_ptr, b_ptr, M, N,
                out_stride_m, out_stride_n,
                a_stride_m, a_stride_n,
                b_stride_m, b_stride_n,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Elementwise: out = a * b over a 2D tensor [M, N], float32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    a_ptrs = a_ptr + (offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n)
    b_ptrs = b_ptr + (offs_m[:, None] * b_stride_m + offs_n[None, :] * b_stride_n)

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    out_ptrs = out_ptr + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)
    tl.store(out_ptrs, a * b, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        """
        Compute shared_activated = SiLU(gate_output) * up_output,
        where:
          gate_output = hidden_states @ shared_expert_gate_weight   # [M, H]
          up_output    = hidden_states @ shared_expert_up_weight    # [M, H]
        All computation is done via Triton kernels. Output is bfloat16.
        """
        assert hidden_states.is_cuda and shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, \
            "All inputs must be on CUDA device for Triton execution."

        # Ensure float32 for computation
        A = hidden_states  # [M, K]
        Wg = shared_expert_gate_weight  # [K, H]
        Wu = shared_expert_up_weight     # [K, H]
        M = A.shape[0]
        K = A.shape[1]
        H = Wg.shape[1]

        # Compute gate_output = A @ Wg
        gate_output = torch.empty((M, H), dtype=torch.float32, device=A.device)
        grid_g = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        _matmul_kernel[grid_g](
            gate_output, A, Wg,
            M, H, K,
            A.stride(0), A.stride(1),
            Wg.stride(0), Wg.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32
        )

        # Compute up_output = A @ Wu
        up_output = torch.empty((M, H), dtype=torch.float32, device=A.device)
        grid_u = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        _matmul_kernel[grid_u](
            up_output, A, Wu,
            M, H, K,
            A.stride(0), A.stride(1),
            Wu.stride(0), Wu.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32
        )

        # SiLU on gate_output
        silu_gate = torch.empty_like(gate_output)
        grid_silu = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        _silu_kernel[grid_silu](
            silu_gate, gate_output,
            M, H,
            silu_gate.stride(0), silu_gate.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_M=128, BLOCK_N=64
        )

        # Multiply SiLU(gate) with up_output
        activated = torch.empty_like(silu_gate)
        grid_mul = (triton.cdiv(M, 128), triton.cdiv(H, 64))
        _mul_kernel[grid_mul](
            activated, silu_gate, up_output,
            M, H,
            activated.stride(0), activated.stride(1),
            silu_gate.stride(0), silu_gate.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_M=128, BLOCK_N=64
        )

        # Return in bfloat16
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
