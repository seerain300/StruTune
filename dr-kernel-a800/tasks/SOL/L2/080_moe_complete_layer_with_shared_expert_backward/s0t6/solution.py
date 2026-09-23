import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    C_ptr, A_ptr, B_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * A_stride_m + (k + offs_k[None, :]) * A_stride_k)
        b_ptrs = B_ptr + ((k + offs_k[:, None]) * B_stride_k + offs_n[None, :] * B_stride_n)
        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        # Load tiles as fp32
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        # Fused multiply-add
        acc += tl.dot(a, b)

    # Store result to C (fp32)
    c_ptrs = C_ptr + (offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _silu_kernel(out_ptr, x_ptr, M, N, out_stride_m, out_stride_n, x_stride_m, x_stride_n):
    # 2D tiling over [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_m = 64
    tile_n = 64
    offs_m = pid_m * tile_m + tl.arange(0, tile_m)
    offs_n = pid_n * tile_n + tl.arange(0, tile_n)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs_m[:, None] * x_stride_m + offs_n[None, :] * x_stride_n, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    silu_x = x_fp32 * tl.sigmoid(x_fp32)
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, silu_x, mask=mask)


@triton.jit
def _mul_kernel(out_ptr, a_ptr, b_ptr, M, N, a_stride_m, a_stride_n, b_stride_m, b_stride_n, out_stride_m, out_stride_n):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_m = 64
    tile_n = 64
    offs_m = pid_m * tile_m + tl.arange(0, tile_m)
    offs_n = pid_n * tile_n + tl.arange(0, tile_n)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptr + offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs_m[:, None] * b_stride_m + offs_n[None, :] * b_stride_n, mask=mask, other=0.0).to(tl.float32)
    c = a * b
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, c, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        """
        Compute the shared expert output: activated = silu(hidden @ gate_weight) * (hidden @ up_weight)

        Inputs:
        - hidden_states: [M, hidden_size], dtype=torch.bfloat16 or torch.float32, device GPU
        - shared_expert_gate_weight: [N1, hidden_size], dtype bfloat16/float32
        - shared_expert_up_weight: [N1, hidden_size], dtype bfloat16/float32

        Output:
        - activated: [M, N1], bfloat16
        """
        assert hidden_states.is_cuda, "Inputs must be on CUDA device for Triton."
        device = hidden_states.device

        M = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        N1 = shared_expert_gate_weight.shape[0]

        # Ensure contiguity and fp32 for matmul
        hidden_f32 = hidden_states.contiguous().to(torch.float32)
        gate_w_f32 = shared_expert_gate_weight.contiguous().to(torch.float32)  # [N1, hidden_size]
        up_w_f32 = shared_expert_up_weight.contiguous().to(torch.float32)     # [N1, hidden_size]

        # 1) shared_gate_output = hidden @ gate_w  -> [M, N1] using Triton matmul
        shared_gate_output = torch.empty((M, N1), dtype=torch.float32, device=device)
        grid1 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _matmul_kernel[grid1](
            shared_gate_output, hidden_f32, gate_w_f32,
            M, N1, hidden_size,
            hidden_f32.stride(0), hidden_f32.stride(1),
            gate_w_f32.stride(1), gate_w_f32.stride(0),  # B's strides: (k, n) => (stride_k, stride_n)
            shared_gate_output.stride(0), shared_gate_output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # 2) shared_up_output = hidden @ up_w  -> [M, N1] using Triton matmul
        shared_up_output = torch.empty((M, N1), dtype=torch.float32, device=device)
        grid2 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _matmul_kernel[grid2](
            shared_up_output, hidden_f32, up_w_f32,
            M, N1, hidden_size,
            hidden_f32.stride(0), hidden_f32.stride(1),
            up_w_f32.stride(1), up_w_f32.stride(0),  # (k, n)
            shared_up_output.stride(0), shared_up_output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # 3) Compute silu(shared_gate_output) -> [M, N1], using Triton elementwise kernel (fp32)
        silu_output = torch.empty((M, N1), dtype=torch.float32, device=device)
        grid3 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _silu_kernel[grid3](
            silu_output, shared_gate_output,
            M, N1, silu_output.stride(0), silu_output.stride(1),
            shared_gate_output.stride(0), shared_gate_output.stride(1),
        )

        # 4) Compute activated = silu_output * shared_up_output -> [M, N1], using Triton elementwise kernel (fp32)
        activated = torch.empty((M, N1), dtype=torch.float32, device=device)
        grid4 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
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
