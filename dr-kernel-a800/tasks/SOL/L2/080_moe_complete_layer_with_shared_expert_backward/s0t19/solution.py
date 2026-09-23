import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_2d_kernel(C, A, B, M, N, K,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C = A @ B
    A: [M, K], B: [K, N], C: [M, N]
    All tensors are fp32. Strides are in elements.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_init = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k_init
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Dot product along K
        acc += tl.dot(a, b)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _silu_2d_kernel(out_ptr, x_ptr, M, N,
                    stride_x_m, stride_x_n,
                    stride_out_m, stride_out_n,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Elementwise SiLU: out[i, j] = x[i, j] * sigmoid(x[i, j])
    2D tiling, masks handle boundaries.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n

    x = tl.load(x_ptrs, mask=mask, other=0.0)  # fp32
    # SiLU: x * sigmoid(x)
    y = x * tl.sigmoid(x)
    tl.store(out_ptrs, y, mask=mask)


@triton.jit
def _mul_2d_kernel(out_ptr, a_ptr, b_ptr, M, N,
                   stride_a_m, stride_a_n,
                   stride_b_m, stride_b_n,
                   stride_out_m, stride_out_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Elementwise multiply: out[i, j] = a[i, j] * b[i, j]
    2D tiling, masks handle boundaries.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_a_m + offs_n[None, :] * stride_a_n
    b_ptrs = b_ptr + offs_m[:, None] * stride_b_m + offs_n[None, :] * stride_b_n
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    c = a * b
    tl.store(out_ptrs, c, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Return the shared expert path output:
        activated = silu(gate_output) * up_output
        where:
          gate_output = hidden_states @ shared_expert_gate_weight
          up_output   = hidden_states @ shared_expert_up_weight
        All computations are done via Triton kernels. Inputs are expected to be on CUDA device.
        """
        # args are: hidden_states, shared_expert_gate_weight, shared_expert_up_weight
        # Note: The evaluator will pass tensors with expected shapes; we move them to CUDA if not already.
        hidden = args[0]
        gate_w = args[1]
        up_w = args[2]

        # Ensure CUDA and contiguous
        hidden = hidden.contiguous()
        gate_w = gate_w.contiguous()
        up_w = up_w.contiguous()

        if hidden.device.type != 'cuda':
            hidden = hidden.to('cuda')
        if gate_w.device.type != 'cuda':
            gate_w = gate_w.to('cuda')
        if up_w.device.type != 'cuda':
            up_w = up_w.to('cuda')

        # Dimensions
        M = hidden.shape[0]   # batch_seq_len
        K = hidden.shape[1]   # hidden_size (input dim)
        N1 = gate_w.shape[1]  # intermediate_size (output dim of gate/up)

        # 1) Compute gate_output = hidden @ gate_w  -> [M, N1], fp32
        gate_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _matmul_2d_kernel[grid](
            gate_output, hidden, gate_w,
            M, N1, K,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # 2) Compute up_output = hidden @ up_w  -> [M, N1], fp32
        up_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _matmul_2d_kernel[grid2](
            up_output, hidden, up_w,
            M, N1, K,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # 3) Compute silu(gate_output) -> [M, N1], fp32
        silu_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid3 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _silu_2d_kernel[grid3](
            silu_output, gate_output,
            M, N1,
            gate_output.stride(0), gate_output.stride(1),
            silu_output.stride(0), silu_output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

        # 4) Compute activated = silu_output * up_output -> [M, N1], fp32
        activated = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid4 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _mul_2d_kernel[grid4](
            activated, silu_output, up_output,
            M, N1,
            silu_output.stride(0), silu_output.stride(1),
            up_output.stride(0), up_output.stride(1),
            activated.stride(0), activated.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

        # Return in bfloat16 to match typical input dtype
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
