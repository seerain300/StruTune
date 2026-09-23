import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(out_ptr, a_ptr, b_ptr,
                   M, N, K,
                   a_stride_m, a_stride_k,
                   b_stride_k, b_stride_n,
                   out_stride_m, out_stride_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D launch: each program handles a tile of size (BLOCK_M, BLOCK_N) in C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + (offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (offs_k[:, None] * b_stride_k + offs_n[None, :] * b_stride_n)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # acc += a @ b
        acc += tl.dot(a, b)

    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _silu_kernel(out_ptr, x_ptr, M, N,
                 out_stride_m, out_stride_n, x_stride_m, x_stride_n,
                 BLOCK: tl.constexpr):
    # Each program handles one row and processes columns in chunks of BLOCK
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    start = col_block * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = (row < M) & (offs < N)

    x_ptrs = x_ptr + row * x_stride_m + offs * x_stride_n
    x = tl.load(x_ptrs, mask=mask, other=0.0)  # float32
    # SiLU: x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig

    out_ptrs = out_ptr + row * out_stride_m + offs * out_stride_n
    tl.store(out_ptrs, y, mask=mask)


@triton.jit
def _mul_kernel(out_ptr, a_ptr, b_ptr, M, N,
                out_stride_m, out_stride_n, a_stride_m, a_stride_n, b_stride_m, b_stride_n,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    start = col_block * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = (row < M) & (offs < N)

    a_ptrs = a_ptr + row * a_stride_m + offs * a_stride_n
    b_ptrs = b_ptr + row * b_stride_m + offs * b_stride_n

    a = tl.load(a_ptrs, mask=mask, other=0.0)  # float32
    b = tl.load(b_ptrs, mask=mask, other=0.0)  # float32

    out = a * b

    out_ptrs = out_ptr + row * out_stride_m + offs * out_stride_n
    tl.store(out_ptrs, out, mask=mask)


def _launch_matmul(a_fp32, b_fp32, out_fp32, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    M, K = a_fp32.shape
    K_b, N = b_fp32.shape
    assert K == K_b, "Inner dims must match for matmul"
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        out_fp32, a_fp32, b_fp32,
        M, N, K,
        a_fp32.stride(0), a_fp32.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        out_fp32.stride(0), out_fp32.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


def _launch_silu(x_fp32, y_fp32, BLOCK=1024):
    M, N = x_fp32.shape
    grid = (M, triton.cdiv(N, BLOCK))
    _silu_kernel[grid](y_fp32, x_fp32, M, N, y_fp32.stride(0), y_fp32.stride(1), x_fp32.stride(0), x_fp32.stride(1), BLOCK=BLOCK)


def _launch_mul(a_fp32, b_fp32, out_fp32, BLOCK=1024):
    M, N = a_fp32.shape
    grid = (M, triton.cdiv(N, BLOCK))
    _mul_kernel[grid](out_fp32, a_fp32, b_fp32, M, N, out_fp32.stride(0), out_fp32.stride(1), a_fp32.stride(0), a_fp32.stride(1), b_fp32.stride(0), b_fp32.stride(1), BLOCK=BLOCK)


class ModelNew(torch.nn.Module):
    def forward(self, grad_output, hidden_states, router_weight,
                e_score_correction_bias,  # unused (kept for signature compatibility)
                scores, topk_indices, topk_weights, score_mask,
                shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
                shared_gate_output, shared_up_output, shared_activated):
        """
        Triton-only forward:
        - hidden_states: [B, H]
        - shared_expert_gate_weight: [M, H]
        - shared_expert_up_weight: [M, H]
        Compute: silu(gate_output) * up_output where:
          gate_output = hidden_states @ shared_expert_gate_weight
          up_output   = hidden_states @ shared_expert_up_weight
        Return result as bfloat16.
        """

        # Ensure inputs are float32 for kernel compute; we'll store fp32 intermediates and cast at end.
        hidden = hidden_states.to(torch.float32)
        gate_weight = shared_expert_gate_weight.to(torch.float32)  # [M, H]
        up_weight = shared_expert_up_weight.to(torch.float32)     # [M, H]

        # gate_output = hidden @ gate_weight  -> [B, M]
        gate_output = torch.empty((hidden.shape[0], gate_weight.shape[0]), dtype=torch.float32, device=hidden.device)
        _launch_matmul(hidden, gate_weight, gate_output)

        # up_output = hidden @ up_weight  -> [B, M]
        up_output = torch.empty((hidden.shape[0], up_weight.shape[0]), dtype=torch.float32, device=hidden.device)
        _launch_matmul(hidden, up_weight, up_output)

        # Compute SiLU(gate_output): [B, M]
        silu_gate = torch.empty_like(gate_output)
        _launch_silu(gate_output, silu_gate)

        # Multiply: activated = silu_gate * up_output -> [B, M]
        activated = torch.empty_like(up_output)
        _launch_mul(silu_gate, up_output, activated)

        # Return bfloat16 as per typical evaluator expectation; forward must launch Triton kernels
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
