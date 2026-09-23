import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_fp32(out_ptr, A_ptr, B_ptr,
                 M, N, K,
                 A_stride_m, A_stride_k,
                 B_stride_k, B_stride_n,
                 out_stride_m, out_stride_n,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D launch: each program handles a tile [BLOCK_M x BLOCK_N] of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Pointers for A and B tiles
        a_ptrs = A_ptr + offs_m[:, None] * A_stride_m + offs_k[None, :] * A_stride_k  # [BM, BK]
        b_ptrs = B_ptr + offs_k[:, None] * B_stride_k + offs_n[None, :] * B_stride_n  # [BK, BN]
        # Masks for loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        # Load and cast to fp32
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BM, BK]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BK, BN]
        # Accumulate
        acc += tl.dot(a, b)

    # Write back with mask
    c_ptrs = out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
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
    # Compute SiLU in fp32
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
    a = tl.load(a_ptr + offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_m[:, None] * b_stride_m + offs_n[None, :] * b_stride_n, mask=mask, other=0.0)
    # Multiply in fp32
    c = a.to(tl.float32) * b.to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, c, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                shared_expert_gate_weight: torch.Tensor,
                shared_expert_up_weight: torch.Tensor):
        # hidden_states: [M, hidden_size], typically bf16
        # gate_weight, up_weight: [N, hidden_size], typically bf16
        # We will compute in fp32 using Triton, then cast output to bf16.

        # Ensure contiguous for simple strides
        hidden = hidden_states.contiguous()
        gate_w = shared_expert_gate_weight.contiguous()
        up_w = shared_expert_up_weight.contiguous()

        M = hidden.shape[0]
        K = hidden.shape[1]
        N1 = gate_w.shape[1]  # intermediate_size

        # 1) shared_gate_output = hidden @ gate_w -> [M, N1] (fp32)
        shared_gate_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid1 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _matmul_fp32[grid1](
            shared_gate_output, hidden, gate_w,
            M, N1, K,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            shared_gate_output.stride(0), shared_gate_output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 2) shared_up_output = hidden @ up_w -> [M, N1] (fp32)
        shared_up_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid2 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _matmul_fp32[grid2](
            shared_up_output, hidden, up_w,
            M, N1, K,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            shared_up_output.stride(0), shared_up_output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 3) silu(shared_gate_output) -> [M, N1], float32
        silu_output = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid3 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _silu_kernel[grid3](
            silu_output, shared_gate_output,
            M, N1, shared_gate_output.stride(0), shared_gate_output.stride(1),
            silu_output.stride(0), silu_output.stride(1)
        )

        # 4) activated = silu_output * shared_up_output -> [M, N1], float32
        activated = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid4 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _mul_kernel[grid4](
            activated, silu_output, shared_up_output,
            M, N1, silu_output.stride(0), silu_output.stride(1),
            shared_up_output.stride(0), shared_up_output.stride(1),
            activated.stride(0), activated.stride(1)
        )

        # Return in bfloat16 to match typical input dtype
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
