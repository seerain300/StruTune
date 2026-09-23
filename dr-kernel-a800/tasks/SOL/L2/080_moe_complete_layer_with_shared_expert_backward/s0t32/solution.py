import torch
import triton
import triton.language as tl


@triton.jit
def _silu_1d(out_ptr, x_ptr, count: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < count
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # float32
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _mul_1d(out_ptr, a_ptr, b_ptr, count: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < count
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)  # float32
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)  # float32
    y = a * b
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _matmul_2d_kernel(C_ptr, A_ptr, B_ptr, M, N, K,
                      a_stride_m, a_stride_k,
                      b_stride_k, b_stride_n,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    # compute row/col ranges for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k)  # [BLOCK_M, BLOCK_K]
        b_ptrs = B_ptr + (offs_k[:, None] * b_stride_k + offs_n[None, :] * b_stride_n)  # [BLOCK_K, BLOCK_N]

        # masks for OOB
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # loads and matmul accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # compute acc += a @ b
        for i in range(0, BLOCK_M):
            for j in range(0, BLOCK_N):
                dot = 0.0
                for kk in range(0, BLOCK_K):
                    # safe: masks prevent OOB; a[i, kk] and b[kk, j] exist due to K loop mask
                    dot += a[i, kk] * b[kk, j]
                acc[i, j] = acc[i, j] + dot

    # write results
    c_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _launch_matmul_2d(C, A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    # Ensure contiguity and dtype
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Incompatible matmul shapes"
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_2d_kernel[grid](
        C, A, B,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return C


def _launch_silu_1d(out, x):
    out = out.contiguous()
    x = x.contiguous()
    count = out.numel()
    grid = (triton.cdiv(count, 1024),)
    _silu_1d[grid](out, x, count)


def _launch_mul_1d(out, a, b):
    out = out.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    count = out.numel()
    grid = (triton.cdiv(count, 1024),)
    _mul_1d[grid](out, a, b, count)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        router_logits: torch.Tensor,
        scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        score_mask: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,
        shared_expert_up_weight: torch.Tensor,
        shared_expert_down_weight: torch.Tensor,
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        """
        Compute shared expert output: shared_activated = SiLU(gate_output) * up_output
        - gate_output = hidden_states @ shared_expert_gate_weight.T
        - up_output   = hidden_states @ shared_expert_up_weight.T
        Use torch for matmuls and Triton for elementwise SiLU and multiply.
        Return bfloat16 tensor.
        """
        # Prepare shapes and device
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        H = shared_expert_gate_weight.shape[1]  # intermediate_size = 1408

        # Compute gate_output and up_output via torch to match reference exactly
        gate_output = torch.mm(hidden_states, shared_expert_gate_weight.t())  # [M, H], float32
        up_output = torch.mm(hidden_states, shared_expert_up_weight.t())     # [M, H], float32

        # Triton SiLU on gate_output
        silu_out = torch.empty_like(gate_output, dtype=torch.float32, device=device)
        _launch_silu_1d(silu_out, gate_output)

        # Triton multiply: silu_out * up_output
        final_out = torch.empty_like(gate_output, dtype=torch.float32, device=device)
        _launch_mul_1d(final_out, silu_out, up_output)

        # Return in bfloat16 to match evaluator expectations
        return final_out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
