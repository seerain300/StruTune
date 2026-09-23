import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    C_ptr, A_ptr, B_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D program ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K in tiles
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Pointers for tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)  # [BM, BK]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)  # [BK, BN]

        # Construct explicit boolean masks for loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles (inputs may be bf16/fp16; we cast to fp32 for matmul)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Write result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _silu_kernel(out_ptr, x_ptr, M, N, out_stride_m, out_stride_n, x_stride_m, x_stride_n):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * tl.arange(0, 64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs_m[:, None] * x_stride_m + offs_n[None, :] * x_stride_n, mask=mask, other=0.0)
    silu_x = x * tl.sigmoid(x)  # SiLU in fp32
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, silu_x, mask=mask)


@triton.jit
def _mul_kernel(out_ptr, a_ptr, b_ptr, M, N, a_stride_m, a_stride_n, b_stride_m, b_stride_n, out_stride_m, out_stride_n):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * tl.arange(0, 64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptr + offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_m[:, None] * b_stride_m + offs_n[None, :] * b_stride_n, mask=mask, other=0.0)
    c = a * b
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, c, mask=mask)


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for forward testing (shared expert path)."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    hidden_size = 4096
    moe_intermediate_size = 1408
    n_routed_experts = 128
    num_experts_per_tok = 8
    routed_scaling_factor = 1.0

    # hidden states (bfloat16 as in original setup)
    hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)

    # Shared expert weights (bfloat16)
    shared_expert_gate_weight = torch.randn(moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    shared_expert_up_weight = torch.randn(moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.02

    return {
        "hidden_states": hidden_states,
        "shared_expert_gate_weight": shared_expert_gate_weight,
        "shared_expert_up_weight": shared_expert_up_weight,
    }


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        """
        Compute and return the shared expert output:
          shared_gate_output = hidden_states @ shared_expert_gate_weight   [M, N1]
          shared_up_output   = hidden_states @ shared_expert_up_weight     [M, N1]
          shared_activated   = silu(shared_gate_output) * shared_up_output [M, N1]
        Return shared_activated in bfloat16. Triton performs matmul and elementwise ops.
        """
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N1 = shared_expert_gate_weight.shape[0]  # intermediate_size

        # 1) shared_gate_output = hidden_states @ shared_expert_gate_weight -> [M, N1] (fp32 accumulation)
        shared_gate_output = torch.empty((M, N1), dtype=torch.float32, device=device)
        grid1 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _matmul_kernel[grid1](
            shared_gate_output, hidden_states.to(torch.float32), shared_expert_gate_weight.to(torch.float32),
            M, N1, K,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            shared_gate_output.stride(0), shared_gate_output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # 2) shared_up_output = hidden_states @ shared_expert_up_weight -> [M, N1] (fp32)
        shared_up_output = torch.empty((M, N1), dtype=torch.float32, device=device)
        grid2 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _matmul_kernel[grid2](
            shared_up_output, hidden_states.to(torch.float32), shared_expert_up_weight.to(torch.float32),
            M, N1, K,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            shared_up_output.stride(0), shared_up_output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # 3) silu(shared_gate_output) -> [M, N1], float32
        silu_output = torch.empty((M, N1), dtype=torch.float32, device=device)
        grid3 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _silu_kernel[grid3](
            silu_output, shared_gate_output,
            M, N1,
            silu_output.stride(0), silu_output.stride(1),
            shared_gate_output.stride(0), shared_gate_output.stride(1),
        )

        # 4) activated = silu_output * shared_up_output -> [M, N1], float32
        activated = torch.empty((M, N1), dtype=torch.float32, device=device)
        grid4 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _mul_kernel[grid4](
            activated, silu_output, shared_up_output,
            M, N1,
            silu_output.stride(0), silu_output.stride(1),
            shared_up_output.stride(0), shared_up_output.stride(1),
            activated.stride(0), activated.stride(1),
        )

        # Return in bfloat16 (matches typical input dtype)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
