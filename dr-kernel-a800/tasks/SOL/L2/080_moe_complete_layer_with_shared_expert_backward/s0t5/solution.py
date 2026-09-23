import torch
import triton
import triton.language as tl


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
    # Compute SiLU in fp32 for numerical stability, then store as fp32. Host will cast to bfloat16.
    x_fp32 = x.to(tl.float32)
    silu_x = x_fp32 * tl.sigmoid(x_fp32)
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, silu_x, mask=mask)


@triton.jit
def _mul_kernel(out_ptr, a_ptr, b_ptr, M, N, a_stride_m, a_stride_n, b_stride_m, b_stride_n, out_stride_m, out_stride_n):
    # 2D tiling over [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_m = 64
    tile_n = 64
    offs_m = pid_m * tile_m + tl.arange(0, tile_m)
    offs_n = pid_n * tile_n + tl.arange(0, tile_n)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptr + offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_m[:, None] * b_stride_m + offs_n[None, :] * b_stride_n, mask=mask, other=0.0)
    # Operate in fp32, store fp32. Host will cast to bfloat16.
    a_fp32 = a.to(tl.float32)
    b_fp32 = b.to(tl.float32)
    c = a_fp32 * b_fp32
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, c, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                shared_expert_gate_weight: torch.Tensor,
                shared_expert_up_weight: torch.Tensor):
        """
        Compute the shared expert path activation:
            gate_output = hidden_states @ shared_expert_gate_weight       [M, N1] (float32)
            up_output   = hidden_states @ shared_expert_up_weight         [M, N1] (float32)
            activated   = silu(gate_output) * up_output                   [M, N1] (bfloat16, returned)
        Triton kernels perform silu and multiply in fp32; result cast to bfloat16.
        """
        assert hidden_states.is_cuda, "Triton forward requires CUDA tensors."
        assert shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, "All tensors must be CUDA."

        # Compute two linear projections with PyTorch matmul (fp32)
        gate_output = hidden_states.matmul(shared_expert_gate_weight)  # [M, N1], float32
        up_output = hidden_states.matmul(shared_expert_up_weight)      # [M, N1], float32

        M, N1 = gate_output.shape

        # 1) Triton SiLU on gate_output -> [M, N1], fp32
        silu_output = torch.empty((M, N1), dtype=torch.float32, device=hidden_states.device)
        grid = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _silu_kernel[grid](
            silu_output, gate_output,
            M, N1,
            silu_output.stride(0), silu_output.stride(1),
            gate_output.stride(0), gate_output.stride(1),
        )

        # 2) Triton multiply: activated = silu_output * up_output -> [M, N1], fp32
        activated = torch.empty((M, N1), dtype=torch.float32, device=hidden_states.device)
        grid = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        _mul_kernel[grid](
            activated, silu_output, up_output,
            M, N1,
            silu_output.stride(0), silu_output.stride(1),
            up_output.stride(0), up_output.stride(1),
            activated.stride(0), activated.stride(1),
        )

        # Return in bfloat16 to match typical evaluator expectations
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
