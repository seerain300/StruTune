import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(C, A, B, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    C = A @ B
    A: [M, K], B: [K, N], C: [M, N]
    Triton 2D tiling with masks for boundaries.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_remaining = K
    while k_remaining > 0:
        k_chunk = tl.minimum(k_remaining, BLOCK_K)
        k_mask = offs_k < k_chunk
        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N) & k_mask[None, :]
        b_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N) & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_remaining -= BLOCK_K

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _silu_1d_kernel(out_ptr, x_ptr, M, N, stride_x_m, stride_x_n, stride_out_m, stride_out_n, BLOCK: tl.constexpr):
    """
    Elementwise SiLU: out[i, j] = x[i, j] * sigmoid(x[i, j])
    1D tiling over columns for each row.
    """
    pid_m = tl.program_id(0)
    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + pid_m * stride_x_m + offs * stride_x_n, mask=mask, other=0.0)
    y = x * tl.sigmoid(x)  # Triton provides tl.sigmoid
    tl.store(out_ptr + pid_m * stride_out_m + offs * stride_out_n, y, mask=mask)


@triton.jit
def _mul_1d_kernel(out_ptr, a_ptr, b_ptr, M, N, stride_a_m, stride_a_n, stride_b_m, stride_b_n, stride_out_m, stride_out_n, BLOCK: tl.constexpr):
    """
    Elementwise multiply: out[i, j] = a[i, j] * b[i, j]
    1D tiling over columns for each row.
    """
    pid_m = tl.program_id(0)
    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(a_ptr + pid_m * stride_a_m + offs * stride_a_n, mask=mask, other=0.0)
    b = tl.load(b_ptr + pid_m * stride_b_m + offs * stride_b_n, mask=mask, other=0.0)
    c = a * b
    tl.store(out_ptr + pid_m * stride_out_m + offs * stride_out_n, c, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, grad_output: torch.Tensor,
                hidden_states: torch.Tensor,
                shared_expert_gate_weight: torch.Tensor,
                shared_expert_up_weight: torch.Tensor):
        """
        Compute shared expert output:
        gate_output = F.linear(hidden_states, shared_expert_gate_weight)
        up_output   = F.linear(hidden_states, shared_expert_up_weight)
        activated   = silu(gate_output) * up_output
        Returns activated (cast to bfloat16).
        Triton is used for all matmuls and elementwise ops.
        """
        # Ensure tensors are on CUDA; compute in float32
        device = hidden_states.device
        hidden = hidden_states if device.type == "cuda" else hidden_states.to("cuda")
        gate_w = shared_expert_gate_weight if device.type == "cuda" else shared_expert_gate_weight.to("cuda")
        up_w = shared_expert_up_weight if device.type == "cuda" else shared_expert_up_weight.to("cuda")

        hidden_f32 = hidden.to(torch.float32)
        gate_w_f32 = gate_w.to(torch.float32)
        up_w_f32 = up_w.to(torch.float32)

        M = hidden_f32.shape[0]  # batch_seq_len
        K = hidden_f32.shape[1]  # hidden_size
        N1 = gate_w_f32.shape[1]  # intermediate_size (1408)

        # Allocate outputs for matmul
        gate_output = torch.empty((M, N1), dtype=torch.float32, device=device)  # [M, N1]
        up_output = torch.empty((M, N1), dtype=torch.float32, device=device)    # [M, N1]

        # Triton: gate_output = hidden @ gate_w
        grid = (triton.cdiv(M, 128), triton.cdiv(N1, 128))
        _matmul_kernel[grid](
            gate_output, hidden_f32, gate_w_f32,
            M, N1, K,
            hidden_f32.stride(0), hidden_f32.stride(1),
            gate_w_f32.stride(0), gate_w_f32.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        )

        # Triton: up_output = hidden @ up_w
        _matmul_kernel[grid](
            up_output, hidden_f32, up_w_f32,
            M, N1, K,
            hidden_f32.stride(0), hidden_f32.stride(1),
            up_w_f32.stride(0), up_w_f32.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        )

        # Triton: silu(gate_output) -> silu_gate
        silu_gate = torch.empty_like(gate_output, dtype=torch.float32, device=device)
        grid_silu = (M, triton.cdiv(N1, 1024))
        _silu_1d_kernel[grid_silu](
            silu_gate, gate_output,
            M, N1,
            gate_output.stride(0), gate_output.stride(1),
            silu_gate.stride(0), silu_gate.stride(1),
            BLOCK=1024,
        )

        # Triton: activated = silu_gate * up_output
        activated = torch.empty_like(gate_output, dtype=torch.float32, device=device)
        grid_mul = (M, triton.cdiv(N1, 1024))
        _mul_1d_kernel[grid_mul](
            activated, silu_gate, up_output,
            M, N1,
            silu_gate.stride(0), silu_gate.stride(1),
            up_output.stride(0), up_output.stride(1),
            activated.stride(0), activated.stride(1),
            BLOCK=1024,
        )

        # Return in bfloat16 to match typical input dtype
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
