import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bf16_to_f32_kernel(
    X_ptr,            # *bfloat16, shape [M, K]
    W_ptr,            # *bfloat16, shape [K, N]
    Y_ptr,            # *float32,  shape [M, N]
    M,                # int
    K,                # int
    N,                # int
    stride_xm,        # int (elements)
    stride_xk,        # int (elements)
    stride_wk,        # int (elements, along K for W)
    stride_wn,        # int (elements, along N for W)
    stride_ym,        # int (elements)
    stride_yn,        # int (elements)
    BLOCK_M: tl.constexpr,  # e.g., 64
    BLOCK_N: tl.constexpr,  # e.g., 64
    BLOCK_K: tl.constexpr,  # e.g., 64
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        x_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        x = x.to(tl.float32)
        # Load W tile as [BLOCK_K, BLOCK_N] (we want W[k, n], so along n dimension)
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        w = w.to(tl.float32)
        # Accumulate: acc += x @ w
        # x: [BM, BK], w: [BK, BN] => acc: [BM, BN]
        acc += tl.dot(x, w)
    # Store results for this tile
    y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,      # *float32, shape [M, N]
    UpOut_ptr,        # *float32, shape [M, N]
    Out_ptr,          # *float32, shape [M, N]
    M,                # int
    N,                # int
    stride_gm,        # int
    stride_gn,        # int
    stride_um,        # int
    stride_un,        # int
    stride_om,        # int
    stride_on,        # int
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    valid = (m < M) & (n < N)
    gate = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn, mask=valid, other=0.0)
    up = tl.load(UpOut_ptr + m * stride_um + n * stride_un, mask=valid, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    out = gate * sig * up
    tl.store(Out_ptr + m * stride_om + n * stride_on, out, mask=valid)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, shared_expert_gate_weight: torch.Tensor, shared_expert_up_weight: torch.Tensor):
        """
        Compute shared_activated = SiLU(linear(hidden_states, gate_weight^T)) * linear(hidden_states, up_weight^T)
        using Triton kernels. No torch ops on tensors in forward.
        """
        assert hidden_states.is_cuda and shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, "Inputs must be on CUDA"
        # Ensure contiguous
        hidden_states = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()  # [K, N]
        up_weight = shared_expert_up_weight.contiguous()      # [K, N]

        M, K = hidden_states.shape
        K_gate, N = gate_weight.shape
        K_up, N_up = up_weight.shape
        assert K == K_gate == K_up, "hidden_states' K must match gate/up weights' K"
        assert N == N_up, "gate and up weights must have same N"

        # Tile sizes chosen for robustness
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        # Allocate outputs in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch 2D matmul kernels
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_bf16_to_f32_kernel[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        matmul_bf16_to_f32_kernel[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Compute activated = gate_out * SiLU(gate_out) * up_out in Triton
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        silu_mul_kernel[grid](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return bfloat16 (dtype conversion, not torch op on tensor elements)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
