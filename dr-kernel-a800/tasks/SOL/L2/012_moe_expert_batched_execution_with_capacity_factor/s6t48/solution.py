import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute C[M] = A[H] @ B[H, M], where A is a row vector
# One program handles a block of M outputs. We loop over H in tiles.
@triton.jit
def bmm_row_AxB_C(
    A_ptr, B_ptr, C_ptr,
    H, M,
    stride_A_row, stride_A_col,
    stride_B_row, stride_B_col,
    stride_C_row, stride_C_col,
    BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr
):
    # Each program handles a block of M outputs
    m_off = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over H dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        # Load A tile (vector length BLOCK_H)
        a = tl.load(
            A_ptr + 0 * stride_A_row + h_idx * stride_A_col,
            mask=h_idx < H,
            other=0.0
        )  # [BLOCK_H], use row 0
        # Load B tile (matrix [BLOCK_H, BLOCK_M])
        b = tl.load(
            B_ptr + h_idx[:, None] * stride_B_row + m_off[None, :] * stride_B_col,
            mask=(h_idx[:, None] < H) & (m_off[None, :] < M),
            other=0.0
        )
        # Accumulate: acc += sum over H of a * b
        acc += tl.sum(a[:, None] * b, axis=0)

    # Store results
    tl.store(C_ptr + 0 * stride_C_row + m_off * stride_C_col, acc, mask=m_off < M)


# Triton kernel: elementwise SiLU on vector X[M] -> Y[M]
@triton.jit
def silu_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X_ptr + offs, mask=offs < N, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(Y_ptr + offs, y, mask=offs < N)


# Triton kernel: compute E[H] = C[M] @ D[M, H], where C is a vector, D is a matrix
# One program handles a block of H outputs. We loop over M in tiles.
@triton.jit
def bmm_row_CxD_E(
    C_ptr, D_ptr, E_ptr,
    M, H,
    stride_C, stride_D_row, stride_D_col,
    stride_E,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    h_off = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for m_off in range(0, M, BLOCK_M):
        m_idx = m_off + tl.arange(0, BLOCK_M)
        # Load C tile (vector length BLOCK_M)
        c = tl.load(C_ptr + m_idx, mask=m_idx < M, other=0.0)
        # Load D tile (matrix [BLOCK_M, BLOCK_H])
        d = tl.load(
            D_ptr + m_idx[:, None] * stride_D_row + h_off[None, :] * stride_D_col,
            mask=(m_idx[:, None] < M) & (h_off[None, :] < H),
            other=0.0
        )
        # Accumulate: acc += sum over M of c * d
        acc += tl.sum(c[:, None] * d, axis=0)

    tl.store(E_ptr + h_off * stride_E, acc, mask=h_off < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        """
        hidden_states: [num_tokens, hidden_size], bfloat16
        selected_experts: [num_tokens, num_experts_per_tok], int64
        routing_weights: [num_tokens, num_experts_per_tok], bfloat16
        expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size], bfloat16
        expert_up_weights: [num_experts, hidden_size, moe_intermediate_size], bfloat16
        expert_down_weights: [num_experts, moe_intermediate_size, hidden_size], bfloat16
        Returns: [num_tokens, hidden_size], bfloat16
        """
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda \
            and expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
            "All tensors must be CUDA tensors."

        device = hidden_states.device
        dtype = hidden_states.dtype

        num_tokens, hidden_size = hidden_states.shape
        num_experts, H, M = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Compute in float32 for numerical stability, return in bfloat16
        result = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float32)

        # For each token, iterate its selected experts
        for t in range(num_tokens):
            exp_ids = selected_experts[t]  # [num_experts_per_tok], int64
            for k in range(num_experts_per_tok):
                exp = int(exp_ids[k].item())
                # hidden_state row
                hs_row = hidden_states[t].to(torch.float32)  # [hidden_size]

                # Gate: gate_out = hs_row @ expert_gate_weights[exp] -> [hidden_size, M]
                B_gate = expert_gate_weights[exp].to(torch.float32)  # [H, M]
                gate_out = torch.empty(hidden_size, M, device=device, dtype=torch.float32)
                grid_gate = (triton.cdiv(M, 128),)
                bmm_row_AxB_C[grid_gate](
                    hs_row, B_gate, gate_out,
                    H, M,
                    B_gate.stride(0), B_gate.stride(1),
                    gate_out.stride(0), gate_out.stride(1),
                    BLOCK_H=128, BLOCK_M=128,
                    num_warps=4
                )

                # Up: up_out = hs_row @ expert_up_weights[exp] -> [hidden_size, M]
                B_up = expert_up_weights[exp].to(torch.float32)  # [H, M]
                up_out = torch.empty(hidden_size, M, device=device, dtype=torch.float32)
                grid_up = (triton.cdiv(M, 128),)
                bmm_row_AxB_C[grid_up](
                    hs_row, B_up, up_out,
                    H, M,
                    B_up.stride(0), B_up.stride(1),
                    up_out.stride(0), up_out.stride(1),
                    BLOCK_H=128, BLOCK_M=128,
                    num_warps=4
                )

                # SiLU on gate_out
                silu_out = torch.empty(M, device=device, dtype=torch.float32)
                grid_silu = (triton.cdiv(M, 128),)
                silu_kernel[grid_silu](
                    gate_out, silu_out, M, BLOCK=128, num_warps=4
                )

                # Multiply by up_out elementwise
                activated = silu_out * up_out  # [M]

                # Down: expert_outputs = activated @ expert_down_weights[exp]
                D_down = expert_down_weights[exp].to(torch.float32)  # [M, hidden_size]
                expert_outputs = torch.empty(hidden_size, device=device, dtype=torch.float32)
                grid_down = (triton.cdiv(hidden_size, 128),)
                bmm_row_CxD_E[grid_down](
                    activated, D_down, expert_outputs,
                    M, hidden_size,
                    D_down.stride(0), D_down.stride(1), D_down.stride(2),
                    expert_outputs.stride(0),
                    BLOCK_M=128, BLOCK_H=128,
                    num_warps=4
                )

                # Aggregate per token using routing_weights
                if routing_weights is not None and t < routing_weights.shape[0] and exp < routing_weights[t].shape[0]:
                    weight = float(routing_weights[t, k].item())
                    result[t] += weight * expert_outputs

        # Cast back to bfloat16 for output to match original tensor dtype
        return result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
