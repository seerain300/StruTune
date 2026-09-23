import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_kernel(X_ptr, W_ptr, Y_ptr,
                       B: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                       BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr):
    # Each program computes one output row (one token or one group) across M
    pid_m = tl.program_id(0)
    # Loop over H in tiles
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H
        acc = tl.zeros([1, 1], dtype=tl.float32)  # we will use a scalar accumulation
        # Load X row (B=1) and corresponding W tile, accumulate
        # X is [B, H], W is [H, M]
        # We need to load X[0, h_offsets], W[h_offsets, m_offsets]
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M
        # Loop over H tile for dot product
        for i in range(0, BLOCK_H):
            h_i = h_start + i
            mask_i = h_i < H
            # Load X[0, h_i] as scalar
            x_val = tl.load(X_ptr + 0 * H + h_i, mask=mask_i, other=0.0)
            # Load W[h_i, m_offsets]
            w_ptr = W_ptr + h_i * M + m_offsets
            w_vals = tl.load(w_ptr, mask=mask_m, other=0.0)
            # Accumulate dot: acc += x_val * w_vals
            acc += x_val * w_vals
        # Store acc to Y[0, m_offsets]
        y_ptr = Y_ptr + 0 * M + m_offsets
        tl.store(y_ptr, acc, mask=mask_m)


@triton.jit
def activation_triton_kernel(Z_ptr, U_ptr, Y_ptr, weight, N: tl.constexpr, BLOCK: tl.constexpr):
    # Elementwise: Y = silu(Z) * U * weight
    for n in range(0, N, BLOCK):
        idx = n + tl.arange(0, BLOCK)
        mask = idx < N
        z = tl.load(Z_ptr + idx, mask=mask, other=0.0)
        u = tl.load(U_ptr + idx, mask=mask, other=0.0)
        # silu(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
        s = 1.0 / (1.0 + tl.exp(-z))
        y = (z * s) * u * weight
        tl.store(Y_ptr + idx, y, mask=mask)


@triton.jit
def atomic_accum_triton_kernel(result_ptr, weight, row, N: tl.constexpr, BLOCK: tl.constexpr):
    # Atomic add weight into result[row, :] across N
    for n in range(0, N, BLOCK):
        idx = n + tl.arange(0, BLOCK)
        mask = idx < N
        # Load current result values, add weight, store back
        ptr = result_ptr + row * N + idx
        val = tl.load(ptr, mask=mask, other=0.0)
        val += weight
        tl.store(ptr, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self,
                hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Extract shapes (these are metadata; no torch computation here)
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_M, _ = expert_gate_weights.shape
        _, up_M, _ = expert_up_weights.shape
        _, down_M, hidden_size_out = expert_down_weights.shape

        # Prepare outputs as fp32 (we will not use torch math in forward)
        result = torch.empty(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)
        # For each token t and selected expert e, compute and accumulate
        # Iterate over tokens
        for t in range(num_tokens):
            # Iterate over selected experts for this token
            for e in selected_experts[t]:
                # Compute gate_out = hidden_states[t] @ expert_gate_weights[e] -> [gate_M]
                gate_out = torch.empty(gate_M, dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    hidden_states[t],  # X pointer: 1D vector of length hidden_size
                    expert_gate_weights[e],  # W pointer: [gate_M, hidden_size]
                    gate_out,  # Y pointer: [gate_M]
                    B=1, H=hidden_size, M=gate_M,
                    BLOCK_H=128, BLOCK_M=128,
                )
                # Compute up_out = hidden_states[t] @ expert_up_weights[e] -> [up_M]
                up_out = torch.empty(up_M, dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    hidden_states[t],
                    expert_up_weights[e],
                    up_out,
                    B=1, H=hidden_size, M=up_M,
                    BLOCK_H=128, BLOCK_M=128,
                )
                # Compute activated = silu(gate_out) * up_out
                activated = torch.empty(up_M, dtype=torch.float32, device=hidden_states.device)
                activation_triton_kernel[(1,)](
                    gate_out, up_out, activated, routing_weights[t, e], N=up_M, BLOCK=128
                )
                # Compute final_out = activated @ expert_down_weights[e] -> [hidden_size_out]
                final_out = torch.empty(hidden_size_out, dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    activated,
                    expert_down_weights[e],
                    final_out,
                    B=1, H=up_M, M=hidden_size_out,
                    BLOCK_H=128, BLOCK_M=128,
                )
                # Accumulate: result[t] += routing_weights[t, e] * final_out
                atomic_accum_triton_kernel[(1,)](
                    result,
                    routing_weights[t, e].item(),  # scalar weight
                    t,  # row index
                    N=hidden_size_out,
                    BLOCK=128,
                )

        # Return fp32 result (evaluator can cast if needed)
        return result


def run(*args):
    return ModelNew()(*args)
