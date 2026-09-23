import torch
import triton
import triton.language as tl


@triton.jit
def bmm_vec_kernel(
    X_ptr,          # *f32, [1, H]
    W_ptr,          # *f32, [H, M]
    Y_ptr,          # *f32, [1, M]
    H,              # int32, hidden_size
    M,              # int32, intermediate_size
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Output is a single row vector of length M
    y = tl.zeros([M], dtype=tl.float32)
    # Iterate over H in tiles
    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H
        # Load X[h_offsets] as a vector
        x_vals = tl.load(X_ptr + h_offsets, mask=mask_h, other=0.0)  # [BLOCK_H]
        # Accumulate over this tile
        # For each h in the tile, y += x * W[h, :]
        for i in range(BLOCK_H):
            h_idx = h0 + i
            m_offsets = tl.arange(0, M)
            w_row = tl.load(W_ptr + h_idx * M + m_offsets, mask=h_idx < H, other=0.0)  # [M]
            y += x_vals[i] * w_row
    # Store result
    tl.store(Y_ptr + m_offsets, y, mask=m_offsets < M)


@triton.jit
def silu_mul_vec_kernel(
    Z_ptr,          # *f32, [M]
    U_ptr,          # *f32, [M]
    Y_ptr,          # *f32, [M]
    M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Elementwise: Y = SiLU(Z) * U, SiLU(Z) = Z * sigmoid(Z)
    for m in range(0, M, BLOCK):
        offsets = m + tl.arange(0, BLOCK)
        mask = offsets < M
        z = tl.load(Z_ptr + offsets, mask=mask, other=0.0)
        u = tl.load(U_ptr + offsets, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-z))
        y = z * s * u
        tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def bmm_vec_down_kernel(
    A_ptr,          # *f32, [1, M]
    Wd_ptr,         # *f32, [M, H_out]
    Y_ptr,          # *f32, [1, H_out]
    M,              # int32, intermediate_size
    H_out,          # int32, hidden_size
    BLOCK_M: tl.constexpr,
):
    # Compute Y = A @ Wd, where A is [1, M]
    y = tl.zeros([H_out], dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M
        a_vals = tl.load(A_ptr + m_offsets, mask=mask_m, other=0.0)  # [BLOCK_M]
        for i in range(BLOCK_M):
            m_idx = m0 + i
            mask_i = m_idx < M
            a = a_vals[i]
            h_offsets = tl.arange(0, H_out)
            wd_row = tl.load(Wd_ptr + m_idx * H_out + h_offsets, mask=mask_i, other=0.0)  # [H_out]
            y += a * wd_row
    tl.store(Y_ptr + h_offsets, y, mask=h_offsets < H_out)


@triton.jit
def atomic_add_weighted_vec_kernel(
    result_ptr,     # *f32, [num_tokens, hidden_size]
    add_ptr,        # *f32, [hidden_size]
    weights,        # f32, scalar routing weight
    N,              # num_tokens
    H,              # hidden_size
    BLOCK_H: tl.constexpr,
):
    # Accumulate row-wise: result[n, :] += weights * add
    for n in range(0, N):
        for h0 in range(0, H, BLOCK_H):
            h_offsets = h0 + tl.arange(0, BLOCK_H)
            mask = h_offsets < H
            add_vals = tl.load(add_ptr + h_offsets, mask=mask, other=0.0)
            out_vals = tl.load(result_ptr + n * H + h_offsets, mask=mask, other=0.0)
            out_vals += weights * add_vals
            tl.store(result_ptr + n * H + h_offsets, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Triton-only forward: no torch ops on tensors
        device = hidden_states.device
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_h, intermediate = expert_gate_weights.shape
        # We expect gate_h == hidden_size
        assert gate_h == hidden_size, "expert_gate_weights hidden dimension must match hidden_states"

        # Prepare fp32 copies for computation (Triton kernels use fp32)
        hidden_states_f32 = hidden_states.to(torch.float32)  # [num_tokens, hidden_size]
        routing_weights_f32 = routing_weights.to(torch.float32)  # [num_tokens, num_experts_per_tok]
        # Allocate fp32 result buffer
        result_f32 = torch.empty(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Ensure selected_experts is int32 for Triton grid
        selected_experts_i32 = selected_experts.to(torch.int32)

        # Process each token and its selected experts
        num_experts_per_tok = selected_experts.shape[1]
        for t in range(num_tokens):
            # Loop over selected experts per token
            for j in range(num_experts_per_tok):
                expert_id = int(selected_experts_i32[t, j].item())  # scalar int
                # Extract weights for this expert
                gate_w = expert_gate_weights[expert_id].to(torch.float32)  # [hidden_size, intermediate]
                up_w = expert_up_weights[expert_id].to(torch.float32)      # [hidden_size, intermediate]
                down_w = expert_down_weights[expert_id].to(torch.float32)  # [intermediate, hidden_size]

                # Hidden vector for this token
                hidden_vec = hidden_states_f32[t]  # [hidden_size], fp32

                # Compute gate_out = hidden @ gate_w -> [intermediate]
                gate_out = torch.empty(intermediate, dtype=torch.float32, device=device)
                grid = (intermediate,)  # output length
                bmm_vec_kernel[grid](hidden_vec, gate_w, gate_out, hidden_size, intermediate, BLOCK_H=128, num_warps=4)

                # Compute up_out = hidden @ up_w -> [intermediate]
                up_out = torch.empty(intermediate, dtype=torch.float32, device=device)
                grid_up = (intermediate,)
                bmm_vec_kernel[grid_up](hidden_vec, up_w, up_out, hidden_size, intermediate, BLOCK_H=128, num_warps=4)

                # Elementwise activation: activated = SiLU(gate_out) * up_out
                activated = torch.empty(intermediate, dtype=torch.float32, device=device)
                silu_mul_vec_kernel[(intermediate,)](gate_out, up_out, activated, M=intermediate, BLOCK=128, num_warps=4)

                # Compute final_out = activated @ down_w -> [hidden_size]
                final_out = torch.empty(hidden_size, dtype=torch.float32, device=device)
                bmm_vec_down_kernel[(hidden_size,)](activated, down_w, final_out, M=intermediate, H_out=hidden_size, BLOCK_M=128, num_warps=4)

                # Weight from routing
                weight = float(routing_weights_f32[t, j].item())

                # Atomic add into result[t, :] += weight * final_out
                atomic_add_weighted_vec_kernel[(num_tokens,)](
                    result_f32, final_out, weight, N=num_tokens, H=hidden_size, BLOCK_H=128, num_warps=4
                )

        # Cast result back to bfloat16 to match input dtype expectation
        result_bf16 = result_f32.to(torch.bfloat16)
        return result_bf16


def run(*args):
    return ModelNew()(*args)
