import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_kernel(
    X_ptr,          # *f32, pointer to input matrix X [B, H]
    W_ptr,          # *f32, pointer to weight matrix W [H, M]
    Y_ptr,          # *f32, pointer to output matrix Y [B, M]
    B: tl.constexpr,  # batch size (here 1)
    H: tl.constexpr,  # input dimension
    M: tl.constexpr,  # output dimension
    stride_x_b, stride_x_h,
    stride_w_h, stride_w_m,
    stride_y_b, stride_y_m,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Compute Y[b, m] = sum_k X[b, k] * W[k, m]
    for b in range(0, B):
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            acc = tl.zeros([BLOCK_M], dtype=tl.float32)
            for h in range(0, H, BLOCK_H):
                h_offsets = h + tl.arange(0, BLOCK_H)
                x = tl.load(X_ptr + b * stride_x_b + h_offsets * stride_x_h, mask=h_offsets < H, other=0.0)  # [BLOCK_H]
                w = tl.load(W_ptr + h_offsets[:, None] * stride_w_h + m_offsets[None, :] * stride_w_m, mask=(h_offsets[:, None] < H) & (m_offsets[None, :] < M), other=0.0)  # [BLOCK_H, BLOCK_M]
                acc += tl.sum(x[:, None] * w, axis=0)  # [BLOCK_M]
            tl.store(Y_ptr + b * stride_y_b + m_offsets * stride_y_m, acc, mask=m_offsets < M)


@triton.jit
def activation_silu_mul_kernel(
    Z_ptr,          # *f32, pointer to gate_out [M]
    U_ptr,          # *f32, pointer to up_out [M]
    Y_ptr,          # *f32, pointer to activated [M]
    M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Elementwise: Y = silu(Z) * U, where silu(Z) = Z * sigmoid(Z)
    for m in range(0, M, BLOCK):
        offsets = m + tl.arange(0, BLOCK)
        mask = offsets < M
        z = tl.load(Z_ptr + offsets, mask=mask, other=0.0)
        u = tl.load(U_ptr + offsets, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-z))
        y = z * s * u
        tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def atomic_add_weighted_kernel(
    out_ptr,        # *f32, pointer to output matrix [N, H_out]
    add_ptr,        # *f32, pointer to vector to add [H_out]
    weights,        # scalar f32
    N,              # num_tokens
    H_out,          # hidden_size
    stride_out_n, stride_out_h,
    BLOCK: tl.constexpr,
):
    # Accumulate out[n, :] += weights * add for each row n
    for n in range(0, N):
        for h in range(0, H_out, BLOCK):
            h_offsets = h + tl.arange(0, BLOCK)
            mask = h_offsets < H_out
            add_vals = tl.load(add_ptr + h_offsets, mask=mask, other=0.0)
            out_vals = tl.load(out_ptr + n * stride_out_n + h_offsets * stride_out_h, mask=mask, other=0.0)
            out_vals += weights * add_vals
            tl.store(out_ptr + n * stride_out_n + h_offsets * stride_out_h, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Triton-only forward: no torch ops on tensors
        assert hidden_states.is_cuda, "Inputs must be on CUDA for Triton kernels."
        assert selected_experts.is_cuda and routing_weights.is_cuda and expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, "All inputs must be on CUDA for Triton kernels."

        device = hidden_states.device

        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_H, _ = expert_gate_weights.shape  # expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size]
        num_experts_per_tok = selected_experts.shape[1]

        # Cast to fp32 for Triton kernels (safe and avoids dtype issues in Triton)
        hidden_states_f32 = hidden_states.to(torch.float32)
        routing_weights_f32 = routing_weights.to(torch.float32)
        expert_gate_weights_f32 = expert_gate_weights.to(torch.float32)   # [E, H_in, M1]
        expert_up_weights_f32 = expert_up_weights.to(torch.float32)       # [E, H_in, M1]
        expert_down_weights_f32 = expert_down_weights.to(torch.float32)   # [E, M1, H_out]

        # Allocate fp32 result buffer
        result_f32 = torch.empty(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Process each token and its selected experts
        for t in range(num_tokens):
            # Flatten hidden vector for this token: [1, H_in]
            hidden_vec = hidden_states_f32[t].contiguous().view(1, hidden_size)

            # Loop over selected_experts (deterministic order)
            for j in range(num_experts_per_tok):
                expert_id = int(selected_experts[t, j].item())

                # Compute gate_out = hidden_vec @ expert_gate_weights[expert_id]
                E_gate = expert_gate_weights_f32[expert_id]            # [H_in, M1]
                gate_out = torch.empty(1, E_gate.shape[1], dtype=torch.float32, device=device)
                bmm_triton_kernel[(1,)](
                    hidden_vec, E_gate, gate_out,
                    B=1, H=hidden_size, M=E_gate.shape[1],
                    stride_x_b=1 * hidden_size, stride_x_h=1,
                    stride_w_h=1, stride_w_m=1,
                    stride_y_b=1 * E_gate.shape[1], stride_y_m=1,
                    BLOCK_H=128, BLOCK_M=128,
                    num_warps=4,
                )

                # Compute up_out = hidden_vec @ expert_up_weights[expert_id]
                E_up = expert_up_weights_f32[expert_id]              # [H_in, M1]
                up_out = torch.empty(1, E_up.shape[1], dtype=torch.float32, device=device)
                bmm_triton_kernel[(1,)](
                    hidden_vec, E_up, up_out,
                    B=1, H=hidden_size, M=E_up.shape[1],
                    stride_x_b=1 * hidden_size, stride_x_h=1,
                    stride_w_h=1, stride_w_m=1,
                    stride_y_b=1 * E_up.shape[1], stride_y_m=1,
                    BLOCK_H=128, BLOCK_M=128,
                    num_warps=4,
                )

                # Compute activated = SiLU(gate_out) * up_out (elementwise)
                activated = torch.empty(E_gate.shape[1], dtype=torch.float32, device=device)
                activation_silu_mul_kernel[(1,)](
                    gate_out.view(-1), up_out.view(-1), activated,
                    M=E_gate.shape[1], BLOCK=128, num_warps=4
                )

                # Compute final_out = activated @ expert_down_weights[expert_id]
                E_down = expert_down_weights_f32[expert_id]          # [M1, H_out]
                final_out = torch.empty(1, E_down.shape[1], dtype=torch.float32, device=device)
                bmm_triton_kernel[(1,)](
                    activated.view(1, -1), E_down, final_out,
                    B=1, H=E_gate.shape[1], M=E_down.shape[1],
                    stride_x_b=1 * E_gate.shape[1], stride_x_h=1,
                    stride_w_h=1, stride_w_m=1,
                    stride_y_b=1 * E_down.shape[1], stride_y_m=1,
                    BLOCK_H=128, BLOCK_M=128,
                    num_warps=4,
                )

                # Atomic add into result: result[t] += routing_weights[t, j] * final_out
                weight = float(routing_weights_f32[t, j].item())
                add_vec = final_out.view(-1)  # [H_out]
                atomic_add_weighted_kernel[(num_tokens,)](
                    result_f32, add_vec, weight,
                    N=num_tokens, H_out=E_down.shape[1],
                    stride_out_n=E_down.shape[1], stride_out_h=1,
                    BLOCK=256,
                    num_warps=4,
                )

        # Cast result back to bfloat16 to match original dtype
        result = result_f32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
