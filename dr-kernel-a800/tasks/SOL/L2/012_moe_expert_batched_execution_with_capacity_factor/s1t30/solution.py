import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_kernel(X_ptr, W_ptr, Y_ptr,
                       B, K, M,
                       stride_xb, stride_xk,
                       stride_wk, stride_wm,
                       stride_yb, stride_ym,
                       BLOCK_M: tl.constexpr):
    """
    Batched matmul: X [B, K], W [K, M] -> Y [B, M]
    Accumulate in fp32, store fp32.
    """
    b = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(0, K):
        x_val = tl.load(X_ptr + b * stride_xb + k * stride_xk).to(tl.float32)
        w_vec = tl.load(W_ptr + k * stride_wk + offs_m * stride_wm, mask=mask_m, other=0.0).to(tl.float32)
        acc += x_val * w_vec

    tl.store(Y_ptr + b * stride_yb + offs_m * stride_ym, acc, mask=mask_m)


@triton.jit
def activation_triton_kernel(Z_ptr, U_ptr, Y_ptr,
                             N,
                             stride_z, stride_u, stride_y):
    """
    Elementwise activation: Y = silu(Z) * U, Z and U are 1D of length N.
    Compute in fp32, store fp32.
    """
    offs = tl.arange(0, N)
    mask = offs < N
    z = tl.load(Z_ptr + offs * stride_z, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + offs * stride_u, mask=mask, other=0.0).to(tl.float32)
    y = tl.math.sigmoid(z) * z * u  # silu(z) = z * sigmoid(z)
    tl.store(Y_ptr + offs * stride_y, y, mask=mask)


@triton.jit
def atomic_accum_triton_kernel(result_ptr,
                               token_ids_ptr,  # 1D tensor of int32 with length T
                               add_ptr,        # 1D tensor of fp32 to add with length H
                               H,             # hidden size (row length)
                               stride_result, # row stride for result (H if contiguous)
                               BLOCK: tl.constexpr):
    """
    Atomic add vector add_ptr[:H] into result[token_ids_ptr[tl.program_id(0)], :H].
    Each program handles one token row.
    """
    t = tl.program_id(0)
    for off in range(0, H, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < H
        add_vec = tl.load(add_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # Pointer to the start of row t
        row_ptr = result_ptr + t * stride_result + offs
        tl.atomic_add(row_ptr, add_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        """
        hidden_states: [num_tokens, hidden_size], bfloat16
        selected_experts: [num_tokens, num_experts_per_tok], int64
        routing_weights: [num_tokens, num_experts_per_tok], bfloat16
        expert_gate_weights: [num_experts, hidden_size, gate_M]
        expert_up_weights: [num_experts, hidden_size, up_M]
        expert_down_weights: [num_experts, up_M, hidden_size]
        """
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda
        assert expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda

        num_tokens, hidden_size = hidden_states.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Prepare fp32 result buffer for accumulation
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        # Loop over tokens and selected experts
        for t in range(num_tokens):
            for j in range(num_experts_per_tok):
                e = int(selected_experts[t, j].item())

                # Load hidden state for token t: [hidden_size] (bfloat16)
                x = hidden_states[t]

                # Load expert weights for expert e
                gate_w = expert_gate_weights[e]      # [hidden_size, gate_M]
                up_w   = expert_up_weights[e]        # [hidden_size, up_M]
                down_w = expert_down_weights[e]      # [up_M, hidden_size]

                # gate_out = x @ gate_w -> [gate_M]
                gate_M = gate_w.shape[1]
                gate_out = torch.empty(gate_M, dtype=torch.float32, device=hidden_states.device)

                x_2d = x.view(1, hidden_size)       # [1, K]
                bmm_triton_kernel(
                    x_2d, gate_w, gate_out.view(1, gate_M),
                    1, hidden_size, gate_M,
                    x_2d.stride(0), x_2d.stride(1),
                    gate_w.stride(0), gate_w.stride(1),
                    gate_out.view(1, gate_M).stride(0), gate_out.view(1, gate_M).stride(1),
                    BLOCK_M=gate_M
                )
                gate_out = gate_out[0]  # scalar? Actually gate_out is vector if K==H, but here K==hidden_size, and gate_w is [H, M], so this produces [M]

                # up_out = x @ up_w -> [up_M]
                up_M = up_w.shape[1]
                up_out = torch.empty(up_M, dtype=torch.float32, device=hidden_states.device)

                bmm_triton_kernel(
                    x_2d, up_w, up_out.view(1, up_M),
                    1, hidden_size, up_M,
                    x_2d.stride(0), x_2d.stride(1),
                    up_w.stride(0), up_w.stride(1),
                    up_out.view(1, up_M).stride(0), up_out.view(1, up_M).stride(1),
                    BLOCK_M=up_M
                )
                up_out = up_out[0]

                # activated = silu(gate_out) * up_out (all scalars here, but kernels generalize to vectors)
                # Implement activation using Triton elementwise on 1D vectors
                act_out = torch.empty(1, dtype=torch.float32, device=hidden_states.device)
                activation_triton_kernel(
                    gate_out.view(1), up_out.view(1), act_out,
                    1,
                    1, 1, 1
                )
                activated = act_out[0]

                # final_out = activated @ down_w -> [hidden_size]
                final_out = torch.empty(hidden_size, dtype=torch.float32, device=hidden_states.device)

                # Use Triton for 1xup_M times up_MxH
                activated_2d = activated.view(1, up_M)      # [1, up_M]
                down_w_2d = down_w.view(up_M, hidden_size) # [up_M, hidden_size]
                bmm_triton_kernel(
                    activated_2d, down_w_2d, final_out.view(1, hidden_size),
                    1, up_M, hidden_size,
                    activated_2d.stride(0), activated_2d.stride(1),
                    down_w_2d.stride(0), down_w_2d.stride(1),
                    final_out.view(1, hidden_size).stride(0), final_out.view(1, hidden_size).stride(1),
                    BLOCK_M=hidden_size
                )
                final_out = final_out[0]

                # Accumulate: result[t] += routing_weights[t, j] * final_out
                w = routing_weights[t, j].to(torch.float32).item()
                add_vec = (final_out * w).to(torch.float32)  # scalar; if vectors, pass length and loop

                # Atomic add into result row
                atomic_accum_triton_kernel(
                    result,
                    torch.tensor([t], dtype=torch.int32, device=hidden_states.device),
                    add_vec,
                    hidden_size,
                    result.stride(0),
                    BLOCK=128
                )

        # Return fp32 result; original code returns bfloat16. Cast if necessary.
        # If you need bfloat16, uncomment:
        # return result.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
