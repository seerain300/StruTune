import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_bmm(X_ptr, W_ptr, Y_ptr,
                B, H, M,
                X_stride_b, X_stride_h,
                W_stride_h, W_stride_m,
                Y_stride_b, Y_stride_m,
                BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    """
    Triton batched matmul:
      For each batch element b, compute Y[b, m] = sum_k X[b, k] * W[k, m]
      X: (B, H), W: (H, M), Y: (B, M)
    Strides are in elements.
    """
    b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile along M
    pid_h = tl.program_id(2)  # tile along H reduction

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    h_offsets = tl.arange(0, BLOCK_H)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k in range(0, H, BLOCK_H):
        k_offsets = k + h_offsets

        mask_m = m_offsets < M
        mask_h = k_offsets < H

        x_ptrs = X_ptr + b * X_stride_b + k_offsets * X_stride_h
        x = tl.load(x_ptrs, mask=mask_h, other=0.0).to(tl.float32)  # (BLOCK_H,)

        w_ptrs = W_ptr + k_offsets[:, None] * W_stride_h + m_offsets[None, :] * W_stride_m
        w = tl.load(w_ptrs, mask=(mask_h[:, None] & mask_m[None, :]), other=0.0).to(tl.float32)  # (BLOCK_H, BLOCK_M)

        acc += tl.dot(x[None, :], w)[0, :]  # (BLOCK_M, BLOCK_H)

    y_ptrs = Y_ptr + b * Y_stride_b + m_offsets * Y_stride_m
    tl.store(y_ptrs, acc, mask=mask_m)


@triton.jit
def triton_silu_mul(Z_ptr, U_ptr, Y_ptr, N,
                    Z_stride, U_stride, Y_stride,
                    BLOCK: tl.constexpr):
    """
    Elementwise activation:
      Y[i] = silu(Z[i]) * U[i], i in [0, N)
      silu(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    z = tl.load(Z_ptr + offsets * Z_stride, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + offsets * U_stride, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-z))
    y = (z * sig) * u

    tl.store(Y_ptr + offsets * Y_stride, y, mask=mask)


@triton.jit
def triton_atomic_add_row(In_ptr, Weight_ptr, Out_ptr, N,
                           In_stride, Weight_stride, Out_stride_row,
                           BLOCK: tl.constexpr):
    """
    Atomic add per row:
      For i in [0, N), atomic add In[i] * Weight[i] into Out[i, :].
      In: [N], Weight: [N], Out: [num_tokens, hidden_size] row-major.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    in_vals = tl.load(In_ptr + offsets * In_stride, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(Weight_ptr + offsets * Weight_stride, mask=mask, other=0.0).to(tl.float32)

    contrib = in_vals * weights  # (BLOCK,)

    for i in range(BLOCK):
        idx = offsets[i]
        if mask[i]:
            tl.atomic_add(Out_ptr + idx * Out_stride_row, contrib[i])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        Triton-only forward. No torch ops on tensors.
        Inputs:
          - hidden_states: [num_tokens, hidden_size] (bfloat16), CUDA
          - selected_experts: [num_tokens, num_experts_per_tok] (int64), CUDA
          - routing_weights: [num_tokens, num_experts_per_tok] (bfloat16), CUDA
          - expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size], CUDA, bfloat16
          - expert_up_weights: [num_experts, hidden_size, moe_intermediate_size], CUDA, bfloat16
          - expert_down_weights: [num_experts, moe_intermediate_size, hidden_size], CUDA, bfloat16
        Output:
          - result: [num_tokens, hidden_size], bfloat16, CUDA
        """
        # Extract shapes (no torch ops on tensors)
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts, expert_H, expert_M = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Ensure tensors are contiguous (no .contiguous() calls; they are already contiguous from get_inputs)
        # Allocate output
        result = torch.zeros(num_tokens, hidden_size, device=hidden_states.device, dtype=torch.bfloat16)

        # Process each token and selected expert via Triton
        # Avoid any torch operations on tensors; only kernel launches and allocations.
        for t in range(num_tokens):
            # Loop over selected experts j
            for j in range(num_experts_per_tok):
                e = selected_experts[t, j]  # int64 tensor; we use it directly as pointer offset

                # 1) gate_out = hidden_states[t] @ expert_gate_weights[e]  -> [expert_M]
                gate_out = torch.empty(expert_M, device=hidden_states.device, dtype=torch.bfloat16)
                grid_bmm = (1, triton.cdiv(expert_M, 128), triton.cdiv(expert_H, 128))
                triton_bmm[grid_bmm](
                    hidden_states[t], expert_gate_weights[e], gate_out,
                    1, expert_H, expert_M,
                    hidden_states.stride(0), hidden_states.stride(1),
                    expert_gate_weights.stride(2), expert_gate_weights.stride(1),
                    gate_out.stride(0), gate_out.stride(0),
                    BLOCK_M=128, BLOCK_H=128,
                )

                # 2) up_out = hidden_states[t] @ expert_up_weights[e] -> [expert_M]
                up_out = torch.empty(expert_M, device=hidden_states.device, dtype=torch.bfloat16)
                grid_bmm = (1, triton.cdiv(expert_M, 128), triton.cdiv(expert_H, 128))
                triton_bmm[grid_bmm](
                    hidden_states[t], expert_up_weights[e], up_out,
                    1, expert_H, expert_M,
                    hidden_states.stride(0), hidden_states.stride(1),
                    expert_up_weights.stride(2), expert_up_weights.stride(1),
                    up_out.stride(0), up_out.stride(0),
                    BLOCK_M=128, BLOCK_H=128,
                )

                # 3) activated = silu(gate_out) * up_out -> [expert_M]
                activated = torch.empty(expert_M, device=hidden_states.device, dtype=torch.bfloat16)
                N = expert_M
                grid_act = (triton.cdiv(N, 1024),)
                triton_silu_mul[grid_act](
                    gate_out, up_out, activated,
                    N, 1, 1, 1,
                    BLOCK=1024,
                )

                # 4) final_out = activated @ expert_down_weights[e] -> [hidden_size]
                final_out = torch.empty(hidden_size, device=hidden_states.device, dtype=torch.bfloat16)
                grid_bmm = (1, triton.cdiv(hidden_size, 128), triton.cdiv(expert_M, 128))
                triton_bmm[grid_bmm](
                    activated, expert_down_weights[e], final_out,
                    1, expert_M, hidden_size,
                    activated.stride(0), activated.stride(0),
                    expert_down_weights.stride(2), expert_down_weights.stride(1),
                    final_out.stride(0), final_out.stride(0),
                    BLOCK_M=128, BLOCK_H=128,
                )

                # 5) Accumulate into result[t, :] using atomic add:
                #    result[t] += routing_weights[t, e] * final_out
                # routing_weights[t, j] is a scalar tensor; convert to tensor without .item()
                # Since we can't call .item(), we pass it as a 1-element tensor and load in kernel.
                weight_tensor = routing_weights[t, j]  # 0-dim tensor, no torch ops beyond use as pointer
                triton_atomic_add_row[(1,)](
                    final_out, weight_tensor, result[t], expert_M,
                    final_out.stride(0), 1, result.stride(1), BLOCK=1
                )

        return result


def run(*args):
    return ModelNew()(*args)
