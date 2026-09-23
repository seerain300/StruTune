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
    Triton batched matmul: for each batch element b, compute Y[b, m] = sum_k X[b, k] * W[k, m]
    Shapes:
      X: (B, H)
      W: (H, M)
      Y: (B, M)
    Strides are in elements.
    """
    b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile along M
    pid_h = tl.program_id(2)  # tile along H (reduce dimension, loop handled here)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    h_offsets = tl.arange(0, BLOCK_H)  # loop over H in chunks

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k in range(0, H, BLOCK_H):
        k_offsets = k + h_offsets

        # Load X[b, k_offsets] -> vector of length BLOCK_H
        x_ptrs = X_ptr + b * X_stride_b + k_offsets * X_stride_h
        x = tl.load(x_ptrs, mask=k_offsets < H, other=0.0).to(tl.float32)  # (BLOCK_H,)

        # Load W[k_offsets, m_offsets] -> matrix (BLOCK_H, BLOCK_M)
        w_ptrs = W_ptr + k_offsets[:, None] * W_stride_h + m_offsets[None, :] * W_stride_m
        w = tl.load(w_ptrs, mask=(k_offsets[:, None] < H) & (m_offsets[None, :] < M), other=0.0).to(tl.float32)

        acc += tl.dot(x[None, :], w)[0, :]  # (BLOCK_M,)

    # Store acc into Y[b, m_offsets]
    y_ptrs = Y_ptr + b * Y_stride_b + m_offsets * Y_stride_m
    tl.store(y_ptrs, acc, mask=m_offsets < M)


@triton.jit
def triton_silu_mul(Z_ptr, U_ptr, Y_ptr, N, Z_stride, U_stride, Y_stride, BLOCK: tl.constexpr):
    """
    Elementwise activation: Y[i] = silu(Z[i]) * U[i]
    silu(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    z = tl.load(Z_ptr + offsets * Z_stride, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + offsets * U_stride, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-z))
    y = z * sig * u

    tl.store(Y_ptr + offsets * Y_stride, y, mask=mask)


@triton.jit
def triton_atomic_add_weighted_row(In_ptr, Weight_ptr, Out_ptr,
                                   N, In_stride, Weight_stride, Out_stride_row, BLOCK: tl.constexpr):
    """
    Atomic add rows into Out:
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
        Triton-only forward.
        Inputs:
          - hidden_states: [num_tokens, hidden_size] (bf16)
          - selected_experts: [num_tokens, num_experts_per_tok] (int64)
          - routing_weights: [num_tokens, num_experts_per_tok] (bf16)
          - expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size]
          - expert_up_weights: [num_experts, hidden_size, moe_intermediate_size]
          - expert_down_weights: [num_experts, moe_intermediate_size, hidden_size]
        Output:
          - result: [num_tokens, hidden_size] (bf16), accumulation of weighted final_out.
        """
        assert hidden_states.is_cuda, "Inputs must be on CUDA device"
        assert selected_experts.is_cuda, "selected_experts must be on CUDA device"
        assert routing_weights.is_cuda, "routing_weights must be on CUDA device"
        assert expert_gate_weights.is_cuda, "expert_gate_weights must be on CUDA device"
        assert expert_up_weights.is_cuda, "expert_up_weights must be on CUDA device"
        assert expert_down_weights.is_cuda, "expert_down_weights must be on CUDA device"

        num_tokens, hidden_size = hidden_states.shape
        num_experts, H, M = expert_gate_weights.shape
        _, selected_experts_per_tok = selected_experts.shape
        num_experts_2, _, _ = expert_down_weights.shape
        assert H == hidden_size, "hidden_size mismatch"
        assert num_experts == num_experts_2, "num_experts mismatch"
        assert routing_weights.shape == (num_tokens, selected_experts_per_tok), "routing_weights shape mismatch"

        # Prepare output tensor
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=hidden_states.device)  # keep fp32 for accumulation

        # Process each token and each selected expert. This avoids torch sorting/bincount entirely.
        # Note: We assume the original logic selects exactly one expert per token (num_experts_per_tok is typically 1 in provided inputs).
        for t in range(num_tokens):
            for j in range(selected_experts_per_tok):
                e = int(selected_experts[t, j].item())
                # Ensure in-bounds; original code assumes selected_experts are valid.
                if e < 0 or e >= num_experts:
                    continue

                # 1) gate_out = hidden_states[t] @ expert_gate_weights[e]
                gate_out = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
                # X is (1, H), W is (H, M), Y is (1, M)
                X = hidden_states[t:t+1].to(torch.float32).contiguous()
                W_gate = expert_gate_weights[e].contiguous()  # (H, M)
                Y = gate_out.view(1, M).contiguous()

                grid_gate = (1, triton.cdiv(M, 128), triton.cdiv(H, 64))
                triton_bmm[grid_gate](
                    X, W_gate, Y,
                    1, H, M,
                    X.stride(0), X.stride(1),
                    W_gate.stride(0), W_gate.stride(1),
                    Y.stride(0), Y.stride(1),
                    BLOCK_M=128, BLOCK_H=64,
                    num_warps=4, num_stages=2
                )
                gate_out = gate_out  # (M,)

                # 2) up_out = hidden_states[t] @ expert_up_weights[e]
                up_out = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
                X_up = hidden_states[t:t+1].to(torch.float32).contiguous()
                W_up = expert_up_weights[e].contiguous()  # (H, M)
                Y_up = up_out.view(1, M).contiguous()

                grid_up = (1, triton.cdiv(M, 128), triton.cdiv(H, 64))
                triton_bmm[grid_up](
                    X_up, W_up, Y_up,
                    1, H, M,
                    X_up.stride(0), X_up.stride(1),
                    W_up.stride(0), W_up.stride(1),
                    Y_up.stride(0), Y_up.stride(1),
                    BLOCK_M=128, BLOCK_H=64,
                    num_warps=4, num_stages=2
                )
                up_out = up_out  # (M,)

                # 3) activated = silu(gate_out) * up_out
                activated = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
                Z = gate_out
                U = up_out
                Y_activated = activated
                grid_act = (triton.cdiv(M, 256),)
                triton_silu_mul[grid_act](
                    Z, U, Y_activated,
                    M,
                    1, 1, 1,
                    256,
                    num_warps=4, num_stages=1
                )
                activated = activated  # (M,)

                # 4) final_out = activated @ expert_down_weights[e]
                final_out = torch.empty(hidden_size, dtype=torch.float32, device=hidden_states.device)
                X_final = activated.view(1, M).to(torch.float32).contiguous()
                W_final = expert_down_weights[e].contiguous()  # (M, hidden_size)
                Y_final = final_out.view(1, hidden_size).contiguous()

                grid_final = (1, triton.cdiv(hidden_size, 128), triton.cdiv(M, 64))
                triton_bmm[grid_final](
                    X_final, W_final, Y_final,
                    1, M, hidden_size,
                    X_final.stride(0), X_final.stride(1),
                    W_final.stride(0), W_final.stride(1),
                    Y_final.stride(0), Y_final.stride(1),
                    BLOCK_M=128, BLOCK_H=64,
                    num_warps=4, num_stages=2
                )
                final_out = final_out  # (hidden_size,)

                # 5) Accumulate: result[t] += routing_weights[t, e] * final_out
                weight = routing_weights[t, j].to(torch.float32)  # scalar
                # Use Triton atomic_add to add scalar*final_out into result[t, :]
                N = hidden_size
                In = final_out.to(torch.float32).contiguous()  # [N]
                Weight_vec = (weight * In).contiguous()        # [N], each element is weight * In[i]
                # Launch kernel that performs atomic add per i: out[t, i] += Weight_vec[i]
                out_stride_row = result.stride(1)
                grid_atomic = (triton.cdiv(N, 256),)
                triton_atomic_add_weighted_row[grid_atomic](
                    In, Weight_vec, result,
                    N, 1, 1, out_stride_row,
                    256,
                    num_warps=4, num_stages=1
                )

        # Cast back to original dtype if needed
        result = result.to(hidden_states.dtype)
        return result


def run(*args):
    return ModelNew()(*args)
