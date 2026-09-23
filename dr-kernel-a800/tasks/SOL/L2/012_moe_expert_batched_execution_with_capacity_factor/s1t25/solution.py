import torch
import triton
import triton.language as tl


# Triton kernel: batched matmul for X [B, H] @ W [H, M] -> Y [B, M]
# We implement B=1 in forward calls. We use fp32 for computation.
@triton.jit
def bmm_triton_kernel(X_ptr, W_ptr, Y_ptr,
                      B, H, M,
                      stride_xb, stride_xh,
                      stride_wh, stride_wm,
                      stride_yb, stride_ym,
                      BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    # One program handles one row (b in [0, B)) and a block of columns (m)
    b = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over H dimension in chunks of BLOCK_H
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        # Load X[b, offs_h]
        x = tl.load(X_ptr + b * stride_xb + offs_h * stride_xh, mask=offs_h < H, other=0.0)  # [BLOCK_H], fp32
        # Load W[offs_h, offs_m] as [BLOCK_H, BLOCK_M]
        w = tl.load(W_ptr + offs_h[:, None] * stride_wh + offs_m[None, :] * stride_wm,
                    mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)  # [BLOCK_H, BLOCK_M]
        # Accumulate: sum over H
        acc += tl.sum(w * x[:, None], axis=0)

    # Store to Y[b, offs_m]
    tl.store(Y_ptr + b * stride_yb + offs_m * stride_ym, acc, mask=offs_m < M)


# Triton kernel: elementwise activation Y = silu(Z) * U
# Z: [M], U: [M], Y: [M]
@triton.jit
def silu_mul_kernel(Z_ptr, U_ptr, Y_ptr, M, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < M
    z = tl.load(Z_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # silu(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    silu = z * (1.0 / (1.0 + tl.exp(-z)))
    y = silu * u
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton kernel: atomic add result[t] += weight * final_out
# result: [T, H], final_out: [H], weight: scalar
@triton.jit
def atomic_add_weight_kernel(result_ptr, final_ptr, weight, T, H, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK) + tl.program_id(1) * BLOCK
    mask = offs < H
    f = tl.load(final_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(result_ptr + t * H + offs, f * weight)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,           # [num_tokens, hidden_size], bfloat16
                selected_experts: torch.Tensor,        # [num_tokens, num_experts_per_tok], int64
                routing_weights: torch.Tensor,         # [num_tokens, num_experts_per_tok], bfloat16
                expert_gate_weights: torch.Tensor,     # [num_experts, hidden_size, moe_intermediate_size], bfloat16
                expert_up_weights: torch.Tensor,       # [num_experts, hidden_size, moe_intermediate_size], bfloat16
                expert_down_weights: torch.Tensor      # [num_experts, moe_intermediate_size, hidden_size], bfloat16
                ):
        # Ensure CUDA tensors and contiguous
        hidden_states = hidden_states.contiguous()
        selected_experts = selected_experts.contiguous()
        routing_weights = routing_weights.contiguous()
        expert_gate_weights = expert_gate_weights.contiguous()
        expert_up_weights = expert_up_weights.contiguous()
        expert_down_weights = expert_down_weights.contiguous()

        # Extract shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, H, M = expert_gate_weights.shape  # H is hidden_size, M is intermediate size
        _, _, H_out = expert_down_weights.shape       # H_out should be hidden_size

        # Output result: [num_tokens, hidden_size], bfloat16
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=hidden_states.device)

        # Loop over tokens and selected experts
        for t in range(num_tokens):
            # selected_experts[t] gives the selected experts for this token
            for j in range(selected_experts.shape[1]):
                e = int(selected_experts[t, j].item())  # index of expert

                # 1) Compute gate_out = hidden_states[t] @ expert_gate_weights[e]
                x = hidden_states[t].unsqueeze(0)          # [1, H], bfloat16
                w_gate = expert_gate_weights[e]            # [H, M], bfloat16
                # Output y_gate: [1, M], fp32 for compute
                y_gate = torch.empty((1, M), dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    x, w_gate, y_gate,
                    1, H, M,
                    x.stride(0), x.stride(1),
                    w_gate.stride(0), w_gate.stride(1),
                    y_gate.stride(0), y_gate.stride(1),
                    BLOCK_M=128, BLOCK_H=64
                )

                # 2) Compute up_out = hidden_states[t] @ expert_up_weights[e]
                w_up = expert_up_weights[e]                # [H, M], bfloat16
                y_up = torch.empty((1, M), dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    x, w_up, y_up,
                    1, H, M,
                    x.stride(0), x.stride(1),
                    w_up.stride(0), w_up.stride(1),
                    y_up.stride(0), y_up.stride(1),
                    BLOCK_M=128, BLOCK_H=64
                )

                # 3) Activation: activated = silu(gate_out) * up_out
                gate_out = y_gate.view(-1)                 # [M], fp32
                up_out = y_up.view(-1)                    # [M], fp32
                activated = torch.empty((M,), dtype=torch.float32, device=hidden_states.device)
                silu_mul_kernel[(triton.cdiv(M, 128),)](
                    gate_out, up_out, activated,
                    M, BLOCK=128
                )

                # 4) Compute final_out = activated @ expert_down_weights[e]
                w_down = expert_down_weights[e]           # [M_out, H], bfloat16, M_out == M
                M_out, H_out = w_down.shape
                # We want output [1, H_out]
                x2 = activated.unsqueeze(0)               # [1, M_out]
                y_final = torch.empty((1, H_out), dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    x2, w_down, y_final,
                    1, M_out, H_out,
                    x2.stride(0), x2.stride(1),
                    w_down.stride(0), w_down.stride(1),
                    y_final.stride(0), y_final.stride(1),
                    BLOCK_M=128, BLOCK_H=64
                )
                final_out = y_final.view(-1)              # [H_out], fp32

                # 5) Accumulate: result[t] += routing_weights[t, j] * final_out
                weight = float(routing_weights[t, j].item())
                atomic_add_weight_kernel[(1, triton.cdiv(H_out, 128),)](
                    result, final_out, weight,
                    num_tokens, hidden_size, BLOCK=128
                )

        return result


def run(*args):
    return ModelNew()(*args)
