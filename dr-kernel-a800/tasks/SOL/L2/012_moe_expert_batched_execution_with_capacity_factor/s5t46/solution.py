import math
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_forward_triton_kernel(
    hidden_states_ptr,               # *fp32 flattened [num_tokens * hidden_size]
    selected_experts_ptr,            # *int32 flattened [num_tokens * num_experts_per_tok]
    routing_weights_ptr,             # *fp32 flattened [num_tokens * num_experts_per_tok]
    expert_gate_weights_ptr,         # *fp32 flattened [num_experts * hidden_size * intermediate_size]
    expert_up_weights_ptr,           # *fp32 flattened [num_experts * hidden_size * intermediate_size]
    expert_down_weights_ptr,         # *fp32 flattened [num_experts * intermediate_size * hidden_size]
    result_ptr,                      # *fp32 flattened [num_tokens * hidden_size]
    num_tokens: tl.constexpr,        # int
    hidden_size: tl.constexpr,       # int
    num_experts_per_tok: tl.constexpr,  # int
    intermediate_size: tl.constexpr,     # int
):
    pid = tl.program_id(0)  # token id
    if pid >= num_tokens:
        return

    # Base row offset in flattened hidden_states
    row_start = pid * hidden_size

    # Loop over each selected expert for this token
    for j in range(0, num_experts_per_tok):
        # Compute flattened index of (pid, j)
        idx = pid * num_experts_per_tok + j

        # Load expert id (int32) and routing weight (fp32)
        expert_id = tl.load(selected_experts_ptr + idx)
        routing_weight = tl.load(routing_weights_ptr + idx)  # fp32

        # Load hidden state for this token
        hs = tl.zeros((hidden_size,), dtype=tl.float32)
        for i in range(0, hidden_size):
            hs[i] = tl.load(hidden_states_ptr + row_start + i)

        # gate_out: [hidden_size, intermediate_size]
        gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        gate_weight_size = hidden_size * intermediate_size
        base_expert_gate = expert_id * gate_weight_size
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                w_idx = base_expert_gate + h * intermediate_size + k
                w_val = tl.load(expert_gate_weights_ptr + w_idx)  # fp32
                gate_out[h, k] = hs[h] * w_val

        # up_out: [hidden_size, intermediate_size]
        up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        base_expert_up = expert_id * (hidden_size * intermediate_size)
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                w_idx = base_expert_up + h * intermediate_size + k
                w_val = tl.load(expert_up_weights_ptr + w_idx)  # fp32
                up_out[h, k] = hs[h] * w_val

        # SiLU(gate_out) = gate_out * sigmoid(gate_out)
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                z = gate_out[h, k]
                sigmoid_z = 1.0 / (1.0 + tl.exp(-z))
                gate_out[h, k] = z * sigmoid_z

        # Multiply gate_out with up_out
        prod = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                prod[h, k] = gate_out[h, k] * up_out[h, k]

        # Compute final output: [hidden_size]
        down_weight_size = intermediate_size * hidden_size
        base_expert_down = expert_id * down_weight_size
        output_row = tl.zeros((hidden_size,), dtype=tl.float32)
        for i in range(0, hidden_size):
            for k in range(0, intermediate_size):
                down_idx = base_expert_down + k * hidden_size + i
                dw_val = tl.load(expert_down_weights_ptr + down_idx)  # fp32
                output_row[i] += prod[i, k] * dw_val

        # Atomic add weighted contribution into result
        row_start_out = pid * hidden_size
        for i in range(0, hidden_size):
            tl.atomic_add(result_ptr + row_start_out + i, output_row[i] * routing_weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure contiguity and dtype
        hidden_states_fp32 = hidden_states.contiguous().to(torch.float32)
        selected_experts_i32 = selected_experts.contiguous().to(torch.int32)  # cast to int32 for kernel
        routing_weights_fp32 = routing_weights.contiguous().to(torch.float32)

        # Flatten pointers for Triton
        hidden_flat = hidden_states_fp32.reshape(-1)
        selected_flat = selected_experts_i32.reshape(-1)
        routing_flat = routing_weights_fp32.reshape(-1)

        # Prepare expert weights as fp32 contiguous
        expert_gate_fp32 = expert_gate_weights.contiguous().to(torch.float32)
        expert_up_fp32 = expert_up_weights.contiguous().to(torch.float32)
        expert_down_fp32 = expert_down_weights.contiguous().to(torch.float32)

        num_tokens, hidden_size = hidden_states_fp32.shape
        num_experts_per_tok = selected_experts_i32.shape[1]
        num_experts, gate_h, intermediate_size = expert_gate_fp32.shape
        assert gate_h == hidden_size, "hidden_size mismatch in gate weights"

        # Output buffer in fp32: [num_tokens, hidden_size]
        result_fp32 = torch.zeros((num_tokens * hidden_size,), dtype=torch.float32, device=hidden_states_fp32.device)

        # Launch Triton kernel: 1D grid over tokens
        grid = (num_tokens,)
        _moe_forward_triton_kernel[grid](
            hidden_flat,
            selected_flat,
            routing_flat,
            expert_gate_fp32.reshape(-1),
            expert_up_fp32.reshape(-1),
            expert_down_fp32.reshape(-1),
            result_fp32,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            num_warps=1,
            num_stages=1,
        )

        # Return result cast to bfloat16 to match original code
        return result_fp32.reshape(num_tokens, hidden_size).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
