import math
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_forward_triton_kernel(
    hidden_states_ptr,         # *fp32, flattened as [num_tokens * hidden_size]
    selected_experts_ptr,      # *int32, flattened as [num_tokens * num_experts_per_tok]
    routing_weights_ptr,       # *fp32, flattened as [num_tokens * num_experts_per_tok]
    expert_gate_weights_ptr,   # *fp32, flattened as [num_experts * hidden_size * intermediate_size]
    expert_up_weights_ptr,     # *fp32, flattened as [num_experts * hidden_size * intermediate_size]
    expert_down_weights_ptr,   # *fp32, flattened as [num_experts * intermediate_size * hidden_size]
    result_ptr,                # *fp32, flattened as [num_tokens * hidden_size]
    num_tokens: tl.int32,
    hidden_size: tl.int32,
    num_experts_per_tok: tl.int32,
    intermediate_size: tl.int32,
):
    pid = tl.program_id(axis=0)  # token id in [0, num_tokens)

    # Base offsets for this token
    row_start_hs = pid * hidden_size
    row_start_out = pid * hidden_size

    # Iterate over selected experts for this token
    for j in range(0, num_experts_per_tok):
        # Load selected expert id (int32) and routing weight (fp32)
        expert_id = tl.load(selected_experts_ptr + pid * num_experts_per_tok + j).to(tl.int32)
        routing_weight = tl.load(routing_weights_ptr + pid * num_experts_per_tok + j).to(tl.float32)

        # Load hidden state row for this token: [hidden_size]
        hs = tl.zeros((hidden_size,), dtype=tl.float32)
        for i in range(0, hidden_size):
            hs[i] = tl.load(hidden_states_ptr + row_start_hs + i)

        # Compute gate_out: [hidden_size, intermediate_size]
        gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        gate_weight_size = hidden_size * intermediate_size
        base_expert_gate = expert_id * gate_weight_size
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                w_idx = base_expert_gate + h * intermediate_size + k
                w_val = tl.load(expert_gate_weights_ptr + w_idx).to(tl.float32)
                gate_out[h, k] = hs[h] * w_val

        # Compute up_out: [hidden_size, intermediate_size]
        up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        base_expert_up = expert_id * gate_weight_size
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                w_idx = base_expert_up + h * intermediate_size + k
                w_val = tl.load(expert_up_weights_ptr + w_idx).to(tl.float32)
                up_out[h, k] = hs[h] * w_val

        # SiLU(gate_out): gate_out * sigmoid(gate_out)
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
                dw_val = tl.load(expert_down_weights_ptr + down_idx).to(tl.float32)
                output_row[i] += prod[i, k] * dw_val

        # Atomic add weighted contribution into result
        for i in range(0, hidden_size):
            tl.atomic_add(result_ptr + row_start_out + i, output_row[i] * routing_weight)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Inputs: same as original get_inputs
        hidden_states, selected_experts, routing_weights, expert_gate_weights, expert_up_weights, expert_down_weights = args

        # Only allocations, contiguity, casts, and Triton kernel launch are allowed in forward.
        hidden_states_fp32 = hidden_states.contiguous().to(torch.float32)
        selected_experts_i32 = selected_experts.contiguous().to(torch.int32)  # Triton prefers int32
        routing_weights_fp32 = routing_weights.contiguous().to(torch.float32)
        expert_gate_weights_fp32 = expert_gate_weights.contiguous().to(torch.float32)
        expert_up_weights_fp32 = expert_up_weights.contiguous().to(torch.float32)
        expert_down_weights_fp32 = expert_down_weights.contiguous().to(torch.float32)

        num_tokens = hidden_states_fp32.shape[0]
        hidden_size = hidden_states_fp32.shape[1]
        num_experts_per_tok = selected_experts_i32.shape[1]
        num_experts = expert_gate_weights_fp32.shape[0]
        intermediate_size = expert_gate_weights_fp32.shape[2]

        # Flatten hidden states for row loads
        hidden_states_flat = hidden_states_fp32.reshape(-1)

        # Output buffer in fp32: [num_tokens, hidden_size] flattened
        result_fp32 = torch.zeros((num_tokens * hidden_size,), dtype=torch.float32, device=hidden_states_fp32.device)

        # Launch Triton kernel: 1D grid over tokens
        grid = (num_tokens,)
        _moe_forward_triton_kernel[grid](
            hidden_states_flat,
            selected_experts_i32.reshape(-1),
            routing_weights_fp32.reshape(-1),
            expert_gate_weights_fp32.reshape(-1),
            expert_up_weights_fp32.reshape(-1),
            expert_down_weights_fp32.reshape(-1),
            result_fp32,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            num_warps=1,
            num_stages=1,
        )

        # Return result cast to bfloat16 to match original dtype
        return result_fp32.reshape(num_tokens, hidden_size).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
