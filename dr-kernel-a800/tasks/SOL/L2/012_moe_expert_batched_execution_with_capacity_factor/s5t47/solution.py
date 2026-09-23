import math
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_forward_triton_kernel(
    hidden_states_ptr,                # *fp32 flattened [num_tokens * hidden_size]
    selected_experts_ptr,             # *int64 flattened [num_tokens * num_experts_per_tok]
    routing_weights_ptr,              # *fp32 flattened [num_tokens * num_experts_per_tok]
    expert_gate_weights_ptr,          # *fp32 [num_experts * hidden_size * intermediate_size]
    expert_up_weights_ptr,            # *fp32 [num_experts * hidden_size * intermediate_size]
    expert_down_weights_ptr,          # *fp32 [num_experts * intermediate_size * hidden_size]
    result_ptr,                       # *fp32 flattened [num_tokens * hidden_size]
    num_tokens,                       # int32 runtime
    hidden_size,                      # int32 runtime
    num_experts_per_tok,              # int32 runtime
    intermediate_size,                # int32 runtime
):
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return

    # Iterate over selected experts for this token
    for j in range(0, num_experts_per_tok):
        # Load expert id and routing weight
        expert_id = tl.load(selected_experts_ptr + token_id * num_experts_per_tok + j).to(tl.int64)
        routing_weight = tl.load(routing_weights_ptr + token_id * num_experts_per_tok + j).to(tl.float32)

        # Load hidden state row for this token
        hs = tl.zeros((hidden_size,), dtype=tl.float32)
        for h in range(0, hidden_size):
            idx = token_id * hidden_size + h
            hs[h] = tl.load(hidden_states_ptr + idx).to(tl.float32)

        # Compute gate_out: [hidden_size, intermediate_size] = hs * gate_weights[expert_id]
        gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        gate_weight_size = hidden_size * intermediate_size
        base_gate = expert_id * gate_weight_size
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                w_idx = base_gate + h * intermediate_size + k
                w_val = tl.load(expert_gate_weights_ptr + w_idx).to(tl.float32)
                gate_out[h, k] = hs[h] * w_val

        # Compute up_out: [hidden_size, intermediate_size] = hs * up_weights[expert_id]
        up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        base_up = expert_id * gate_weight_size
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                w_idx = base_up + h * intermediate_size + k
                w_val = tl.load(expert_up_weights_ptr + w_idx).to(tl.float32)
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

        # Compute final output: [hidden_size] = prod @ down_weights[expert_id]
        down_weight_size = intermediate_size * hidden_size
        base_down = expert_id * down_weight_size
        output_row = tl.zeros((hidden_size,), dtype=tl.float32)
        for i in range(0, hidden_size):
            for k in range(0, intermediate_size):
                down_idx = base_down + k * hidden_size + i
                dw_val = tl.load(expert_down_weights_ptr + down_idx).to(tl.float32)
                output_row[i] += prod[i, k] * dw_val

        # Atomic add weighted contribution into result
        row_start = token_id * hidden_size
        for i in range(0, hidden_size):
            tl.atomic_add(result_ptr + row_start + i, output_row[i] * routing_weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure inputs are contiguous and use float32 for computation.
        # Forward does NOT use any torch tensor operations beyond allocation/casting/launch.
        hidden_states_fp32 = hidden_states.contiguous().to(torch.float32)
        selected_experts_i64 = selected_experts.contiguous().to(torch.int64)
        routing_weights_fp32 = routing_weights.contiguous().to(torch.float32)
        expert_gate_weights_fp32 = expert_gate_weights.contiguous().to(torch.float32)
        expert_up_weights_fp32 = expert_up_weights.contiguous().to(torch.float32)
        expert_down_weights_fp32 = expert_down_weights.contiguous().to(torch.float32)

        num_tokens = hidden_states_fp32.shape[0]
        hidden_size = hidden_states_fp32.shape[1]
        num_experts_per_tok = selected_experts_i64.shape[1]
        intermediate_size = expert_gate_weights_fp32.shape[2]

        # Flatten hidden states for row loads
        hidden_states_flat = hidden_states_fp32.reshape(-1)

        # Output buffer in fp32: [num_tokens, hidden_size] flattened
        result_fp32 = torch.zeros((num_tokens * hidden_size,), dtype=torch.float32, device=hidden_states_fp32.device)

        # Launch Triton kernel: 1D grid over tokens
        grid = (num_tokens,)
        _moe_forward_triton_kernel[grid](
            hidden_states_flat,
            selected_experts_i64.reshape(-1),
            routing_weights_fp32.reshape(-1),
            expert_gate_weights_fp32.reshape(-1),
            expert_up_weights_fp32.reshape(-1),
            expert_down_weights_fp32.reshape(-1),
            result_fp32,
            num_tokens,
            hidden_size,
            num_experts_per_tok,
            intermediate_size,
            num_warps=1,
            num_stages=1,
        )

        # Return result cast to bfloat16 to match original code's output dtype
        return result_fp32.reshape(num_tokens, hidden_size).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
