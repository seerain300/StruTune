import math
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_forward_triton_kernel(
    hidden_states_ptr,               # *fp32 flattened [num_tokens * hidden_size]
    selected_experts_ptr,            # *int64 flattened [num_tokens * num_experts_per_tok]
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
    pid = tl.program_id(0)  # one program per token
    if pid >= num_tokens:
        return

    # For this token, iterate over its selected experts and compute weighted outputs
    for j in range(0, num_experts_per_tok):
        # Load selected expert id and routing weight
        # selected_experts is flattened: idx = pid * num_experts_per_tok + j
        expert_id = tl.load(selected_experts_ptr + pid * num_experts_per_tok + j).to(tl.int32)
        routing_weight = tl.load(routing_weights_ptr + pid * num_experts_per_tok + j).to(tl.float32)

        # Load hidden state row for token pid
        # hidden_states_ptr is flattened: idx = pid * hidden_size + h
        hs = tl.zeros((hidden_size,), dtype=tl.float32)
        for h in range(0, hidden_size):
            hs[h] = tl.load(hidden_states_ptr + pid * hidden_size + h).to(tl.float32)

        # Compute gate_out: [hidden_size, intermediate_size] = hs @ expert_gate[expert_id]
        gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        base_gate = expert_id * hidden_size * intermediate_size
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                w_idx = base_gate + h * intermediate_size + k
                w_val = tl.load(expert_gate_weights_ptr + w_idx).to(tl.float32)
                gate_out[h, k] = hs[h] * w_val

        # Compute up_out: [hidden_size, intermediate_size] = hs @ expert_up[expert_id]
        up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        base_up = expert_id * hidden_size * intermediate_size
        for h in range(0, hidden_size):
            for k in range(0, intermediate_size):
                w_idx = base_up + h * intermediate_size + k
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

        # Compute final output: [hidden_size] = prod @ expert_down[expert_id]
        down_weight_size = intermediate_size * hidden_size
        base_down = expert_id * down_weight_size
        output_row = tl.zeros((hidden_size,), dtype=tl.float32)
        for i in range(0, hidden_size):
            for k in range(0, intermediate_size):
                down_idx = base_down + k * hidden_size + i
                dw_val = tl.load(expert_down_weights_ptr + down_idx).to(tl.float32)
                output_row[i] += prod[i, k] * dw_val

        # Atomic add weighted contribution into result
        row_start_out = pid * hidden_size
        for i in range(0, hidden_size):
            tl.atomic_add(result_ptr + row_start_out + i, output_row[i] * routing_weight)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same inputs as the original get_inputs():
        # hidden_states: [num_tokens, hidden_size] (bfloat16)
        # selected_experts: [num_tokens, num_experts_per_tok] (int64)
        # routing_weights: [num_tokens, num_experts_per_tok] (bfloat16)
        # expert_gate_weights: [num_experts, hidden_size, intermediate_size] (bfloat16)
        # expert_up_weights:   [num_experts, hidden_size, intermediate_size] (bfloat16)
        # expert_down_weights: [num_experts, intermediate_size, hidden_size] (bfloat16)

        # Extract shapes (these are provided in the evaluation via the axes)
        # We assume args[0] is hidden_states, args[1] selected_experts, args[2] routing_weights, args[3/4/5] are weights
        # In practice, forward will be called with these tensors; but since the evaluator may supply them, we simply use args.

        hidden_states = args[0]
        selected_experts = args[1]
        routing_weights = args[2]
        # expert weights are the remaining args
        # For safety, assume there are exactly 6 args:
        expert_gate_weights = args[3]
        expert_up_weights = args[4]
        expert_down_weights = args[5]

        # Ensure contiguity and cast to float32 for computation
        hidden_states_fp32 = hidden_states.to(torch.float32).contiguous()
        selected_experts_i64 = selected_experts.contiguous()
        routing_weights_fp32 = routing_weights.to(torch.float32).contiguous()
        expert_gate_weights_fp32 = expert_gate_weights.to(torch.float32).contiguous()
        expert_up_weights_fp32 = expert_up_weights.to(torch.float32).contiguous()
        expert_down_weights_fp32 = expert_down_weights.to(torch.float32).contiguous()

        num_tokens = hidden_states_fp32.shape[0]
        hidden_size = hidden_states_fp32.shape[1]
        # Derive num_experts_per_tok from selected_experts
        num_experts_per_tok = selected_experts_i64.shape[1]
        # Derive intermediate_size from gate/up weights
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
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            num_warps=1,
            num_stages=1,
        )

        # Return result cast to bfloat16 to match original
        return result_fp32.reshape(num_tokens, hidden_size).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
