import math
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_forward_kernel(
    hidden_states_ptr,               # *bf16, shape: [num_tokens, hidden_size]
    selected_experts_ptr,            # *int64, shape: [num_tokens, num_experts_per_tok]
    routing_weights_ptr,             # *bf16, shape: [num_tokens, num_experts_per_tok]
    expert_gate_weights_ptr,         # *bf16, shape: [num_experts, hidden_size, intermediate_size]
    expert_up_weights_ptr,           # *bf16, shape: [num_experts, hidden_size, intermediate_size]
    expert_down_weights_ptr,         # *bf16, shape: [num_experts, intermediate_size, hidden_size]
    result_ptr,                      # *fp32, shape: [num_tokens, hidden_size], output buffer
    num_tokens: tl.int32,
    hidden_size: tl.int32,
    num_experts_per_tok: tl.int32,
    intermediate_size: tl.int32,
):
    # 2D grid: (token_id, expert_j)
    t = tl.program_id(0)  # token index
    j = tl.program_id(1)  # selected expert index per token

    # Compute offsets for hidden state row
    hidden_row_offsets = t * hidden_size + tl.arange(0, hidden_size)
    hidden_row = tl.load(hidden_states_ptr + hidden_row_offsets).to(tl.float32)  # [hidden_size], fp32

    # Load selected expert id for (t, j)
    expert_id = tl.load(selected_experts_ptr + t * num_experts_per_tok + j)  # int64

    # Load routing weight for (t, j)
    routing_w = tl.load(routing_weights_ptr + t * num_experts_per_tok + j).to(tl.float32)  # fp32

    # Compute gate_out = hidden_row @ gate_weight[expert_id, :, :]
    gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for k in range(0, hidden_size):
        hidden_k = hidden_row[k]  # scalar fp32
        for m in range(0, intermediate_size):
            gate_w_ptr = expert_gate_weights_ptr + expert_id * hidden_size * intermediate_size + k * intermediate_size + m
            gate_w = tl.load(gate_w_ptr).to(tl.float32)
            gate_out[k, m] = hidden_k * gate_w

    # Compute up_out = hidden_row @ up_weight[expert_id, :, :]
    up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for k in range(0, hidden_size):
        hidden_k = hidden_row[k]  # scalar fp32
        for m in range(0, intermediate_size):
            up_w_ptr = expert_up_weights_ptr + expert_id * hidden_size * intermediate_size + k * intermediate_size + m
            up_w = tl.load(up_w_ptr).to(tl.float32)
            up_out[k, m] = hidden_k * up_w

    # Apply SiLU: silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
    activated = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for k in range(0, hidden_size):
        for m in range(0, intermediate_size):
            z = gate_out[k, m]
            sig = 1.0 / (1.0 + tl.exp(-z))
            activated[k, m] = z * sig * up_out[k, m]

    # Compute expert_outputs = activated @ down_weight[expert_id, :, :]
    expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
    for k in range(0, hidden_size):
        for m in range(0, intermediate_size):
            down_w_ptr = expert_down_weights_ptr + expert_id * intermediate_size * hidden_size + m * hidden_size + k
            down_w = tl.load(down_w_ptr).to(tl.float32)
            expert_outputs[k] += activated[k, m] * down_w

    # Atomic add weighted contribution into result[t, :]
    result_row_offsets = t * hidden_size + tl.arange(0, hidden_size)
    contrib_vec = tl.zeros((hidden_size,), dtype=tl.float32)
    for k in range(0, hidden_size):
        contrib_vec[k] = expert_outputs[k] * routing_w
    for k in range(0, hidden_size):
        tl.atomic_add(result_ptr + result_row_offsets[k], contrib_vec[k])


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args correspond to:
        # hidden_states: [num_tokens, hidden_size], bfloat16
        # selected_experts: [num_tokens, num_experts_per_tok], int64
        # routing_weights: [num_tokens, num_experts_per_tok], bfloat16
        # expert_gate_weights: [num_experts, hidden_size, intermediate_size], bfloat16
        # expert_up_weights:   [num_experts, hidden_size, intermediate_size], bfloat16
        # expert_down_weights: [num_experts, intermediate_size, hidden_size], bfloat16
        hidden_states = args[0]
        selected_experts = args[1]
        routing_weights = args[2]
        expert_gate_weights = args[3]
        expert_up_weights = args[4]
        expert_down_weights = args[5]

        device = hidden_states.device
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts_per_tok = selected_experts.shape[1]
        intermediate_size = expert_gate_weights.shape[2]

        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        selected_experts = selected_experts.contiguous()
        routing_weights = routing_weights.contiguous()

        # Output buffer in fp32
        result_fp32 = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid over tokens and selected experts per token
        grid = (num_tokens, num_experts_per_tok)
        _moe_forward_kernel[grid](
            hidden_states,
            selected_experts,
            routing_weights,
            expert_gate_weights,
            expert_up_weights,
            expert_down_weights,
            result_fp32,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            num_warps=1,
            num_stages=1,
        )

        # Return result in bfloat16 to match original
        return result_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
