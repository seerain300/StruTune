import math
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_forward_triton_kernel(
    hidden_states_ptr,          # *fp32, shape [num_tokens, hidden_size]
    selected_experts_ptr,       # *int64, shape [num_tokens, num_experts_per_tok]
    routing_weights_ptr,        # *fp32, shape [num_tokens, num_experts_per_tok]
    # expert weights as 1D flattened arrays (fp32):
    expert_gate_weights_flat_ptr,   # *fp32, shape [num_experts * hidden_size * intermediate_size]
    expert_up_weights_flat_ptr,     # *fp32, shape [num_experts * hidden_size * intermediate_size]
    expert_down_weights_flat_ptr,   # *fp32, shape [num_experts * intermediate_size * hidden_size]
    result_ptr,                  # *fp32, shape [num_tokens, hidden_size], flattened row-major
    hidden_size: tl.constexpr,  # int
    intermediate_size: tl.constexpr,  # int
    num_experts_per_tok: tl.constexpr,  # int
):
    # 2D grid: each program handles one (token, j) pair
    token = tl.program_id(0)  # int
    j = tl.program_id(1)      # int

    # Compute base offsets
    # Row base for hidden_states and result: token * hidden_size
    hidden_row_offset = token * hidden_size
    # Load selected expert id and routing weight for this (token, j)
    expert_id = tl.load(selected_experts_ptr + token * num_experts_per_tok + j)  # int64
    routing_weight = tl.load(routing_weights_ptr + token * num_experts_per_tok + j)  # fp32

    # Load hidden state row: [hidden_size]
    hidden_row = tl.load(hidden_states_ptr + hidden_row_offset + tl.arange(0, hidden_size))  # fp32, vector of length hidden_size

    # Compute gate_out: hidden_row @ gate_weight
    # gate_weight is [hidden_size, intermediate_size], flattened row-major:
    # rows = hidden_size, cols = intermediate_size
    # For each column col in [0, intermediate_size), gate_weight_row = gate_weight[expert_id, :, col]
    gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for col in range(0, intermediate_size):
        # gate_weight_row offset: ((expert_id * hidden_size * intermediate_size) + (row * intermediate_size + col))
        base_expert = expert_id * hidden_size * intermediate_size
        gate_weight_row_offset = base_expert + (tl.arange(0, hidden_size) * intermediate_size + col)
        gate_weight_row = tl.load(expert_gate_weights_flat_ptr + gate_weight_row_offset)
        # gate_out[:, col] = dot(hidden_row, gate_weight_row)
        gate_out[:, col] = tl.dot(hidden_row, gate_weight_row)

    # Compute up_out: hidden_row @ up_weight (same shape as gate_out)
    up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for col in range(0, intermediate_size):
        base_expert_up = expert_id * hidden_size * intermediate_size
        up_weight_row_offset = base_expert_up + (tl.arange(0, hidden_size) * intermediate_size + col)
        up_weight_row = tl.load(expert_up_weights_flat_ptr + up_weight_row_offset)
        up_out[:, col] = tl.dot(hidden_row, up_weight_row)

    # SiLU(gate_out): gate_out * sigmoid(gate_out)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-gate_out))
    silu = gate_out * sig

    # activated = SiLU(gate_out) * up_out
    activated = silu * up_out  # [hidden_size, intermediate_size]

    # Compute expert_outputs = activated @ down_weight, result is [hidden_size]
    expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
    for n in range(0, hidden_size):
        # down_weight_row offset: ((expert_id * intermediate_size * hidden_size) + (col * hidden_size + n))
        for col in range(0, intermediate_size):
            base_expert_down = expert_id * intermediate_size * hidden_size
            down_weight_row_offset = base_expert_down + (col * hidden_size + tl.arange(0, hidden_size))
            # We need down_weight_row[n] element, which is at offset col * hidden_size + n
            down_weight_elem = tl.load(expert_down_weights_flat_ptr + (base_expert_down + col * hidden_size + n))
            expert_outputs[n] += tl.sum(activated[n, col] * down_weight_elem)

    # Weighted contribution
    weighted = routing_weight * expert_outputs

    # Atomic add into result[token, :]
    result_row_offset = token * hidden_size
    # atomic_add supports adding to a 1D vector
    tl.atomic_add(result_ptr + result_row_offset + tl.arange(0, hidden_size), weighted)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args expected order: hidden_states, selected_experts, routing_weights,
        # expert_gate_weights, expert_up_weights, expert_down_weights
        hidden_states = args[0]
        selected_experts = args[1]  # int64
        routing_weights = args[2]   # bfloat16
        expert_gate_weights = args[3]  # bfloat16 [num_experts, hidden_size, intermediate_size]
        expert_up_weights = args[4]    # bfloat16 [num_experts, hidden_size, intermediate_size]
        expert_down_weights = args[5]  # bfloat16 [num_experts, intermediate_size, hidden_size]

        device = hidden_states.device
        dtype_hs = hidden_states.dtype  # bfloat16
        dtype_rw = routing_weights.dtype  # bfloat16

        # Extract shapes
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        # selected_experts: [num_tokens, num_experts_per_tok]
        num_experts_per_tok = selected_experts.shape[1]
        # gating weights: [num_experts, hidden_size, intermediate_size]
        num_experts = expert_gate_weights.shape[0]
        intermediate_size = expert_gate_weights.shape[2]

        # Ensure contiguity and convert to fp32 for compute
        hidden_states_fp32 = hidden_states.contiguous().to(torch.float32)          # [num_tokens, hidden_size]
        selected_experts_i64 = selected_experts.contiguous()                       # [num_tokens, num_experts_per_tok]
        routing_weights_fp32 = routing_weights.contiguous().to(torch.float32)      # [num_tokens, num_experts_per_tok]
        expert_gate_weights_fp32 = expert_gate_weights.contiguous().to(torch.float32).reshape(-1)  # 1D
        expert_up_weights_fp32 = expert_up_weights.contiguous().to(torch.float32).reshape(-1)      # 1D
        expert_down_weights_fp32 = expert_down_weights.contiguous().to(torch.float32).reshape(-1)   # 1D

        # Output buffer in fp32: [num_tokens, hidden_size]
        result_fp32 = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=device)

        # Launch Triton kernel: 2D grid over tokens and selected experts
        grid = (num_tokens, num_experts_per_tok)
        _moe_forward_triton_kernel[grid](
            hidden_states_fp32,
            selected_experts_i64,
            routing_weights_fp32,
            expert_gate_weights_fp32,
            expert_up_weights_fp32,
            expert_down_weights_fp32,
            result_fp32,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            num_warps=4,
            num_stages=2,
        )

        # Return result cast to bfloat16 to match original dtype
        return result_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
