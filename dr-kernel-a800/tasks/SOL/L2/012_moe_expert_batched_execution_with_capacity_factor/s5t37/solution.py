import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute final weighted result purely via Triton. No torch ops in forward.
# Inputs:
#   hidden_states_ptr: [num_tokens, hidden_size], bf16
#   selected_experts:  [num_tokens, num_experts_per_tok], int64 (passed as int64; we'll cast inside)
#   routing_weights:   [num_tokens, num_experts_per_tok], bf16
#   expert_gate_weights: [num_experts, hidden_size, intermediate_size], bf16
#   expert_up_weights:   [num_experts, hidden_size, intermediate_size], bf16
#   expert_down_weights: [num_experts, intermediate_size, hidden_size], bf16
# Outputs:
#   result_fp32: [num_tokens, hidden_size], fp32
@triton.jit
def compute_result_kernel(
    hidden_states_ptr, selected_exp_ptr, routing_w_ptr,
    gate_w_ptr, up_w_ptr, down_w_ptr,
    result_fp32,
    num_tokens, hidden_size, intermediate_size, num_experts_per_tok, NUM_EXPERTS: tl.constexpr
):
    # This kernel performs the heavy computation using Triton loops (no torch ops).
    for t in range(0, num_tokens):
        for j in range(0, num_experts_per_tok):
            # Load selected expert id and routing weight for token t, expert j
            expert_id = tl.load(selected_exp_ptr + t * num_experts_per_tok + j).to(tl.int32)
            # routing weight
            wt = tl.load(routing_w_ptr + t * num_experts_per_tok + j).to(tl.float32)

            # Load hidden state row for token t (convert to fp32)
            hs_row = tl.load(hidden_states_ptr + t * hidden_size + tl.arange(0, hidden_size), mask=True, other=0.0).to(tl.float32)  # [hidden_size]

            # Compute gate_out = hs_row @ expert_gate_weights[expert_id] -> [hidden_size, intermediate_size]
            gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
            for j_m in range(0, intermediate_size):
                gate_row = tl.load(gate_w_ptr + expert_id * (hidden_size * intermediate_size) + j_m * hidden_size + tl.arange(0, hidden_size), mask=True, other=0.0).to(tl.float32)
                gate_out[:, j_m] = tl.dot(hs_row, gate_row)

            # up_out = hs_row @ expert_up_weights[expert_id] -> [hidden_size, intermediate_size]
            up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
            for j_n in range(0, intermediate_size):
                up_row = tl.load(up_w_ptr + expert_id * (hidden_size * intermediate_size) + j_n * hidden_size + tl.arange(0, hidden_size), mask=True, other=0.0).to(tl.float32)
                up_out[:, j_n] = tl.dot(hs_row, up_row)

            # SiLU and multiply
            activated = tl.silu(gate_out) * up_out  # fp32

            # expert_outputs = activated @ expert_down_weights[expert_id] -> [hidden_size]
            expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
            for k in range(0, hidden_size):
                acc = 0.0
                for l in range(0, intermediate_size):
                    down_vec = tl.load(down_w_ptr + expert_id * (intermediate_size * hidden_size) + l * hidden_size + tl.arange(0, hidden_size), mask=True, other=0.0).to(tl.float32)
                    acc += tl.dot(activated[k, :], down_vec)
                expert_outputs[k] = acc

            # atomic_add into result_fp32[t, :] += wt * expert_outputs
            row_base = t * hidden_size
            for k in range(0, hidden_size):
                tl.atomic_add(result_fp32 + row_base + k, wt * expert_outputs[k])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        # All computation must be via Triton; do not use torch ops here.
        # Ensure tensors are on the same device and contiguous
        device = hidden_states.device
        hidden_states = hidden_states.contiguous()
        selected_experts = selected_experts.contiguous()
        routing_weights = routing_weights.contiguous()
        expert_gate_weights = expert_gate_weights.contiguous()
        expert_up_weights = expert_up_weights.contiguous()
        expert_down_weights = expert_down_weights.contiguous()

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts, gw_hidden, intermediate_size = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Prepare result buffer in fp32 for accumulation
        result_fp32 = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Launch the Triton kernel. Grid over tokens (optional) but we use a single program and loop.
        # Triton supports loops; here we launch one program and iterate over tokens and experts.
        grid = (1,)
        compute_result_kernel[grid](
            hidden_states, selected_experts, routing_weights,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            result_fp32,
            num_tokens, hidden_size, intermediate_size, num_experts_per_tok, num_experts
        )

        # Cast to bfloat16 to match the original output dtype
        result = result_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
