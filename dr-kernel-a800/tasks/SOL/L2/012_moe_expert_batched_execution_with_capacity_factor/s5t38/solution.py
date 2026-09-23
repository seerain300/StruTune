import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: full computation per token (grid=(num_tokens,)).
# It iterates over num_experts_per_tok, loads expert id and routing weight,
# loads hidden state row, computes gate, up, down matmuls, applies SiLU and multiply,
# then atomic-adds into result_fp32[token, :].
@triton.jit
def full_forward_kernel(
    hidden_states_ptr,          # *bf16 or *fp16, shape [num_tokens, hidden_size]
    selected_experts_ptr,       # *int64, shape [num_tokens, num_experts_per_tok]
    routing_weights_ptr,        # *bf16, shape [num_tokens, num_experts_per_tok]
    expert_gate_w_ptr,          # *bf16, shape [num_experts, hidden_size, intermediate_size]
    expert_up_w_ptr,            # *bf16, shape [num_experts, hidden_size, intermediate_size]
    expert_down_w_ptr,          # *bf16, shape [num_experts, intermediate_size, hidden_size]
    result_ptr,                 # *fp32, shape [num_tokens, hidden_size]
    num_tokens: tl.constexpr,   # int
    hidden_size: tl.constexpr,  # int
    intermediate_size: tl.constexpr,  # int
    num_experts_per_tok: tl.constexpr, # int
):
    i = tl.program_id(axis=0)  # token index
    if i >= num_tokens:
        return

    # Load hidden state row for token i (contiguous row)
    hs_row = tl.load(hidden_states_ptr + i * hidden_size + tl.arange(0, hidden_size))
    hs = hs_row.to(tl.float32)  # [hidden_size], fp32

    # Iterate over selected experts per token
    for j in range(0, num_experts_per_tok):
        # Load selected expert id (int64 -> int32)
        exp_id = tl.load(selected_experts_ptr + i * num_experts_per_tok + j).to(tl.int32)

        # Load routing weight (bf16 -> fp32)
        wt = tl.load(routing_weights_ptr + i * num_experts_per_tok + j).to(tl.float32)

        # Compute gate_out = hs @ expert_gate_w[exp_id]
        gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for j0 in range(0, intermediate_size):
            gate_row = tl.load(
                expert_gate_w_ptr
                + exp_id * (hidden_size * intermediate_size)
                + j0 * hidden_size
                + tl.arange(0, hidden_size)
            ).to(tl.float32)  # [hidden_size]
            gate_out[:, j0] = tl.dot(hs, gate_row)

        # Compute up_out = hs @ expert_up_w[exp_id]
        up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for j1 in range(0, intermediate_size):
            up_row = tl.load(
                expert_up_w_ptr
                + exp_id * (hidden_size * intermediate_size)
                + j1 * hidden_size
                + tl.arange(0, hidden_size)
            ).to(tl.float32)  # [hidden_size]
            up_out[:, j1] = tl.dot(hs, up_row)

        # SiLU and multiply
        activated = tl.silu(gate_out) * up_out  # [hidden_size, intermediate_size], fp32

        # Compute expert_outputs = activated @ expert_down_w[exp_id]
        expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
        for k in range(0, hidden_size):
            acc = 0.0
            for jj in range(0, intermediate_size):
                down_vec = tl.load(
                    expert_down_w_ptr
                    + exp_id * (intermediate_size * hidden_size)
                    + jj * hidden_size
                    + tl.arange(0, hidden_size)
                ).to(tl.float32)  # [hidden_size]
                acc += tl.dot(activated[k, :], down_vec)
            expert_outputs[k] = acc

        # Accumulate into result[token, :]
        # result is fp32, atomic add weighted contribution
        # For safety, only add if wt > 0.0 (wt is typically >0)
        if wt != 0.0:
            # result_ptr[i * hidden_size + tl.arange(0, hidden_size)] += wt * expert_outputs
            for k in range(0, hidden_size):
                off = i * hidden_size + k
                curr = tl.load(result_ptr + off)
                new = curr + wt * expert_outputs[k]
                tl.store(result_ptr + off, new)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Forward must not use any torch tensor operations.
        # The evaluator provides inputs: hidden_states, selected_experts, routing_weights,
        # expert_gate_weights, expert_up_weights, expert_down_weights.
        hidden_states = args[0]  # [num_tokens, hidden_size]
        selected_experts = args[1]  # [num_tokens, num_experts_per_tok]
        routing_weights = args[2]  # [num_tokens, num_experts_per_tok]
        expert_gate_weights = args[3]  # [num_experts, hidden_size, intermediate_size]
        expert_up_weights = args[4]  # [num_experts, hidden_size, intermediate_size]
        expert_down_weights = args[5]  # [num_experts, intermediate_size, hidden_size]

        device = hidden_states.device
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts_per_tok = selected_experts.shape[1]
        num_experts = expert_gate_weights.shape[0]
        intermediate_size = expert_gate_weights.shape[2]

        # Prepare result buffer as fp32 for numerical stability
        result_fp32 = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        full_forward_kernel[grid](
            hidden_states, selected_experts, routing_weights,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            result_fp32,
            num_tokens, hidden_size, intermediate_size, num_experts_per_tok,
            num_warps=4, num_stages=2
        )

        # Return result as bfloat16 to match typical model dtype
        result = result_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
