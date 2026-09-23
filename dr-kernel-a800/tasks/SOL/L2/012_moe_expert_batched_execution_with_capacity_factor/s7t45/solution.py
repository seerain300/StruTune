import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    dtype = torch.bfloat16

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Generate valid expert indices - each token selects num_experts_per_tok unique experts
    selected_experts = torch.zeros(num_tokens, num_experts_per_tok, dtype=torch.int64, device=device)
    for i in range(num_tokens):
        perm = torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        selected_experts[i] = perm

    # Generate routing weights that sum to 1 for each token
    routing_logits = torch.randn(num_tokens, num_experts_per_tok, dtype=dtype, device=device)
    routing_weights = F.softmax(routing_logits.float(), dim=-1).to(dtype)

    # Expert weights: Xavier init with 1/sqrt(fan_in)
    # gate/up: shape [num_experts, hidden_size, moe_intermediate_size], fan_in = hidden_size (dim 1)
    expert_gate_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    expert_up_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    # down: shape [num_experts, moe_intermediate_size, hidden_size], fan_in = moe_intermediate_size (dim 1)
    expert_down_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(moe_intermediate_size)

    return {
        "hidden_states": hidden_states,
        "selected_experts": selected_experts,
        "routing_weights": routing_weights,
        "expert_gate_weights": expert_gate_weights,
        "expert_up_weights": expert_up_weights,
        "expert_down_weights": expert_down_weights,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, moe_intermediate_size = expert_gate_weights.shape
    num_experts_per_tok = selected_experts.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    capacity = max(int((num_tokens * num_experts_per_tok / num_experts) * 1.25), 1)

    # Flatten all token-expert assignments: (num_tokens * K,)
    flat_experts = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    flat_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(num_experts_per_tok)

    # Sort by expert ID (stable to match original sequential assignment order)
    sorted_experts, sorted_indices = flat_experts.sort(stable=True)
    sorted_weights = flat_weights[sorted_indices]
    sorted_token_ids = flat_token_ids[sorted_indices]

    # After sorting, tokens for the same expert are contiguous.
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    starts = torch.zeros(num_experts, dtype=torch.long, device=device)
    starts[1:] = counts[:-1].cumsum(0)

    # Position within group = global_sorted_index - start_of_expert_group
    within_pos = torch.arange(len(sorted_experts), device=device) - starts[sorted_experts]

    # Apply capacity constraint (same semantics: first `capacity` per expert admitted)
    valid = within_pos < capacity
    v_exp = sorted_experts[valid]
    v_pos = within_pos[valid]
    v_tok = sorted_token_ids[valid]
    v_wt = sorted_weights[valid]

    # Prepare expert inputs: expert_inputs[e, n, :] = hidden_states[token_id] for valid positions
    expert_inputs = torch.zeros(num_experts, capacity, hidden_size, dtype=dtype, device=device)
    expert_inputs[v_exp, v_pos] = hidden_states[v_tok]

    # Batched expert forward pass (same 3 BMMs as original)
    gate_out = torch.bmm(expert_inputs, expert_gate_weights)
    up_out = torch.bmm(expert_inputs, expert_up_weights)

    # SwiGLU activation: activated = silu(gate_out) * up_out
    activated = F.silu(gate_out.float()).to(dtype) * up_out

    expert_outputs = torch.bmm(activated, expert_down_weights)

    # Vectorized gather + weighted aggregation.
    valid_out = expert_outputs[v_exp, v_pos]  # (num_valid, hidden_size)
    weighted_out = v_wt.unsqueeze(1) * valid_out

    result = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
    # Accumulate contributions per token
    result.index_add_(0, v_tok, weighted_out)

    return result


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure inputs are on the same device
        # (In evaluation, inputs are already provided on the correct device.)
        return run(hidden_states, selected_experts, routing_weights, expert_gate_weights, expert_up_weights, expert_down_weights)


def run(*args):
    return ModelNew()(*args)
