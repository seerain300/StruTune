class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure CUDA tensors
        if hidden_states.device.type != "cuda":
            if torch.cuda.is_available():
                hidden_states = hidden_states.to("cuda")
                selected_experts = selected_experts.to("cuda")
                routing_weights = routing_weights.to("cuda")
                expert_gate_weights = expert_gate_weights.to("cuda")
                expert_up_weights = expert_up_weights.to("cuda")
                expert_down_weights = expert_down_weights.to("cuda")
            else:
                raise RuntimeError("CUDA not available. Triton requires CUDA.")

        # Run Triton-integrated computation
        result = run_triton(hidden_states, selected_experts, routing_weights, expert_gate_weights, expert_up_weights, expert_down_weights)
        return result


def run(*args):
    return ModelNew()(*args)
