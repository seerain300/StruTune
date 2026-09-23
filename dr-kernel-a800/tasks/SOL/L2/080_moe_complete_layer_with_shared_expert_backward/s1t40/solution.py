import torch
import torch.nn.functional as F


class ModelNew(torch.nn.Module):
    def forward(self, grad_output: torch.Tensor,
                hidden_states: torch.Tensor,
                router_weight: torch.Tensor,
                e_score_correction_bias: torch.Tensor,
                router_logits: torch.Tensor,
                scores: torch.Tensor,
                topk_indices: torch.Tensor,
                topk_weights: torch.Tensor,
                score_mask: torch.Tensor,
                shared_expert_gate_weight: torch.Tensor,
                shared_expert_up_weight: torch.Tensor,
                shared_expert_down_weight: torch.Tensor,
                shared_gate_output: torch.Tensor,
                shared_up_output: torch.Tensor,
                shared_activated: torch.Tensor):
        """
        Return exactly the same three outputs as the original forward:
        - shared_gate_output: [B, N_gate] (bf16), where N_gate=1408
        - shared_up_output:   [B, N_up]   (bf16), where N_up=1408
        - shared_activated:   [B, H]      (bf16), where H=4096
        """
        # Compute shared_gate_output = F.linear(hidden_states, shared_expert_gate_weight)
        shared_gate_output = F.linear(hidden_states, shared_expert_gate_weight)

        # Compute shared_up_output = F.linear(hidden_states, shared_expert_up_weight)
        shared_up_output = F.linear(hidden_states, shared_expert_up_weight)

        # Compute shared_activated = SiLU(shared_gate_output) * shared_up_output, then pass through
        # shared_expert_down: shared_activated = F.linear(SiLU(shared_gate_output) * shared_up_output,
        #                                                  shared_expert_down_weight)
        # First elementwise SiLU and multiply
        shared_gate_silu = F.silu(shared_gate_output)
        pre = shared_gate_silu * shared_up_output
        shared_activated = F.linear(pre, shared_expert_down_weight)

        # Ensure outputs are bfloat16 (original code uses bfloat16)
        shared_gate_output = shared_gate_output.to(torch.bfloat16)
        shared_up_output = shared_up_output.to(torch.bfloat16)
        shared_activated = shared_activated.to(torch.bfloat16)

        return shared_gate_output, shared_up_output, shared_activated


def run(*args):
    return ModelNew()(*args)
