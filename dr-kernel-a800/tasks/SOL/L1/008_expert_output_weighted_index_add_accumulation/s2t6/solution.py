import torch


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Performs scatter-add of expert_outputs into final_hidden_states along dim=0 using indices.
    That is: output[token_indices[i]] += expert_outputs[i] for all i.
    """
    # Clone to preserve original random initialization (matches get_inputs behavior)
    output = final_hidden_states.clone()
    # Accumulate using index_add along dim=0
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure the input tensors are on a device where index_add is supported
        # (PyTorch handles CPU/GPU; we just call index_add as in the original)
        return run(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
