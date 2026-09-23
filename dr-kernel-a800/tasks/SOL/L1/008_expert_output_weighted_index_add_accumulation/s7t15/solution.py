import torch


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Match reference behavior: clone first, then accumulate
        out = final_hidden_states.clone()

        # Ensure tensors are on CUDA (get_inputs places tensors on device already)
        assert out.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        # index_add expects indices as long (int64)
        token_indices = token_indices.to(torch.long)

        # index_add accumulates along dim=0: out[index[i]] += source[i, :]
        out.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return out


def run(*args):
    return ModelNew()(*args)
