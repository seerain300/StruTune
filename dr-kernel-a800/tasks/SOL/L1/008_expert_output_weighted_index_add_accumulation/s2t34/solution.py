class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-optimized index_add along dim=0:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure tensors are contiguous and dtypes are as expected
        out = final_hidden_states.clone().contiguous()
        expert = expert_outputs.contiguous()
        indices = token_indices.to(torch.int32).contiguous()

        N = expert.shape[0]
        H = expert.shape[1]

        # Choose BLOCK deterministically based on H to minimize chunks
        if H <= 64:
            BLOCK = 64
            num_warps = 4
        elif H <= 128:
            BLOCK = 128
            num_warps = 4
        else:
            BLOCK = 256
            num_warps = 8

        # Launch one program per selected token row
        grid = (N,)
        _index_add_rows_kernel[grid](out, expert, indices, N, H, BLOCK=BLOCK, num_warps=num_warps, num_stages=2)
        return out


def run(*args):
    return ModelNew()(*args)
