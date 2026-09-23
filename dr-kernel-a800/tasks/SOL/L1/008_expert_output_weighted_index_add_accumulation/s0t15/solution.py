class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          output[token_indices[i]] += expert_outputs[i] for i in [0, N)
        output has shape (M, H), expert_outputs (N, H), token_indices (N,).
        """
        # Ensure device compatibility: Triton requires CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be CUDA for Triton"

        # Clone to initialize output; no in-place modifications to the input
        output = final_hidden_states.clone()

        # If token_indices is int64, convert to int32 for faster Triton indexing (safe as long as M < 2^31)
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Heuristic tuning for BLOCK_H and warps
        if H >= 256:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        elif H >= 128:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        elif H >= 64:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2
        else:
            BLOCK_H = 32
            num_warps = 2
            num_stages = 2

        # Grid: each program handles one source row and one H-block
        grid = (N, triton.cdiv(H, BLOCK_H))

        scatter_add_n_h_kernel[grid](
            output, expert_outputs, token_indices,
            N=N, H=H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
