class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward that performs the accumulation without using PyTorch index_add.
        final_hidden_states: output buffer of shape (batch_seq_len, hidden_size), dtype bfloat16, on CUDA
        expert_outputs: weighted expert outputs of shape (num_selected_tokens, hidden_size), dtype bfloat16, on CUDA
        token_indices: long tensor of shape (num_selected_tokens,), values in [0, batch_seq_len)
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA device for Triton kernel"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA device for Triton kernel"
        assert token_indices.is_cuda, "token_indices must be on CUDA device for Triton kernel"

        # Ensure dtypes and contiguity
        H = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        num_selected_tokens = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == hidden_size, "expert_outputs' second dim must match hidden_size"
        assert token_indices.shape[0] == num_selected_tokens, "token_indices length must match num_selected_tokens"

        # Triton expects int32 indices; cast safely
        # token_indices are in [0, H), so int32 is fine
        idx32 = token_indices.to(torch.int32)

        # Allocate output as zeros; we will fill it via Triton kernel
        # If you want to keep the original final_hidden_states intact, create a new output
        output = torch.empty_like(final_hidden_states)
        # Initialize to zeros (not strictly necessary if we only write once per token, but we keep zeros to match original semantics)
        # However, since we write all contributions in the kernel, we can skip zeros and write initial values.
        # To be safe, initialize zeros.
        output.zero_()

        # Choose a block size for columns
        BLOCK_SIZE = 128  # good default; adjust if hidden_size is very large

        # Launch a 1D grid over tokens
        grid = (num_selected_tokens,)

        scatter_add_expert_kernel[grid](
            output, idx32, expert_outputs,
            H, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
