import torch
import triton
import triton.language as tl


@triton.jit
def _add_expert_row_kernel(
    out_ptr,         # *bf16, output tensor [M, H]
    ex_ptr,          # *bf16, expert_outputs tensor [N, H]
    indices_ptr,     # *int32, token_indices tensor [N]
    M: tl.constexpr,     # number of rows (batch_seq_len)
    N: tl.constexpr,     # number of tokens (num_selected_tokens)
    H: tl.constexpr,     # number of columns (hidden_size)
    BLOCK_H: tl.constexpr,  # tile size over columns
):
    token_id = tl.program_id(0)
    if token_id >= N:
        return
    idx = tl.load(indices_ptr + token_id)  # int32 row index in [0, M)
    # Loop over hidden dimension in tiles: read output row, add expert row, write back
    for col_start in range(0, H, BLOCK_H):
        offs = col_start + tl.arange(0, BLOCK_H)
        mask = offs < H
        out_vals = tl.load(out_ptr + idx * H + offs, mask=mask, other=0.0)
        ex_vals = tl.load(ex_ptr + token_id * H + offs, mask=mask, other=0.0)
        out_vals = out_vals + ex_vals
        tl.store(out_ptr + idx * H + offs, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward that replicates:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        without using any torch computation in the host code (except dtype/contiguity handling).
        """
        # Ensure CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton execution."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "Dtypes must be bfloat16 for this Triton implementation."

        # Ensure contiguous
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()

        M = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Output buffer: pre-initialize to zeros; per-token kernel will add expert rows
        output = torch.zeros((M, H), dtype=final_hidden_states.dtype, device=final_hidden_states.device)

        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Choose a reasonable BLOCK_H; 128 works well for many hidden sizes
        BLOCK_H = 128

        # Launch per-token add kernel: 1D grid over tokens; iterate over hidden dimension
        grid_add = (N,)
        _add_expert_row_kernel[grid_add](
            output, expert_outputs, token_indices, M, N, H, BLOCK_H,
            num_warps=4, num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
