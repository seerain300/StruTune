import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16 pointer, shape [M, H]
    expert_ptr,            # *bf16 or *fp16 pointer, shape [N, H]
    index_ptr,             # *int32 pointer, shape [N]
    M: tl.constexpr,       # number of rows in output (M = batch_size * seq_len)
    N: tl.constexpr,       # number of source rows (N = batch_seq_len * num_experts_per_tok)
    H: tl.constexpr,       # hidden size
    BLOCK_H: tl.constexpr  # chunk size over H
):
    # Each program handles one source row 'n'
    n = tl.program_id(axis=0)
    if n >= N:
        return

    # Load token index for this source row (int32)
    dest_row = tl.load(index_ptr + n)

    # Iterate over H in chunks of size BLOCK_H
    h_start = 0
    while h_start < H:
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load the corresponding expert outputs for this row and chunk
        # Row-major addressing: expert_ptr + n * H + offs
        vals = tl.load(expert_ptr + n * H + offs, mask=mask, other=0)

        # Compute destination addresses: output_ptr + dest_row * H + offs
        dest_addrs = output_ptr + dest_row * H + offs

        # Atomic add the values into the output
        tl.atomic_add(dest_addrs, vals, mask=mask)

        h_start += BLOCK_H


def _choose_launch_params(H: int):
    # Heuristic launch parameter selection based on H
    if H >= 4096:
        return 256, 8, 3
    elif H >= 1024:
        return 128, 4, 2
    elif H >= 256:
        return 128, 4, 2
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton prefers int32 for indices; ensure it and bounds
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Output is the same tensor we need to accumulate into
        output = final_hidden_states

        # Choose kernel launch params
        BLOCK_H, num_warps, num_stages = _choose_launch_params(H)

        # Grid: one program per source row
        grid = (N,)

        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            M, N, H, BLOCK_H,
            num_warps=num_warps, num_stages=num_stages
        )

        return output


def run(*args):
    return ModelNew()(*args)
