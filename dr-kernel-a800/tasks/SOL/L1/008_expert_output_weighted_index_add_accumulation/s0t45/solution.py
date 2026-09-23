import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 pointer, shape [M, H]
    expert_ptr,            # *bf16 pointer, shape [N, H]
    index_ptr,             # *int32 pointer, shape [N]
    M, H, N,               # int32 scalars
    BLOCK_H: tl.constexpr  # tile size along H
):
    # Each program handles one source row (n in [0, N))
    n = tl.program_id(0)
    if n >= N:
        return

    # Process H in chunks of BLOCK_H
    h = 0
    while h < H:
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load token index for this source row (int32)
        idx = tl.load(index_ptr + n)  # scalar
        idx = idx.to(tl.int32)

        # Base pointers for this row (row-major: stride = H)
        out_row_ptr = output_ptr + idx * H
        exp_row_ptr = expert_ptr + n * H

        # Load expert output chunk (bf16)
        exp_chunk = tl.load(exp_row_ptr + h_offsets, mask=mask, other=0.0)

        # Atomic add into output
        dest_ptr = out_row_ptr + h_offsets
        tl.atomic_add(dest_ptr, exp_chunk, mask=mask)

        h += BLOCK_H


def _select_launch_params(H: int):
    # Heuristics tuned for performance
    if H >= 1024:
        return 256, 8, 3   # BLOCK_H, num_warps, num_stages
    elif H >= 256:
        return 128, 4, 2
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-accelerated scatter-add:
            output[token_indices[i]] += expert_outputs[i]
        where output = final_hidden_states (shape [M, H])
              expert_outputs = shape [N, H]
              token_indices = shape [N], dtype long or int32
        """
        # Ensure CUDA tensors for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors"

        # Ensure contiguous and proper dtypes
        output = final_hidden_states.contiguous()  # bf16
        expert_outputs = expert_outputs.contiguous()  # bf16
        # Triton prefers int32 indices for address arithmetic
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        M, H = output.shape
        N = expert_outputs.shape[0]

        # Choose launch parameters
        BLOCK_H, num_warps, num_stages = _select_launch_params(H)

        # Launch grid: one program per source row
        grid = (N,)

        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            M, H, N,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
