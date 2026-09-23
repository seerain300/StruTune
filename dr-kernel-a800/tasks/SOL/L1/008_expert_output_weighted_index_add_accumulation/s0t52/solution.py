import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16, shape [M, H], row-major contiguous
    expert_ptr,            # *bf16 or *fp16, shape [N, H], row-major contiguous
    index_ptr,             # *int32, shape [N]
    M: tl.int32,           # number of rows in output (M = batch_size * seq_len)
    H: tl.int32,           # hidden size
    BLOCK_H: tl.constexpr, # tile size along H (compile-time for vectorization)
):
    pid = tl.program_id(axis=0)  # one program per source row index
    # Guard: if pid >= N, exit (grid may be sized to N or larger)
    # Triton kernels typically launch with grid=(N,), so this guard is mostly for safety.
    if pid >= tl.num_programs(axis=0):
        return

    # Load the destination row index for this source row
    dest_row = tl.load(index_ptr + pid)  # int32
    dest_row = dest_row  # keep for address calc

    # Iterate over H in chunks of BLOCK_H
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)  # vector of H offsets for this chunk
        mask = offs < H  # valid mask for the last chunk

        # Load expert_outputs[n, offs] as a vector
        # Pointer arithmetic: row n, column offs
        exp_row_ptr = expert_ptr + pid * H
        vals = tl.load(exp_row_ptr + offs, mask=mask, other=0.0)  # dtype follows pointer

        # Compute output addresses: output[dest_row, offs]
        out_row_ptr = output_ptr + dest_row * H
        # Perform atomic add
        tl.atomic_add(out_row_ptr + offs, vals, mask=mask)

        start += BLOCK_H


def _select_launch_params(H: int):
    # Heuristic tuning for BLOCK_H, num_warps, num_stages
    if H >= 4096:
        return 512, 8, 4
    elif H >= 2048:
        return 256, 8, 4
    elif H >= 1024:
        return 256, 8, 3
    elif H >= 512:
        return 128, 4, 3
    elif H >= 256:
        return 128, 4, 2
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA device"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA device"
        assert token_indices.is_cuda, "token_indices must be on CUDA device"

        # Ensure contiguity and dtypes; we keep bf16/fp16 and int32 for indices
        # Make inputs contiguous
        output = final_hidden_states  # do not modify input in-place
        # Create a writable copy of final_hidden_states to accumulate into
        output = output.clone()

        M = output.shape[0]
        H = output.shape[1]
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs second dimension must match hidden size"

        # Triton prefers int32 indices; ensure dtype and contiguity
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Select launch parameters
        BLOCK_H, num_warps, num_stages = _select_launch_params(H)

        # Launch one program per source row
        grid = (N,)
        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
