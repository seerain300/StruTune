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
    BLOCK_H: tl.constexpr, # tile size along H
):
    pid = tl.program_id(axis=0)  # one program per source row (pid in [0, N))
    if pid >= tl.num_programs(axis=0):
        return

    # Each program handles one source row 'pid' and accumulates over all H in chunks of BLOCK_H
    start = 0
    while start < H:
        h_offsets = start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load token index for this source row
        dest_row = tl.load(index_ptr + pid, mask=True, other=0)  # int32
        dest_row = dest_row.to(tl.int32)

        # Compute linear offsets for output and expert rows
        out_offsets = dest_row * H + h_offsets
        expert_row_offsets = pid * H + h_offsets

        # Load current output and expert values for this chunk
        out_vals = tl.load(output_ptr + out_offsets, mask=mask, other=0.0)
        exp_vals = tl.load(expert_ptr + expert_row_offsets, mask=mask, other=0.0)

        # Atomic add
        tl.atomic_add(output_ptr + out_offsets, exp_vals, mask=mask)

        start += BLOCK_H


def _choose_launch_params(H: int):
    # Heuristic tuning for different H sizes
    if H >= 2048:
        return 512, 8, 4
    elif H >= 1024:
        return 256, 8, 3
    elif H >= 256:
        return 128, 4, 3
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure contiguity and dtypes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        assert final_hidden_states.dtype in (torch.bfloat16, torch.float16), "Expected bf16/fp16 dtype"
        assert expert_outputs.dtype == final_hidden_states.dtype, "Dtype of expert_outputs must match output dtype"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64"

        # Make sure tensors are contiguous
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Convert indices to int32 for efficient addressing (safe for these workloads since M,H fit in int32)
        if token_indices.dtype == torch.int64:
            token_indices = token_indices.to(torch.int32)
        assert token_indices.dtype == torch.int32, "token_indices must be int32 after conversion"

        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = token_indices.shape[0]

        # Launch grid: one program per source row
        grid = (N,)

        BLOCK_H, num_warps, num_stages = _choose_launch_params(H)

        scatter_add_row_kernel[grid](
            final_hidden_states, expert_outputs, token_indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return final_hidden_states


def run(*args):
    return ModelNew()(*args)
