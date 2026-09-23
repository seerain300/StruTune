import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16 pointer, shape [M, H]
    expert_ptr,            # *bf16 or *fp16 pointer, shape [N, H]
    index_ptr,             # *int32 pointer, shape [N]
    N: tl.int32,           # number of source rows (num_selected_tokens)
    H: tl.int32,           # hidden size
    BLOCK_H: tl.constexpr, # chunk size over H
):
    # Each program handles one source row n
    n = tl.program_id(axis=0)
    if n >= N:
        return

    # Load target row index for this source row
    # token_indices is int64 in PyTorch; cast to int32 for pointer arithmetic
    idx = tl.load(index_ptr + n)
    # Optionally guard in case of any unexpected values (rare)
    if idx < 0 or idx >= N:
        return

    # Iterate over H in chunks of BLOCK_H and perform atomic adds
    start = 0
    while start < H:
        h = start + tl.arange(0, BLOCK_H)
        mask = h < H

        # Load the corresponding hidden vector for this source row
        vals = tl.load(expert_ptr + n * H + h, mask=mask, other=0.0)

        # Compute destination addresses in output
        dest = idx * H + h

        # Atomic add the vector to the destination row
        tl.atomic_add(output_ptr + dest, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-only forward that performs scatter-add:
            output[token_indices[i]] += expert_outputs[i]
        Input dtypes: final_hidden_states and expert_outputs in bfloat16; token_indices in int64.
        Output: updated final_hidden_states with expert contributions added.
        """
        # Ensure device is CUDA; Triton requires GPU
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA device."

        # Clone to preserve original input; we'll update this clone
        output = final_hidden_states.clone()

        # Shapes
        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Token indices should be int32 for efficient pointer arithmetic in Triton
        # (PyTorch defaults to int64; safe to cast)
        index32 = token_indices.to(torch.int32)

        # Choose BLOCK_H based on H to balance throughput and occupancy
        if H >= 2048:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 3
        elif H >= 1024:
            BLOCK_H = 256
            num_warps = 4
            num_stages = 3
        elif H >= 512:
            BLOCK_H = 256
            num_warps = 4
            num_stages = 2
        elif H >= 256:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        # Launch Triton kernel: one program per source row
        grid = (N,)

        scatter_add_row_kernel[grid](
            output, expert_outputs, index32,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
