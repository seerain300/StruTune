import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,              # *bf16
    expert_ptr,           # *bf16
    indices_ptr,          # *int64 (token_indices)
    N,                    # int32: number of rows to process (sum of token_indices)
    hidden_size,          # int32: size of each row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    pid = tl.program_id(0)
    # Bounds check: if pid >= N, do nothing (defensive, though grid will match N)
    if pid >= N:
        return

    # Load destination row index (token position) for this row
    # indices_ptr is int64 in the input; cast to int32 for pointer arithmetic
    row_idx = tl.load(indices_ptr + pid).to(tl.int32)

    # Loop over hidden dimension and do atomic add
    # This preserves correctness for duplicate row_idx (atomics)
    for j in range(0, hidden_size):
        src_off = pid * hidden_size + j
        dst_off = row_idx * hidden_size + j
        # Atomic add of a single element
        tl.atomic_add(out_ptr + dst_off, tl.load(expert_ptr + src_off))


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        # Match reference: clone the input before accumulation
        out = final_hidden_states.clone()

        # Ensure contiguity and device placement
        # Triton kernels require CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA for Triton execution."

        # Make sure tensors are contiguous
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        N = token_indices.numel()  # number of expert outputs
        hidden_size = expert_outputs.shape[1]

        # Launch kernel: one program per row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            N, hidden_size,
            BLOCK_SIZE=1,  # one row per program; avoid over-large BLOCK_SIZE
            num_warps=4,   # moderate parallelism
        )

        return out


def run(*args):
    return ModelNew()(*args)
