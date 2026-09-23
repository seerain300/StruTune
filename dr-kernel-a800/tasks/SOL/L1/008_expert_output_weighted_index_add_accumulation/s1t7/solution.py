import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_kernel(
    output_ptr,          # *bfloat16, shape [M, H]
    source_ptr,          # *bfloat16, shape [N, H]
    index_ptr,           # *int32,    shape [N]
    M: tl.constexpr,     # int: number of rows in output
    N: tl.constexpr,     # int: number of source rows
    H: tl.constexpr      # int: hidden size
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this source row
    dest = tl.load(index_ptr + pid)  # int32

    # Loop over hidden dimension and atomic add per element
    for j in range(0, H):
        src_val = tl.load(source_ptr + pid * H + j)  # bfloat16
        tl.atomic_add(output_ptr + dest * H + j, src_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of the original run() function.
        Performs atomic accumulation: output[token_indices[i]] += expert_outputs[i] for all i.
        Returns the updated output tensor.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."

        # Clone to match original semantics (index_add is in-place on a clone)
        output = final_hidden_states.clone()
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        M = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        N = expert_outputs.shape[0]  # num_selected_tokens

        # Triton grid: one program per source row
        grid = (N,)

        # Cast indices to int32 for Triton
        index_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel
        scatter_add_rows_atomic_kernel[grid](
            output,            # output_ptr
            expert_outputs,    # source_ptr
            index_i32,         # index_ptr
            M,                 # M
            N,                 # N
            H                  # H
        )

        return output


def run(*args):
    return ModelNew()(*args)
