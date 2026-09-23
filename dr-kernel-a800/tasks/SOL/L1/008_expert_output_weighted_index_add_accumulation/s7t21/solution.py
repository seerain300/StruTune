import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_kernel(
    out_ptr,        # *bf16, shape (N, H)
    src_ptr,        # *bf16, shape (M, H)
    indices_ptr,    # *int32, shape (M,)
    N,              # int32, number of rows in out (batch_seq_len)
    H: tl.constexpr # hidden size (compile-time constant for loop)
):
    # One program per selected token (row i in src)
    i = tl.program_id(0)
    if i >= N:
        return

    # Destination row index for this selected token
    row_idx = tl.load(indices_ptr + i)  # int32

    # Loop over hidden dimension and accumulate with atomic add
    for j in range(H):
        # Compute source and destination addresses
        src_offset = i * H + j
        dst_offset = row_idx * H + j

        # Load source value (bf16)
        val = tl.load(src_ptr + src_offset)

        # Atomic add to destination
        tl.atomic_add(out_ptr + dst_offset, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton kernel."

        # Clone to match reference behavior (index_add on a fresh buffer)
        out = final_hidden_states.clone()

        # Ensure contiguity and dtypes
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        indices_i32 = token_indices.to(torch.int32)

        N = out.shape[0]  # batch_seq_len
        H = out.shape[1]  # hidden_size

        # Launch 1D grid over selected tokens (each program handles one i)
        grid = (N,)
        scatter_add_per_row_kernel[grid](
            out, expert_outputs, indices_i32,
            N,
            H,
            num_warps=1,  # simple kernel; 1 warp per program is sufficient
            num_stages=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
