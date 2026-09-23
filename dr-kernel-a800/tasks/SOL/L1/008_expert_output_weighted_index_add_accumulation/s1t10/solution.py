import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_kernel(
    out_ptr,            # *bfloat16, shape (M, H)
    src_ptr,            # *bfloat16, shape (N, H)
    idx_ptr,            # *int32,    shape (N,)
    M: tl.constexpr,    # rows in output
    H: tl.constexpr,    # hidden size
    N: tl.constexpr,    # number of source rows
):
    # One program handles one source row 'row'
    row = tl.program_id(0)
    # If row >= N, do nothing (defensive; Triton grid will ensure row < N)
    if row >= N:
        return

    # Iterate over hidden dimension and perform atomic add per element
    # H is a constexpr; Triton will unroll or loop as needed.
    for j in range(0, H):
        # Load source value (bfloat16)
        val = tl.load(src_ptr + row * H + j)
        # Load destination row index (int32)
        dst_row = tl.load(idx_ptr + row)
        # Compute output pointer and atomic add
        out_ptr_ij = out_ptr + dst_row * H + j
        tl.atomic_add(out_ptr_ij, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        """
        # Ensure contiguity and dtypes
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        out = final_hidden_states.clone()  # mimic original semantics: clone before index_add

        # Triton expects int32 for indices; convert if necessary
        if token_indices.dtype != torch.int32:
            idx_i32 = token_indices.to(torch.int32)
        else:
            idx_i32 = token_indices

        # Ensure expert_outputs is contiguous
        src = expert_outputs.contiguous()
        out = out.contiguous()

        M = out.shape[0]
        H = out.shape[1]
        N = src.shape[0]

        # Launch: one program per row
        grid = (N,)
        scatter_add_rows_atomic_kernel[grid](
            out, src, idx_i32,
            M=M, H=H, N=N,
            num_warps=4, num_stages=2
        )
        return out


def run(*args):
    return ModelNew()(*args)
