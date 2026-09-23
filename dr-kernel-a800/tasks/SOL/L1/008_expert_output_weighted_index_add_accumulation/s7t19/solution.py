import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_rows_kernel(
    out_ptr,          # *bf16, pointer to out tensor [N, H]
    src_ptr,          # *bf16, pointer to src tensor [M, H]
    indices_ptr,      # *int32, pointer to token indices [M]
    N,                # int32, number of rows in out (batch_seq_len)
    H,                # int32, hidden size
):
    pid = tl.program_id(axis=0)
    # Each program handles one row; assume grid == N for correctness.
    # If grid > N, we could guard, but here we set grid=(N,) in host code.
    row_idx = tl.load(indices_ptr + pid)

    # Loop over hidden dimension and atomic add
    for j in range(0, H):
        out_elem_ptr = out_ptr + row_idx * H + j
        src_elem_ptr = src_ptr + pid * H + j
        val = tl.load(src_elem_ptr)
        tl.atomic_add(out_elem_ptr, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Match reference: clone the accumulation buffer
        out = final_hidden_states.clone()

        # Ensure tensors are contiguous and on CUDA for Triton
        assert out.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton expects int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        N = out.shape[0]  # batch_seq_len (i.e., number of tokens)
        M = expert_outputs.shape[0]  # number of selected tokens
        H = out.shape[1]

        # Launch 1D grid over rows
        grid = (N,)
        scatter_add_atomic_rows_kernel[grid](
            out, expert_outputs, token_indices,
            N, H,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
