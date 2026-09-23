import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_vector_kernel(
    out_ptr,          # *bf16, pointer to out tensor [N, H]
    src_ptr,          # *bf16, pointer to src tensor [M, H]
    indices_ptr,      # *int32, pointer to token indices [M]
    N,                # int32, number of rows in out (batch_seq_len)
    H,                # int32, hidden size
    BLOCK_H: tl.constexpr,  # compile-time hidden size to vectorize over
):
    # 1D grid over rows: one program per row
    pid = tl.program_id(0)
    # Load destination row index for this row
    row_idx = tl.load(indices_ptr + pid)  # pid in [0, N), indices length equals N (M=N per provided code)

    # Vector of column offsets
    offs = tl.arange(0, BLOCK_H)

    # Compute source and destination pointers for this row
    src_row_ptrs = src_ptr + pid * H + offs
    dst_row_ptrs = out_ptr + row_idx * H + offs

    # Load source row slice and atomically add to destination row slice
    vals = tl.load(src_row_ptrs)
    tl.atomic_add(dst_row_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Clone to match reference behavior
        out = final_hidden_states.clone()

        # Triton kernel launch: 1D grid over rows
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        BLOCK_H = H  # vectorize across the full hidden dimension

        grid = (N,)

        scatter_add_atomic_vector_kernel[grid](
            out,
            expert_outputs,
            token_indices.to(torch.int32),
            N,
            H,
            BLOCK_H=BLOCK_H,
            num_warps=4,  # tuneable; 4–8 are typical
        )

        return out


def run(*args):
    return ModelNew()(*args)
