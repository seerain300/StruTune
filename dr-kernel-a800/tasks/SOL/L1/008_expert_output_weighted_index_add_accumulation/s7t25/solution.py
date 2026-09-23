import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_atomic_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H]
    src_ptr,        # *bf16, pointer to src tensor [M, H]
    indices_ptr,    # *int32, pointer to token indices [M]
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this contribution
    row_idx = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension; one atomic add per element
    for j in range(0, H):
        out_offset = row_idx * H + j
        src_offset = pid * H + j
        tl.atomic_add(out_ptr + out_offset, tl.load(src_ptr + src_offset))


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure all tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"

        # Make tensors contiguous and match dtypes
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous().to(torch.int32)

        # Clone to match the reference behavior (clone + index_add_)
        out = final_hidden_states.clone()

        N = out.shape[0]  # batch_seq_len
        M = expert_outputs.shape[0]  # num_selected_tokens
        H = out.shape[1]

        # Launch Triton kernel: one program per row
        grid = (N,)
        scatter_add_per_row_atomic_kernel[grid](out, expert_outputs, token_indices, N, H)

        return out


def run(*args):
    return ModelNew()(*args)
