import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_kernel(
    out_ptr,          # *bf16, shape (N, H)
    src_ptr,          # *bf16, shape (M, H)
    indices_ptr,      # *int32, shape (M,)
    N: tl.constexpr,  # number of rows in out (batch_seq_len)
    H: tl.constexpr,  # hidden size
):
    # Each program handles one row i
    i = tl.program_id(0)
    if i >= N:
        return

    # Load token index for this row (int32)
    index = tl.load(indices_ptr + i)

    # Loop over hidden dimension, atomic add each element
    # We do scalar iteration to ensure robust compilation across envs.
    for j in range(0, H):
        src_val = tl.load(src_ptr + i * H + j)
        dst_ptr = out_ptr + index * H + j
        tl.atomic_add(dst_ptr, src_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Clone to match reference behavior
        out = final_hidden_states.clone()

        # Triton requires int32 for indices in this environment
        indices_i32 = token_indices.to(torch.int32)

        N = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]

        # Launch 1 program per row
        grid = (N,)
        scatter_add_rows_atomic_kernel[grid](out, expert_outputs, indices_i32, N, H, num_warps=1, num_stages=2)

        return out


def run(*args):
    return ModelNew()(*args)
