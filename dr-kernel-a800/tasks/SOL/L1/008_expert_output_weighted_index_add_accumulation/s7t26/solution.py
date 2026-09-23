import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_kernel(
    out_ptr,        # *bf16, shape (N, H), contiguous
    src_ptr,        # *bf16, shape (M, H), contiguous
    indices_ptr,    # *int32, shape (M,), contiguous
    M,              # int32, number of expert outputs to scatter (batch_size * seq_len * num_experts_per_tok)
    N,              # int32, number of rows in out (batch_size * seq_len)
    H,              # int32, hidden size
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Load destination token index for this row
    dst = tl.load(indices_ptr + pid)  # int32

    # Loop over hidden dimension and do atomic add
    # We avoid any vectorized pointer arithmetic to stay robust.
    for j in range(0, H):
        out_offset = dst * H + j
        src_offset = pid * H + j
        # Atomic add to handle duplicates
        tl.atomic_add(out_ptr + out_offset, tl.load(src_ptr + src_offset))


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match reference behavior
        out = final_hidden_states.clone()

        # Ensure tensors are on CUDA and contiguous
        assert out.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Compute shapes
        M = expert_outputs.shape[0]  # number of rows to scatter
        N = final_hidden_states.shape[0]  # number of rows in output buffer

        # Launch Triton kernel: one program per row i
        grid = (M,)
        scatter_add_per_row_kernel[grid](
            out,                  # out_ptr
            expert_outputs,       # src_ptr
            token_indices,        # indices_ptr
            M, N, expert_outputs.shape[1],  # pass M, N, H
        )
        return out


def run(*args):
    return ModelNew()(*args)
