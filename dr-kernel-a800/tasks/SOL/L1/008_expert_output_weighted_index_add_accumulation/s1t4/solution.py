import torch
import triton
import triton.language as tl


@triton.jit
def index_add_dim0_atomic_kernel(
    output_ptr,            # *bf16, pointer to output tensor [M, H]
    expert_outputs_ptr,    # *bf16, pointer to source tensor [N, H]
    index_ptr,             # *int32, pointer to token_indices [N]
    M: tl.constexpr,       # int32, number of rows in output (batch_seq_len)
    N: tl.constexpr,       # int32, number of source rows
    H: tl.constexpr,       # int32, hidden_size
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this source row
    dest = tl.load(index_ptr + pid)  # int32

    # Loop over hidden dimension and atomically add each element
    # Using a simple ascending j loop for clarity and correctness
    for j in range(0, H):
        src_val = tl.load(expert_outputs_ptr + pid * H + j)  # bf16 scalar
        # Atomic add to the corresponding output position
        tl.atomic_add(output_ptr + dest * H + j, src_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of the original run:
        - Clone final_hidden_states to preserve it.
        - Allocate output and perform index-add via a Triton kernel along dim=0.
        Returns the updated output tensor (same semantics as the original).
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."

        # Preserve the original buffer (not modified in-place)
        preserve = final_hidden_states.clone()

        # Prepare output: we will mutate this tensor to match index_add behavior
        # We keep dtype and shape identical.
        output = preserve.clone()

        # Shapes
        M = preserve.shape[0]  # batch_seq_len
        H = preserve.shape[1]  # hidden_size
        N = expert_outputs.shape[0]  # num_selected_tokens

        # Triton grid: one program per source row
        grid = (N,)

        # Cast token_indices to int32 for Triton pointer arithmetic
        index_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel
        index_add_dim0_atomic_kernel[grid](
            output,            # output_ptr
            expert_outputs,    # expert_outputs_ptr
            index_i32,         # index_ptr
            M=M, N=N, H=H,
        )

        return output


def run(*args):
    return ModelNew()(*args)
