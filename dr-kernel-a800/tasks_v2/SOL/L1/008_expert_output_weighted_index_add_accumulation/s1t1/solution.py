import torch
import triton
import triton.language as tl


@triton.jit
def atomic_add_by_index_scalar_kernel(
    output_ptr,        # *bfloat16, shape [M, H]
    source_ptr,        # *bfloat16, shape [N, H]
    index_ptr,         # *int32,    shape [N]
    N,                 # int32, total number of source rows
    M,                 # int32, total number of destination rows (batch_seq_len)
    H: tl.constexpr    # hidden_size (compile-time for loop)
):
    pid = tl.program_id(axis=0)  # program id: which source row to process
    if pid >= N:
        return

    # Load destination index for this source row
    dest = tl.load(index_ptr + pid)  # int32

    # Loop over hidden dimension and atomic add each element
    # Using a simple ascending j loop to minimize variability and match torch.index_add order.
    for j in range(0, H):
        # Load scalar from source row j (bfloat16)
        src_val = tl.load(source_ptr + pid * H + j)
        # Atomic add to destination row j
        tl.atomic_add(output_ptr + dest * H + j, src_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of the original run() function.
        Performs atomic accumulation: output[token_indices[i]] += expert_outputs[i] for all i.
        Returns the updated output tensor.
        """
        # Ensure CUDA device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Prepare output: same shape and dtype as final_hidden_states
        output = torch.empty_like(final_hidden_states)

        # Shapes
        M = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]  # hidden_size
        N = expert_outputs.shape[0]       # num_selected_tokens

        # Triton grid: one program per source row
        grid = (N,)

        # Cast indices to int3


def run(*args):
    return ModelNew()(*args)
