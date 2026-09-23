import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_kernel_1d(
    output_ptr,       # *bf16 (or fp16/fp32), output tensor base
    expert_ptr,       # *bf16, expert_outputs base
    indices_ptr,      # *int32, token_indices
    N,                # int32, number of rows in expert_outputs
    H: tl.constexpr,  # compile-time H for loop
):
    # Each program handles one source row n
    n = tl.program_id(0)

    # Load target row index (int32)
    idx = tl.load(indices_ptr + n)

    # Compute base pointers for this row
    # output row address: idx * H
    # expert row base: n * H
    for h in range(0, H):
        # Load value from expert_outputs[n, h]
        val = tl.load(expert_ptr + n * H + h)
        # Atomic add to output[idx, h]
        tl.atomic_add(output_ptr + idx * H + h, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based replacement for:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We initialize output to zeros and perform scatter-add with Triton kernels.
        """

        # Ensure device and dtype
        device = final_hidden_states.device
        dtype = final_hidden_states.dtype  # should be bfloat16

        # Create output initialized to zeros (index_add starts from zeros)
        output = torch.zeros(final_hidden_states.shape, dtype=dtype, device=device)

        # Number of selected tokens
        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Ensure token_indices is int32 for Triton addressing
        # (convert if needed)
        if token_indices.dtype != torch.int32:
            token_indices_i32 = token_indices.to(torch.int32)
        else:
            token_indices_i32 = token_indices

        # Launch Triton kernel: 1D grid over N
        grid = (N,)

        # Run kernel
        scatter_add_kernel_1d[grid](
            output, expert_outputs, token_indices_i32,
            N, H,
            num_warps=1,  # small per-program work; keep warps low to avoid overhead
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
