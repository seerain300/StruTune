import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    output_ptr,           # *const float (final_hidden_states clone), shape (M, H), bfloat16
    expert_ptr,           # *const float, expert_outputs, shape (N, H), bfloat16
    index_ptr,            # *const int32, token_indices, shape (N,), int32
    N,                    # number of rows in expert_outputs (and number of updates), int32
    H,                    # hidden size, int32
    BLOCK_H: tl.constexpr # chunk size for H
):
    # Each program handles one source row (n) and loops over H in chunks
    n = tl.program_id(0)  # axis 0 over N
    if n >= N:
        return

    # Load target row index (int32)
    idx = tl.load(index_ptr + n)

    # Loop over H in chunks of BLOCK_H
    h = 0
    while h < H:
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load expert_outputs[n, h_offsets] (vectorized over H)
        vals = tl.load(expert_ptr + n * H + h_offsets, mask=mask, other=0.0)

        # Compute destination addresses: output[idx, h_offsets]
        dest = idx * H + h_offsets

        # Atomic add into output
        tl.atomic_add(output_ptr + dest, vals, mask=mask)

        h += BLOCK_H


def _choose_launch_params(H: int):
    # Adaptive tuning for better performance across hidden sizes
    if H >= 256:
        return 256, 8, 2  # BLOCK_H, num_warps, num_stages
    elif H >= 128:
        return 128, 4, 2
    else:
        return 64, 2, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-accelerated scatter-add:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        All computations are performed by Triton kernels.
        """
        # Ensure device and dtypes; inputs are bfloat16 in provided get_inputs
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtypes must be bfloat16"
        # Clone to match index_add semantics (no-op on clone then accumulation)
        output = final_hidden_states.clone()

        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Triton expects int32 for index arithmetic
        index_i32 = token_indices.to(torch.int32)

        # Choose kernel launch parameters
        BLOCK_H, num_warps, num_stages = _choose_launch_params(H)

        # Launch kernel: 1D grid over N
        grid = (N,)

        scatter_add_rows_kernel[grid](
            output, expert_outputs, index_i32,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
