import torch
import triton
import triton.language as tl


@triton.jit
def _atomically_add_experts_kernel(
    out_ptr,        # *half, output tensor to accumulate into (created as clone of final_hidden_states)
    expert_ptr,     # *half, expert_outputs [N, H]
    indices_ptr,    # *int32, token_indices [N]
    N,              # int32: number of selected tokens
    H,              # int32: hidden size
    row_stride,     # int32: stride for rows in out_ptr (for [N, H], typically H)
    col_stride,     # int32: stride for cols in out_ptr (typically 1)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension in chunks of BLOCK
    offs = tl.arange(0, BLOCK)
    col = 0
    while col < H:
        cols = col + offs
        mask = cols < H

        # Load the expert vector slice for this row
        expert_row_base = pid * H
        expert_vals = tl.load(expert_ptr + expert_row_base + cols, mask=mask, other=0.0)

        # Compute output addresses using 2D strides: out_ptr + idx * row_stride + cols * col_stride
        out_addrs = idx * row_stride + cols * col_stride

        # Atomic add the slice into the output row
        tl.atomic_add(out_ptr + out_addrs, expert_vals, mask=mask)

        col += BLOCK


def _next_power_of_two(n: int, max_val: int = 256) -> int:
    # Compute next power-of-two for n, capped at max_val
    if n <= 1:
        return 1
    p = 1
    while p < n and p < max_val:
        p <<= 1
    return min(p, max_val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triton-only implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        All accumulation is performed by the Triton kernel; torch is only used for device/dtype checks and allocation of the initial output buffer.
        """

        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton execution."

        # Ensure dtype and contiguity
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "Dtypes must be bfloat16."
        assert token_indices.dtype == torch.int32, "token_indices must be int32."
        # Make sure inputs are contiguous (typically already true from the benchmark setup)
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Allocate output as a clone of final_hidden_states (this initializes output to zeros)
        output = final_hidden_states.clone()

        # Shapes
        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # For a contiguous [N, H] tensor, row_stride = H, col_stride = 1
        row_stride = H
        col_stride = 1

        # Choose BLOCK size and launch parameters
        BLOCK = _next_power_of_two(H, max_val=256)
        num_warps = 4 if BLOCK <= 128 else 8
        num_stages = 2

        # Launch one program per selected token row
        grid = (N,)
        _atomically_add_experts_kernel[grid](
            output,                # out_ptr
            expert_outputs,        # expert_ptr
            token_indices,         # indices_ptr
            N,
            H,
            row_stride,
            col_stride,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
