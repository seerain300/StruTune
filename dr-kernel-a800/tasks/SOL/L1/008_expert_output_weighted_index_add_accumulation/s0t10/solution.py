import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 128, "num_warps": 8, "num_stages": 2}),
        triton.Config({"BLOCK_N": 64, "BLOCK_H": 64, "num_warps": 4, "num_stages": 2}),
        triton.Config({"BLOCK_N": 128, "BLOCK_H": 64, "num_warps": 8, "num_stages": 2}),
        triton.Config({"BLOCK_N": 32, "BLOCK_H": 128, "num_warps": 4, "num_stages": 2}),
        triton.Config({"BLOCK_N": 128, "BLOCK_H": 128, "num_warps": 8, "num_stages": 2}),
    ],
    key=["N", "H"],
)
@triton.jit
def scatter_add_blocks_kernel(
    output_ptr,          # *pointer* to output [M, H] in bfloat16
    expert_ptr,          # *pointer* to expert_outputs [N, H] in bfloat16
    indices_ptr,         # *pointer* to token_indices [N] int64
    N: tl.constexpr,     # number of source rows
    H: tl.constexpr,     # hidden size (columns)
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(axis=0)  # over N tiles
    pid_h = tl.program_id(axis=1)  # over H tiles

    # offsets for rows and hidden
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)              # [BLOCK_N]
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)              # [BLOCK_H]

    # masks
    mask_n = n_offs < N                                           # [BLOCK_N]
    mask_h = h_offs < H                                           # [BLOCK_H]

    # load token indices for these rows
    # indices are int64 in input; Triton will load them as 64-bit
    idx = tl.load(indices_ptr + n_offs, mask=mask_n, other=0)     # [BLOCK_N], int64
    # derive destination addresses: idx * H + h_offs
    # broadcasting: [BLOCK_N, BLOCK_H]
    dest_offsets = idx[:, None] * H + h_offs[None, :]             # [BLOCK_N, BLOCK_H]

    # load expert outputs for these rows and hidden offsets
    vals = tl.load(
        expert_ptr + n_offs[:, None] * H + h_offs[None, :],
        mask=mask_n[:, None] & mask_h[None, :],
        other=0.0,
    )  # [BLOCK_N, BLOCK_H], bfloat16

    # atomic add into output
    tl.atomic_add(output_ptr + dest_offsets, vals, mask=mask_n[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Perform output[row] += expert_outputs[i] for each i, where row = token_indices[i].
        Triton kernel does the scatter-add with block-wise atomics. Entry point expected to be 'forward'.
        """
        # Ensure dtype and device compatibility
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64"
        # Clone to preserve original buffer semantics (index_add starts from scratch)
        output = final_hidden_states.clone()

        # Shapes
        M, H = output.shape
        N = expert_outputs.shape[0]

        # Launch Triton kernel over tiles of N and H
        # Note: Triton grid depends on meta-parameters BLOCK_N, BLOCK_H provided via autotune
        def grid(meta):
            return (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(H, meta["BLOCK_H"]))

        scatter_add_blocks_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
        )

        return output


def run(*args):
    return ModelNew()(*args)
