import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_fp32_tiled_kernel(
    out_ptr,          # *fp32, shape [M, H] output buffer (already initialized)
    expert_ptr,       # *fp32, shape [N, H] expert outputs
    idx_ptr,          # *int32, shape [N] token indices
    M,                # int32, number of rows (batch_seq_len)
    N,                # int32, number of updates (num_selected_tokens)
    H,                # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # tile size along hidden dimension
):
    # program ids: one per (row, tile)
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # offsets for this tile along hidden dimension
    start = tile_id * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask_vec = offsets < H  # mask for bounds

    # base linear offsets for this row and tile
    row_base_out = row_id * H + offsets

    # initialize local accumulator for this chunk
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # loop over all updates and accumulate
    i = 0
    while i < N:
        idx = tl.load(idx_ptr + i)  # int32 scalar
        # check validity (though idx is guaranteed in [0, M) by construction)
        valid = (idx >= 0) & (idx < M)
        if valid:
            exp_base = i * H + offsets
            v = tl.load(expert_ptr + exp_base, mask=mask_vec, other=0.0)
            acc += v
        i += 1

    # atomically add the accumulated chunk to the output row
    tl.atomic_add(out_ptr + row_base_out, acc, mask=mask_vec)


def _next_power_of_two(x: int) -> int:
    # choose a good BLOCK_SIZE for vectorization
    if x <= 64:
        return 64
    elif x <= 128:
        return 128
    elif x <= 256:
        return 256
    elif x <= 512:
        return 512
    else:
        return 1024  # cap at 1024 for reasonable tile size


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA for Triton execution."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        M = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Accumulate in float32 for correctness (bf16 doesn't support atomic_add)
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Cast expert outputs to float32
        expert_fp32 = expert_outputs.to(torch.float32)

        # Triton kernel launch configuration
        BLOCK_SIZE = _next_power_of_two(H)
        grid = (M, (H + BLOCK_SIZE - 1) // BLOCK_SIZE)

        # Cast indices to int32 for Triton
        idx_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel
        scatter_add_rows_fp32_tiled_kernel[grid](
            out_fp32, expert_fp32, idx_i32,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,   # tuneable; 4 is a good default
            num_stages=2,
        )

        # Cast back to bfloat16 to match original behavior
        result = out_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
