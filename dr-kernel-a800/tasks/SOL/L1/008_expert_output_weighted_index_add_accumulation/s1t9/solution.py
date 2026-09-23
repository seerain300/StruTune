import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_vectorized(output_ptr, source_ptr, index_ptr, N, H, BLOCK_H: tl.constexpr):
    """
    Triton kernel to perform scatter-add along rows:
    For each source row i, add source[i, :] into output[index[i], :].
    We process the hidden dimension H in tiles of size BLOCK_H, issuing vectorized atomic adds.
    Grid: axis 0 = row index, axis 1 = tile index over H.
    """
    pid_row = tl.program_id(axis=0)  # which source row
    pid_tile = tl.program_id(axis=1) # which tile along hidden dimension

    if pid_row >= N:
        return

    # Compute start offset in hidden dimension for this tile
    start = pid_tile * BLOCK_H
    # Compute number of valid elements in this tile
    # Note: Triton doesn't have a direct modulo for runtime values; we keep H divisible by BLOCK_H by host choice.
    offs = start + tl.arange(0, BLOCK_H)
    mask = offs < H  # safety mask (usually fully true when H % BLOCK_H == 0)

    # Load destination index for this source row
    dest = tl.load(index_ptr + pid_row)  # int32

    # Compute base pointers for this row
    # source_ptr is a flat pointer; for row i, the j-th element is at i*H + j
    # We use vectorized load for the tile: load source[pid_row, offs]
    # Note: source_ptr is row-major 2D, but we pass it flattened by rows.
    src_offsets = pid_row * H + offs
    vals = tl.load(source_ptr + src_offsets, mask=mask, other=0.0)  # dtype follows source_ptr

    # Destination offsets: output[dest, offs]
    dest_offsets = dest * H + offs

    # Atomic add vectorized across the tile
    tl.atomic_add(output_ptr + dest_offsets, vals, mask=mask)


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

        # Choose a tile size for the hidden dimension. 128 works well and divides common H values.
        BLOCK_H = 128

        # Triton grid: one program per source row and per tile of H
        grid = (N, (H + BLOCK_H - 1) // BLOCK_H)

        # Cast indices to int32 for Triton
        index32 = token_indices.to(torch.int32)

        # Launch Triton kernel. Use a moderate number of warps; the kernel is simple and vectorized.
        scatter_add_rows_vectorized[grid](output, expert_outputs, index32, N, H, BLOCK_H, num_warps=4, num_stages=2)

        return output


def run(*args):
    return ModelNew()(*args)
