import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_chunks_kernel(
    out_ptr,       # *bf16 or *fp16: output buffer [M, H], M = batch_seq_len
    expert_ptr,    # *bf16 or *fp16: expert outputs [N, H]
    indices_ptr,   # *int32: token indices [N]
    N,             # int32: number of selected tokens (rows in expert_outputs)
    H,             # int32: hidden size (columns)
    BLOCK: tl.constexpr,  # chunk size over hidden dimension
):
    # 2D launch: axis 0 = row (token), axis 1 = chunk along hidden dim
    row_id = tl.program_id(axis=0)
    chunk_id = tl.program_id(axis=1)

    if row_id >= N:
        return

    # Load destination row index for this selected token
    dest_row = tl.load(indices_ptr + row_id)

    # Compute start column for this chunk
    col_start = chunk_id * BLOCK
    cols = col_start + tl.arange(0, BLOCK)
    mask = cols < H

    # Load expert row chunk
    row_offset = row_id * H
    vals = tl.load(expert_ptr + row_offset + cols, mask=mask, other=0.0)

    # Compute output offsets and atomically add
    out_offsets = dest_row * H + cols
    tl.atomic_add(out_ptr + out_offsets, vals, mask=mask)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton kernel."

        # Clone output to match index_add semantics
        out = final_hidden_states.clone()

        # Ensure expert_outputs is contiguous
        expert_outputs = expert_outputs.contiguous()

        # Ensure token_indices are int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        N, H = expert_outputs.shape

        # Choose BLOCK as next power-of-two of H, capped at 256
        BLOCK = min(256, _next_power_of_two(H))

        # Grid: one program per row, and per hidden chunk
        grid = (N, (H + BLOCK - 1) // BLOCK)

        # Heuristic for num_warps based on BLOCK
        num_warps = 4 if BLOCK <= 128 else 8

        # Launch kernel
        _index_add_rows_chunks_kernel[grid](
            out, expert_outputs, token_indices, N, H, BLOCK=BLOCK,
            num_warps=num_warps, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
