import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half, output tensor (final_hidden_states clone) with shape (N_tokens, H)
    expert_ptr,      # *half, expert_outputs with shape (N, H)
    indices_ptr,     # *int32, token_indices with shape (N,)
    N,               # int32, number of selected tokens
    H,               # int32, hidden size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # Base pointers for this row
    # out row stride is H (contiguous), so row base is idx * H
    # expert row base is pid * H
    # We'll loop over hidden dimension in chunks of BLOCK and atomically add
    # out[idx, offs] += expert[pid, offs]

    # Loop over hidden dimension in BLOCKed chunks
    # Note: Triton requires compile-time known step for arange; we handle via a for-range over chunks
    num_chunks = (H + BLOCK - 1) // BLOCK
    for chunk in range(0, num_chunks):
        offs = chunk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < H

        # Load expert vector slice
        expert_row_ptr = expert_ptr + pid * H
        val = tl.load(expert_row_ptr + offs, mask=mask, other=0.0)

        # Compute destination addresses and atomically add
        out_row_ptr = out_ptr + idx * H
        tl.atomic_add(out_row_ptr + offs, val, mask=mask)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be on CUDA."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Clone to preserve original final_hidden_states
        output = final_hidden_states.clone()

        # Shapes
        N = expert_outputs.shape[0]  # number of selected tokens
        H = final_hidden_states.shape[1]  # hidden size

        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Choose BLOCK deterministically: next power-of-two of H, capped at 256
        # This minimizes chunks while keeping good performance.
        BLOCK = _next_power_of_two(H)
        BLOCK = min(BLOCK, 256)
        # Ensure a minimum of 64 for small H
        BLOCK = max(BLOCK, 64)

        # Choose num_warps based on BLOCK
        num_warps = 4 if BLOCK <= 128 else 8
        num_stages = 2

        # Launch one program per row
        grid = (N,)
        _index_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
