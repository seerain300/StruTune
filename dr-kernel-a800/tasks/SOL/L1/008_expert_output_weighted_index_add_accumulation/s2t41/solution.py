import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,        # *half, final_hidden_states (we will write into it)
    expert_ptr,     # *half, expert_outputs (N, H)
    indices_ptr,    # *int32, token_indices (N,)
    N,              # int32, number of selected tokens
    H,              # int32, hidden size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)
    # idx is expected to be in [0, N)

    # Iterate over hidden dimension in chunks of BLOCK
    for r in range(0, H, BLOCK):
        cols = r + tl.arange(0, BLOCK)
        mask = cols < H

        # Load the expert vector chunk for row pid
        expert_offs = pid * H + cols
        val = tl.load(expert_ptr + expert_offs, mask=mask, other=0.0)

        # Compute output offsets for destination row and atomically add
        out_offs = idx * H + cols
        tl.atomic_add(out_ptr + out_offs, val, mask=mask)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are contiguous and on CUDA; clone out to match reference semantics
        out = final_hidden_states.clone()

        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        assert out.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors"

        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Choose BLOCK as next power-of-two of H, capped to 256 to balance throughput
        BLOCK = min(_next_power_of_two(H), 256)
        # Warps selection tuned to BLOCK
        num_warps = 2 if BLOCK <= 64 else (4 if BLOCK <= 128 else 8)
        num_stages = 2

        # Launch: one program per selected token row
        grid = (N,)

        _index_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            N, H,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
