import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half (final_hidden_states clone, we will write into it)
    expert_ptr,      # *half (expert_outputs)
    indices_ptr,     # *int32 (token_indices)
    N,               # int32: number of selected tokens
    H,               # int32: hidden size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load destination row index for this selected token
    idx = tl.load(indices_ptr + pid)
    # idx is expected to be in [0, N). We assume token_indices is valid per the workload setup.

    # Base offsets for this destination row and the source row pid
    base_out = idx * H
    base_expert = pid * H

    # Iterate over hidden dimension in chunks of BLOCK
    # We use a static loop with a runtime condition to keep Triton happy.
    r = 0
    while r < H:
        offs = r + tl.arange(0, BLOCK)
        mask = offs < H

        # Load expert slice (vector) for this row
        vals = tl.load(expert_ptr + base_expert + offs, mask=mask, other=0.0)

        # Atomically add into output row
        tl.atomic_add(out_ptr + base_out + offs, vals, mask=mask)

        r += BLOCK


def _next_power_of_two(n: int) -> int:
    # Returns the next power of two >= n, minimum 1
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure contiguity and dtypes
        out = final_hidden_states.clone().contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton prefers int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Choose BLOCK as next power-of-two of H, capped to 256 for good performance
        BLOCK = _next_power_of_two(H)
        if BLOCK > 256:
            BLOCK = 256  # cap to avoid overly large vectors per program

        # Launch one program per row
        grid = (N,)
        _index_add_rows_kernel[grid](
            out, expert_outputs, token_indices,
            N, H,
            BLOCK=BLOCK,
            num_warps=4,   # tuned for throughput on common GPUs
            num_stages=2,  # some pipelining
        )

        return out


def run(*args):
    return ModelNew()(*args)
