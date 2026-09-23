import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,        # *half, final_hidden_states clone (we will write into it)
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
    # idx is expected to be in [0, N). We assume token_indices is valid.

    # Process the hidden dimension in BLOCK-sized chunks
    num_chunks = tl.cdiv(H, BLOCK)
    for r in range(0, num_chunks):
        cols = r * BLOCK + tl.arange(0, BLOCK)
        mask = cols < H
        # Load expert vector for this row
        vals = tl.load(expert_ptr + pid * H + cols, mask=mask, other=0.0)
        # Compute destination addresses for this row
        dest_addrs = out_ptr + idx * H + cols
        # Atomic add into output
        tl.atomic_add(dest_addrs, vals, mask=mask)


def _next_power_of_two(n: int) -> int:
    # Returns the next power of two >= n, with n >= 1
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on the same device and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        # Clone final_hidden_states for accumulation
        out = final_hidden_states.clone()
        # Ensure dtype consistency (inputs are bfloat16 per get_inputs)
        out = out.to(expert_outputs.dtype)
        expert_outputs = expert_outputs.contiguous()
        # Token indices must be int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Choose BLOCK deterministically to balance performance and resource usage
        BLOCK = min(_next_power_of_two(H), 128)
        num_warps = 2 if BLOCK <= 64 else 4
        grid = (N,)

        _index_add_rows_kernel[grid](
            out, expert_outputs, token_indices, N, H, BLOCK,
            num_warps=num_warps, num_stages=2
        )
        return out


def run(*args):
    return ModelNew()(*args)
