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
    CHUNKS: tl.constexpr,  # number of BLOCK-sized chunks to cover H
    BLOCK: tl.constexpr,   # chunk size (32/64/128/256)
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # Loop over hidden dimension in fixed chunks of size BLOCK
    for chunk in range(CHUNKS):
        start = chunk * BLOCK
        cols = start + tl.arange(0, BLOCK)
        mask = cols < H

        # Load the hidden vector chunk from expert_outputs[pid, :]
        src_ptrs = expert_ptr + pid * H + cols
        vals = tl.load(src_ptrs, mask=mask, other=0.0)  # bfloat16

        # Compute destination pointers for out[idx, cols]
        dst_ptrs = out_ptr + idx * H + cols

        # Atomically add the chunk into the destination row
        tl.atomic_add(dst_ptrs, vals, mask=mask)


def _choose_block(H: int) -> int:
    # Choose BLOCK as a power-of-two near H to minimize iterations while keeping good throughput.
    # Use 32 for very small H, then 64, 128, else 256.
    if H <= 32:
        return 32
    elif H <= 64:
        return 64
    elif H <= 128:
        return 128
    else:
        return 256


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        # Ensure device/dtype consistency
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton kernel."

        # Clone the output buffer to mirror index_add_ semantics
        out = final_hidden_states.clone()

        # Ensure contiguous and dtypes
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous().to(torch.int32)

        N, H = expert_outputs.shape

        # Choose BLOCK deterministically based on H
        BLOCK = _choose_block(H)
        CHUNKS = (H + BLOCK - 1) // BLOCK  # ceil_div

        # Warps tuning based on BLOCK
        if BLOCK <= 64:
            num_warps = 4
        elif BLOCK <= 128:
            num_warps = 4
        else:
            num_warps = 8

        # Launch grid: one program per selected token row
        grid = (N,)

        _index_add_rows_kernel[grid](
            out, expert_outputs, token_indices, N, H,
            CHUNKS=CHUNKS, BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
