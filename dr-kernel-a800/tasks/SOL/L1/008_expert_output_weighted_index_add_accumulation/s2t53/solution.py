import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half (final_hidden_states clone), we will atomically add into it
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
    # Optional safety clamp if you anticipate out-of-range indices:
    # idx = tl.minimum(idx, H)

    # Iterate over hidden dimension in chunks of BLOCK
    # Each iteration: load a BLOCK-wide vector from expert row pid,
    # then atomically add into out[idx, cols]
    cols = tl.arange(0, BLOCK)
    # Loop over tiles
    for start in range(0, H, BLOCK):
        col = start + cols  # vector of column indices for this tile
        mask = col < H      # mask for tail
        # Compute linear offsets: out_row_base + col
        out_row_base = idx * H
        # Load expert tile
        vals = tl.load(expert_ptr + pid * H + col, mask=mask, other=0.0)
        # Atomic add into output
        tl.atomic_add(out_ptr + out_row_base + col, vals, mask=mask)

# Host-side launcher for ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match reference semantics (we are accumulating into a fresh buffer)
        out = final_hidden_states.clone()
        # Ensure contiguity and dtypes
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        device = out.device
        N = expert_outputs.shape[0]
        H = expert_outputs.shape[1]

        # Choose BLOCK and warps deterministically
        BLOCK = 256
        # If H is small, using 4 warps can be fine; for larger H, 8 warps helps occupancy.
        num_warps = 4 if H <= 128 else 8

        grid = (N,)
        _index_add_rows_kernel[grid](
            out, expert_outputs, token_indices, N, H,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
