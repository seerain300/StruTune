import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H], used as output (we will write into it)
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    M: tl.constexpr,  # total rows in out_ptr/src_ptr (M = batch_size * seq_len)
    H: tl.constexpr,  # hidden size
    N: tl.constexpr,  # number of source rows to scatter (N = M * num_experts_per_tok)
    BLOCK_H: tl.constexpr,  # tile size for H (e.g., 256)
):
    # One program per source row
    row = tl.program_id(0)  # row in [0, N)
    if row >= N:
        return

    # Load the destination row index for this source row
    idx = tl.load(indices_ptr + row)
    if idx < 0 or idx >= M:
        # Defensive guard; kernel assumes valid indices (grid=N and provided indices should be valid)
        return

    # Iterate over the hidden dimension in tiles of BLOCK_H
    # For typical H <= 1024, this is a single iteration with BLOCK_H=256.
    for start in range(0, H, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Compute base offsets for out and src rows
        out_base = idx * H + offs
        src_base = row * H + offs

        # Load the source vector (masked for tail)
        src_vals = tl.load(src_ptr + src_base, mask=mask, other=0.0)

        # Atomically add into the output row vector
        tl.atomic_add(out_ptr + out_base, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        # Clone to create the output buffer
        output = final_hidden_states.clone()

        # Shapes
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs second dimension must match hidden size"
        assert token_indices.shape[0] == N, "token_indices length must equal N"

        # Ensure contiguous tensors
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Grid: one program per source row
        grid = (N,)

        # Launch kernel with a proven configuration
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            M=M, H=H, N=N,
            BLOCK_H=256,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
