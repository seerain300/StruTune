import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_2d_kernel(
    output_ptr,            # *bf16 or *fp16, shape [M, H]
    expert_ptr,            # *bf16 or *fp16, shape [N, H]
    index_ptr,             # *int32, shape [N]
    M,                     # number of rows in output (int)
    H,                     # hidden size (int)
    N,                     # number of expert outputs (int)
    BLOCK_N: tl.constexpr, # tile size along N
    BLOCK_H: tl.constexpr, # tile size along H
):
    # 2D grid: tiles along N and H
    pid_n = tl.program_id(0)  # tile id along N
    pid_h = tl.program_id(1)  # tile id along H

    # Offsets for this tile
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    n_mask = n_offsets < N
    h_mask = h_offsets < H

    # Load token indices for this N-tile (int32)
    idx = tl.load(index_ptr + n_offsets, mask=n_mask, other=0)  # [BLOCK_N]

    # Loop over H in chunks of BLOCK_H, perform 2D atomic adds
    start = 0
    while start < H:
        h_block = start + h_offsets  # [BLOCK_H]
        h_mask_block = h_block < H

        # Build 2D pointer grids:
        # Destination offsets: idx[:, None] * H + h_block[None, :]
        # Source offsets: n_offsets[:, None] * H + h_block[None, :]

        # Form 2D masks: rows valid and h_block valid
        mask_2d = n_mask[:, None] & h_mask_block[None, :]

        # Compute destination and source addresses
        dest_offsets = idx[:, None] * H + h_block[None, :]                 # [BLOCK_N, BLOCK_H]
        src_base = n_offsets[:, None] * H + (start + h_offsets[None, :])   # [BLOCK_N, BLOCK_H]

        # Load expert chunk for each row in the tile
        vals = tl.load(expert_ptr + src_base, mask=mask_2d, other=0.0)     # [BLOCK_N, BLOCK_H]

        # Atomic add into output
        tl.atomic_add(output_ptr + dest_offsets, vals, mask=mask_2d)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        assert final_hidden_states.dtype in (torch.bfloat16, torch.float16), "final_hidden_states must be bfloat16 or float16"
        assert expert_outputs.dtype == final_hidden_states.dtype, "expert_outputs must match final_hidden_states dtype"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64"

        # Convert indices to int32 for Triton
        indices = token_indices
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        # Shapes
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Make tensors contiguous
        output = final_hidden_states.contiguous()  # clone semantics
        expert = expert_outputs.contiguous()
        indices = indices.contiguous()

        # Choose tile sizes adaptively
        if H >= 512:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        BLOCK_N = 64
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

        scatter_add_2d_kernel[grid](
            output, expert, indices,
            M, H, N,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
