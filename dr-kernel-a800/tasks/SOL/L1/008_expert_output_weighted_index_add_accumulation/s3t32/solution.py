import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_vec_kernel(
    out_ptr,             # *fp32, shape [M, H], row-major
    token_indices_ptr,   # *i32, shape [N]
    expert_outputs_ptr,  # *fp32, shape [N, H]
    N: tl.constexpr,     # number of updates
    H: tl.constexpr,     # hidden size
    BLOCK_H: tl.constexpr,  # vector chunk size across hidden dim
):
    # One program per update i
    i = tl.program_id(0)
    # Load index (token position) for this update
    idx = tl.load(token_indices_ptr + i)

    # Iterate over hidden dimension in chunks of BLOCK_H
    # We use a range to ensure compile-time unrolling for performance
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < H

        # Load the source chunk for this update
        v = tl.load(expert_outputs_ptr + i * H + h_offsets, mask=mask, other=0.0)

        # Compute output pointer for this row and chunk
        out_row_ptr = out_ptr + idx * H + h_offsets

        # Atomically accumulate into output
        tl.atomic_add(out_row_ptr, v, mask=mask)


def _choose_block_h(H: int):
    # Heuristics for chunk size across hidden dimension
    if H >= 1024:
        return 1024, 8
    elif H >= 512:
        return 512, 4
    else:
        return 256, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on the same device and contiguous
        device = final_hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"
        # Clone and accumulate in fp32 for atomic_add support
        out_fp32 = final_hidden_states.clone().to(torch.float32)

        # Make sure inputs are contiguous
        out_fp32 = out_fp32.contiguous()
        expert_outputs_fp32 = expert_outputs.contiguous().to(torch.float32)
        token_indices_i32 = token_indices.contiguous().to(torch.int32)

        M = out_fp32.shape[0]
        N = expert_outputs_fp32.shape[0]
        H = out_fp32.shape[1]

        BLOCK_H, num_warps = _choose_block_h(H)

        # Launch one program per update
        grid = (N,)

        scatter_add_row_vec_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs_fp32,
            N=N,
            H=H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match the original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
