import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_2d_kernel(
    output_ptr,            # *bf16 or *fp16, shape [M, H], contiguous row-major
    expert_ptr,            # *bf16 or *fp16, shape [N, H], contiguous row-major
    index_ptr,             # *int32, shape [N]
    M,                     # int32, number of rows in output
    H,                     # int32, number of hidden features
    BLOCK_H: tl.constexpr  # tile size along hidden dimension
):
    # 2D launch: pid_n over source rows, pid_h over H tiles
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Compute H tile offsets
    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    # Load token index for this source row
    idx = tl.load(index_ptr + pid_n)
    # Destination row offset: idx * H
    dest_offsets = idx * H + h_offsets

    # Load expert outputs for this row and H-tile
    vals = tl.load(expert_ptr + pid_n * H + h_offsets, mask=mask, other=0.0)

    # Atomic add into output
    tl.atomic_add(output_ptr + dest_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be CUDA"
        # Ensure dtype compatibility
        assert final_hidden_states.dtype in (torch.bfloat16, torch.float16), "Unsupported dtype"
        assert expert_outputs.dtype == final_hidden_states.dtype, "Dtype mismatch between output and expert_outputs"

        # Contiguity for efficient Triton loads
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Triton prefers int32 indices; cast if needed
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M, H = final_hidden_states.shape
        N = expert_outputs.shape[0]

        # Output: clone to avoid modifying caller's buffer
        output = final_hidden_states.clone()

        # Kernel tuning: fixed tile size that performed well
        BLOCK_H = 128
        grid = (N, triton.cdiv(H, BLOCK_H))
        scatter_add_2d_kernel[grid](
            output, expert_outputs, token_indices, M, H,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
