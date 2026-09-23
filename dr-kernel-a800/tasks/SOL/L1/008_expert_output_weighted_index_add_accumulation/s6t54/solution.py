import torch
import triton
import triton.language as tl


@triton.jit
def linear_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    N,        # int: total number of elements to copy
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Extract shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        N = batch_seq_len * hidden_size

        # Allocate output and perform Triton copy (clone) of final_hidden_states
        output = torch.empty((batch_seq_len, hidden_size), dtype=final_hidden_states.dtype, device=final_hidden_states.device)

        # Choose launch config based on N
        if N >= 131072:
            BLOCK = 16384
            num_warps = 8
            num_stages = 4
        elif N >= 16384:
            BLOCK = 8192
            num_warps = 8
            num_stages = 4
        else:
            BLOCK = 4096
            num_warps = 4
            num_stages = 2

        grid = (triton.cdiv(N, BLOCK),)

        linear_copy_kernel[grid](
            final_hidden_states,  # src
            output,                # dst
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Perform PyTorch accumulation (index_add along dim=0)
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
