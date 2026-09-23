import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16, shape [M, H], row-major contiguous
    expert_ptr,            # *bf16 or *fp16, shape [N, H], row-major contiguous
    index_ptr,             # *int32, shape [N]
    M: tl.int32,           # number of rows in output (M = batch_size * seq_len)
    H: tl.int32,           # hidden size
    BLOCK_H: tl.constexpr, # tile size along H
):
    pid = tl.program_id(axis=0)  # one program per source row
    # Load token index for this row (destination row in output)
    dest = tl.load(index_ptr + pid)  # int32

    # Iterate over H in chunks of BLOCK_H
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_H)  # [BLOCK_H] vector of column offsets
        mask = offs < H

        # Load expert_outputs[pid, offs] (row pid of expert_outputs)
        exp_ptrs = expert_ptr + pid * H + offs
        exp_vals = tl.load(exp_ptrs, mask=mask, other=0.0)

        # Compute output pointers: output is row-major contiguous => offset = dest * H + offs
        out_ptrs = output_ptr + dest * H + offs
        tl.atomic_add(out_ptrs, exp_vals, mask=mask)

        start += BLOCK_H


def _choose_launch_params(H: int):
    # Choose BLOCK_H, num_warps, num_stages based on H
    if H >= 4096:
        BLOCK_H = 512
        num_warps = 8
        num_stages = 4
    elif H >= 2048:
        BLOCK_H = 256
        num_warps = 8
        num_stages = 4
    elif H >= 512:
        BLOCK_H = 128
        num_warps = 4
        num_stages = 3
    else:
        BLOCK_H = 64
        num_warps = 2
        num_stages = 2
    return BLOCK_H, num_warps, num_stages


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be CUDA tensors"
        assert expert_outputs.shape == (token_indices.numel(), final_hidden_states.shape[1]), "expert_outputs shape must be (N, H)"
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = token_indices.numel()

        # Make sure expert_outputs is contiguous (row-major)
        expert_outputs = expert_outputs.contiguous()
        # Triton expects int32 indices; convert to int32 to avoid overhead
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        # Output buffer: we accumulate into final_hidden_states (provided as writeable)
        output = final_hidden_states

        # Launch Triton kernel: one program per source row (N)
        grid = (N,)
        BLOCK_H, num_warps, num_stages = _choose_launch_params(H)
        scatter_add_row_kernel[grid](
            output, expert_outputs, token_indices,
            M, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
