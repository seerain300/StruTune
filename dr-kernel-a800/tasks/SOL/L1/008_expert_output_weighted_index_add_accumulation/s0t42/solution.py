import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,          # *bf16 or *fp16, pointer to output [M, H]
    expert_ptr,          # *bf16 or *fp16, pointer to expert outputs [N, H]
    index_ptr,           # *int32, pointer to token indices [N]
    M,                   # int32, number of rows in output (M = batch_size * seq_len)
    H,                   # int32, number of columns in output
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Each program handles one source row i
    pid = tl.program_id(axis=0)
    # Load token index for this source row (int32)
    token = tl.load(index_ptr + pid)  # index into output rows

    # Vector of column offsets for this tile
    h_offsets = tl.arange(0, BLOCK_H)

    start = 0
    while start < H:
        offs = start + h_offsets
        mask = offs < H
        # Load expert_outputs[pid, offs]
        expert_row_ptr = expert_ptr + pid * H + offs
        val = tl.load(expert_row_ptr, mask=mask, other=0.0)
        # Compute destination address: output[token, offs]
        dest_base = output_ptr + (token * H)
        dest_ptr = dest_base + offs
        # Atomic add into output
        tl.atomic_add(dest_ptr, val, mask=mask)
        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        final_hidden_states: (M, H), accumulation buffer to be updated
        expert_outputs: (N, H), weighted expert outputs to scatter-add
        token_indices: (N,), indices into final_hidden_states' rows
        """
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA"
        output = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        M = output.shape[0]
        H = output.shape[1]
        N = expert_outputs.shape[0]

        # Triton prefers int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Choose BLOCK_H adaptively to reduce loop iterations and improve throughput
        if H >= 1024:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 2
        elif H >= 256:
            BLOCK_H = 256  # use 256 for better throughput; loop handles remainder
            num_warps = 8
            num_stages = 2
        elif H >= 128:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        # Grid: one program per source row
        grid = (N,)

        scatter_add_row_kernel[grid](
            output,
            expert_outputs,
            token_indices,
            M,
            H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
