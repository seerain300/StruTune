import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_kernel(
    output_ptr,            # *bf16 or *fp16, shape [M, H], row-major
    expert_ptr,            # *bf16 or *fp16, shape [N, H], row-major
    index_ptr,             # *int32, shape [N]
    M: tl.constexpr,       # number of output rows (M = batch_size * seq_len)
    H: tl.constexpr,       # hidden size
    stride_out_row: tl.constexpr,  # stride along rows for output (typically H)
    stride_exp_row: tl.constexpr,  # stride along rows for expert outputs (typically H)
    BLOCK_H: tl.constexpr,         # hidden tile size (e.g., 64/128/256)
):
    # Each program handles one source row (n in [0, N))
    n = tl.program_id(0)
    if n >= N:
        return

    # Load target output row index for this expert output
    idx = tl.load(index_ptr + n)  # int32

    # Iterate over hidden dimension in chunks of BLOCK_H
    start = 0
    while start < H:
        h_offsets = start + tl.arange(0, BLOCK_H)  # vector [0..BLOCK_H)
        mask = h_offsets < H

        # Load expert outputs for this row and chunk (row-major: n * H + h_offsets)
        expert_row_ptr = expert_ptr + n * stride_exp_row
        vals = tl.load(expert_row_ptr + h_offsets, mask=mask, other=0)

        # Compute output destinations: idx is the target row index
        dest_row_offset = idx * stride_out_row
        out_ptrs = output_ptr + dest_row_offset + h_offsets

        # Atomic add the chunk
        tl.atomic_add(out_ptrs, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
        output = final_hidden_states.clone()
        For i in range(N):
            output[token_indices[i]] += expert_outputs[i]
        """
        # Ensure dtype match (original uses bfloat16)
        assert expert_outputs.dtype == final_hidden_states.dtype, "Dtype mismatch between expert_outputs and output"

        # Clone to avoid modifying the input in-place
        output = final_hidden_states.clone()

        # Shapes
        M, H = output.shape
        N = expert_outputs.shape[0]

        # Ensure expert_outputs is contiguous row-major
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()

        # Triton prefers int32 for index arithmetic (safe since M,H up to 4096 in tests)
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per source row
        # Adaptive tiling based on H
        if H >= 2048:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 3
        elif H >= 512:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 3
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        grid = (N,)  # one program per source row

        # Strides: for row-major [M, H], stride along rows is H
        stride_out_row = H
        stride_exp_row = H

        scatter_add_row_kernel[grid](
            output,                    # output_ptr
            expert_outputs,            # expert_ptr
            token_indices,             # index_ptr
            M, H,                      # constexpr sizes
            stride_out_row,            # stride along rows for output
            stride_exp_row,            # stride along rows for expert outputs
            BLOCK_H,                   # tile size along H
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
