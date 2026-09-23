import torch
import triton
import triton.language as tl


# Triton kernel: per-update scatter-add without atomics.
# One program handles one expert contribution (row), adds it to the corresponding output row.
@triton.jit
def scatter_add_row_kernel(
    out_ptr,            # *float32, [M, H] where M = batch_seq_len
    expert_ptr,         # *float32, [N, H] where N = num_selected_tokens
    indices_ptr,        # *int32,   [N]
    M: tl.int32,        # int: number of rows in output (batch_seq_len)
    N: tl.int32,        # int: number of updates
    H: tl.int32,        # int: hidden_size
    BLOCK_H: tl.constexpr,  # tile size for hidden dimension (e.g., 128 or 256)
):
    i = tl.program_id(0)  # which update
    # Guard in case grid > N (shouldn't happen if we launch exactly N programs)
    if i >= N:
        return

    # Load index of the destination row
    idx = tl.load(indices_ptr + i)  # int32

    # Base pointers for this update
    # Each row has H elements; we process columns in tiles of BLOCK_H
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load v from expert_outputs[i, offs]
        v = tl.load(expert_ptr + i * H + offs, mask=mask, other=0.0)  # float32 vector

        # Load out_row[idx, offs]
        out_row = tl.load(out_ptr + idx * H + offs, mask=mask, other=0.0)  # float32 vector

        # Accumulate
        out_row += v

        # Store back
        tl.store(out_ptr + idx * H + offs, out_row, mask=mask)

        col += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add without atomics:
        out = final_hidden_states.clone()
        out[token_indices[i]] += expert_outputs[i] for all i
        Result is returned as bfloat16 to match original.
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton execution."

        # Ensure contiguity
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        M = final_hidden_states.shape[0]           # batch_seq_len
        N = expert_outputs.shape[0]                # num_selected_tokens
        H = final_hidden_states.shape[1]           # hidden_size

        # Prepare float32 accumulation buffer initialized with final_hidden_states
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Upcast expert outputs to float32 (for accumulation)
        expert_fp32 = expert_outputs.to(torch.float32)

        # Convert indices to int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per update
        BLOCK_H = 128  # tuneable tile size; 128 works well for common hidden sizes
        grid = (N,)

        scatter_add_row_kernel[grid](
            out_fp32,               # out_ptr
            expert_fp32,            # expert_ptr
            indices_i32,            # indices_ptr
            M, N, H,
            BLOCK_H=BLOCK_H,
            num_warps=4,            # modest parallelism; safe across Triton versions
            num_stages=2,           # pipeline stages
        )

        # Cast back to bfloat16 to match original output dtype
        result = out_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
