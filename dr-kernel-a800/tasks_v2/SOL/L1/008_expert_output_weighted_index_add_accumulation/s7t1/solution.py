import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_expert_kernel(
    out_ptr,          # *const bfloat16
    indices_ptr,      # *const int64
    expert_ptr,       # *const bfloat16
    num_rows,         # int32
    hidden_size,      # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE

    # Process BLOCK_SIZE rows per program; guard out-of-range
    for i in range(0, BLOCK_SIZE):
        row_idx = row_start + i
        in_range = row_idx < num_rows

        # Load destination token index as int64, cast to int32 for offset math
        index_val64 = tl.load(indices_ptr + row_idx, mask=in_range, other=0)
        index_val = index_val64.to(tl.int32)

        # Loop over hidden dimension elements; use int32 offsets
        for j in range(0, hidden_size):
            src_off = (row_idx * hidden_size + j)  # int32
            dst_off = (index_val * hidden_size + j)  # int32

            # Load source value (bfloat16)
            val = tl.load(expert_ptr + src_off, mask=in_range, other=0.0)
            # Atomic add into the destination row (bfloat16)
            tl.atomic_add(out_ptr + dst_off, val, mask=in_range)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We allocate a cloned output tensor and perform scatter-add with Triton atomics.
        """
        # Ensure inputs are on CUDA for Triton execution
        if final_hidden_states.device.type != "cuda":
            raise RuntimeError("ModelNew expects tensors on CUDA device for Triton execution.")
        if expert_outputs.device.type != "cuda" or token_indices.device.type != "cuda":
            raise RuntimeError("All inputs must be on CUDA device for Triton execution.")

        # Clone to match PyTorch behavior (index_add is performed on a clone)
        out = final_hidden_states.clone()

        # Ensure contiguous memory
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Sizes
        num_rows = expert_outputs.shape[0]  # number of expert outputs to add
        hidden_size = expert_outputs.shape[1]

        # Launch Triton kernel
        BLOCK_SIZE = 1024  # rows per program; can be tuned
        grid = (triton.cdiv(num_rows, BLOCK_SIZE),)

        scatter_add_expert_kernel[grid](
            out,                   # out_ptr
            token_indices,         # indices_ptr
            expert_outputs,        # expert_ptr
            num_rows,              # num_rows
            hidden_size,           # hidden_size
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,           # typical default
        )

        return out


def run(*args):
    return ModelNew()(*args)
