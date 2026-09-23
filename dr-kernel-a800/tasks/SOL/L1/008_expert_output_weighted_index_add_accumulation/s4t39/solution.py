import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_per_elem_kernel(
    output_ptr,         # *bf16, shape [B, H]
    expert_ptr,         # *bf16, shape [T, H]
    indices_ptr,        # *int64, shape [T]
    B: tl.constexpr,    # batch_seq_len (rows in output)
    H: tl.constexpr,    # hidden_size (columns)
    T: tl.constexpr,    # number of expert outputs
):
    # 2D grid: each program handles one source row i and one hidden column h
    i = tl.program_id(0)  # source row id
    h = tl.program_id(1)  # hidden column id

    # Bounds check (in case grid is larger than T or H)
    if (i < T) and (h < H):
        # Load token index (int64), cast to int32 for pointer arithmetic
        idx64 = tl.load(indices_ptr + i)
        idx = idx64.to(tl.int32)

        # Compute pointers
        # output[row = idx, col = h]
        out_ptr = output_ptr + idx * H + h
        # expert[row = i, col = h]
        expert_ptr_row = expert_ptr + i * H + h

        # Load values (bfloat16)
        out_val = tl.load(out_ptr)                # current output value
        expert_val = tl.load(expert_ptr_row)     # expert value at (i, h)

        # Ensure bf16 arithmetic explicitly
        out_val = out_val.to(tl.bfloat16)
        expert_val = expert_val.to(tl.bfloat16)

        result = out_val + expert_val
        tl.store(out_ptr, result)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We implement the index_add via a Triton kernel that performs deterministic per-element scatter-add.
        """
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        output = final_hidden_states.clone().contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch Triton kernel: one program per (i, h)
        grid = (T, H)

        # Use small num_warps since each program does minimal work
        scatter_add_rows_per_elem_kernel[grid](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,  # keep low to minimize overhead; correctness first
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
