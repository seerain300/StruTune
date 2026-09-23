import torch
import triton
import triton.language as tl


@triton.jit
def _copy_scatter_add_kernel(
    output_ptr,      # *bf16, shape (B, H)
    final_ptr,       # *bf16, shape (B, H)
    expert_ptr,      # *bf16, shape (T, H)
    indices_ptr,     # *int64, shape (T,)
    B: tl.constexpr, # int
    H: tl.constexpr, # int
    T: tl.constexpr, # int
):
    # Phase 1: copy final_hidden_states into output (clone): output[b, h] = final[b, h]
    b = 0
    while b < B:
        row_off = b * H
        h = 0
        while h < H:
            val = tl.load(final_ptr + row_off + h)
            tl.store(output_ptr + row_off + h, val)
            h += 1
        b += 1

    # Phase 2: scatter-add: output[indices[i], h] += expert[i, h]
    i = 0
    while i < T:
        idx64 = tl.load(indices_ptr + i)
        idx = idx64.to(tl.int32)  # ensure 32-bit for pointer arithmetic
        h = 0
        while h < H:
            val = tl.load(expert_ptr + i * H + h)
            current = tl.load(output_ptr + idx * H + h)
            tl.store(output_ptr + idx * H + h, current + val)
            h += 1
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        The computation is performed entirely in Triton kernels.
        """
        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton."

        # Ensure dtype and contiguity
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, \
            "final_hidden_states and expert_outputs must be bfloat16."
        assert token_indices.dtype == torch.long, "token_indices must be long (int64)."

        final = final_hidden_states.contiguous()
        expert = expert_outputs.contiguous()
        indices = token_indices.contiguous()

        B, H = final.shape
        T = indices.shape[0]

        # Allocate output
        output = torch.empty((B, H), dtype=final.dtype, device=final.device)

        # Launch the Triton kernel. Use 1D grid; kernel iterates over all rows and columns.
        grid = (1,)
        _copy_scatter_add_kernel[grid](
            output, final, expert, indices,
            B=B, H=H, T=T,
            num_warps=1,
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
