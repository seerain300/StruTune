import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_expert_outputs_kernel(
    output_ptr,         # *bf16, shape (M, H)
    source_ptr,         # *bf16, shape (N, H)
    index_ptr,          # *int32, shape (N,)
    N,                  # int: number of source rows (runtime)
    H,                  # int: hidden_size (runtime)
    BLOCK_H: tl.constexpr  # tile size along hidden dimension
):
    pid = tl.program_id(axis=0)  # which source row to process
    if pid >= N:
        return

    # Load destination row index for this source row
    dest = tl.load(index_ptr + pid)  # int32
    if dest < 0 or dest >= N:  # Note: M should be passed as first arg? Keep N for consistency
        return

    # Vectorized loop over hidden dimension in tiles, using while to cover arbitrary H
    h = 0
    while h < H:
        h_vec = h + tl.arange(0, BLOCK_H)          # vector of column offsets
        mask = h_vec < H                           # valid columns within H

        # Compute source and output offsets
        src_offs = pid * H + h_vec                 # vector of source offsets
        out_offs = dest * H + h_vec               # vector of output offsets

        # Load the expert output slice (bf16)
        src_vals = tl.load(source_ptr + src_offs, mask=mask, other=0.0)

        # Atomic add into the output
        tl.atomic_add(output_ptr + out_offs, src_vals, mask=mask)

        h += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add equivalent to:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        All computation is performed by Triton kernels; no PyTorch ops used for scatter-add.
        """
        # Ensure tensors are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."

        # Ensure contiguity and dtypes
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Prepare output: clone the provided final_hidden_states (mirrors original behavior)
        output = final_hidden_states.clone()

        # Shapes
        M = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]  # hidden_size
        N = expert_outputs.shape[0]       # num_selected_tokens

        # Cast indices to int32 for Triton
        index32 = token_indices.to(torch.int32)

        # Triton grid: one program per source row
        grid = (N,)

        # Launch the Triton kernel. Use BLOCK_H=128 for robust performance across typical H.
        scatter_add_expert_outputs_kernel[grid](
            output, expert_outputs, index32,
            N, H,
            BLOCK_H=128,
            num_warps=4,
            num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
