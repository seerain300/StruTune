import torch
import triton
import triton.language as tl


@triton.jit
def atomic_add_by_index_kernel(
    output_ptr,           # *bfloat16, shape [batch_seq_len, hidden_size]
    source_ptr,           # *bfloat16, shape [num_selected_tokens, hidden_size]
    index_ptr,            # *int32,    shape [num_selected_tokens]
    N,                    # int32, total number of source rows to process
    M,                    # int32, total number of destination rows (batch_seq_len)
    H: tl.constexpr,      # hidden_size, compile-time for loop unrolling
    BLOCK_H: tl.constexpr # tile size for hidden dimension
):
    pid = tl.program_id(axis=0)  # program id: which source row we process
    # Guard: if pid >= N, return (in case grid > N)
    if pid >= N:
        return

    # Load destination index for this source row
    dest = tl.load(index_ptr + pid)  # int32

    # Process the hidden dimension in tiles of size BLOCK_H
    for j in range(0, H, BLOCK_H):
        offs = j + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load source vector slice for this row
        src_ptr = source_ptr + pid * H + offs
        src_val = tl.load(src_ptr, mask=mask, other=0.0)

        # Compute output pointers for this destination row and hidden slice
        out_ptr = output_ptr + dest * H + offs

        # Atomic add each element
        tl.atomic_add(out_ptr, src_val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of the original run() function.
        Performs atomic accumulation: output[token_indices[i]] += expert_outputs[i] for all i.
        Returns the updated output tensor.
        """
        # Ensure CUDA device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Prepare output: same shape and dtype as final_hidden_states
        output = torch.empty_like(final_hidden_states)

        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        num_selected_tokens = expert_outputs.shape[0]

        # Triton grid: one program per source row
        grid = (num_selected_tokens,)

        # Cast indices to int32 for Triton
        index_i32 = token_indices.to(torch.int32)

        # Launch kernel. We'll use BLOCK_H=128 for good throughput on typical hidden sizes (e.g., 768).
        # Choose num_warps based on H; 4 or 8 is reasonable.
        BLOCK_H = 128
        num_warps = 4  # modest number of warps for small H; Triton will still be fine

        atomic_add_by_index_kernel[grid](
            output, expert_outputs, index_i32,
            num_selected_tokens, batch_seq_len,
            H=hidden_size, BLOCK_H=BLOCK_H,
            num_warps=num_warps
        )

        return output


def run(*args):
    return ModelNew()(*args)
