import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_vec_fp32(
    out_ptr,            # *float32, [M, H]
    token_indices_ptr,  # *int32,   [N]
    expert_ptr,         # *float32, [N, H]
    M: tl.int32,        # rows in out
    N: tl.int32,        # number of updates
    H: tl.int32,        # hidden size
    BLOCK_H: tl.constexpr,  # vector width across hidden dimension
):
    # One program per update i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Destination row index
    idx = tl.load(token_indices_ptr + i)  # int32

    # Base offsets
    # Process hidden dimension in chunks of BLOCK_H
    for col_start in range(0, H, BLOCK_H):
        offs = col_start + tl.arange(0, BLOCK_H)  # [BLOCK_H] vector of column indices
        mask = offs < H  # mask for tail

        # Load v chunk: expert_outputs[i, col_start:col_start+BLOCK_H]
        v = tl.load(expert_ptr + i * H + offs, mask=mask, other=0.0)  # float32 vector

        # Compute output row base pointer
        out_row_ptr = out_ptr + idx * H + offs

        # Atomic add the chunk into the output row
        tl.atomic_add(out_row_ptr, v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        We accumulate in float32 (to avoid bfloat16 atomic limitations) and cast back to bfloat16 at the end.
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be on CUDA for Triton."
        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]
        assert token_indices.shape[0] == N, "token_indices length must match num_selected_tokens."
        assert expert_outputs.shape[1] == H, "expert_outputs hidden dimension must match final_hidden_states."

        # Convert to fp32 for accumulation and clone
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Prepare inputs for Triton: ensure contiguous and correct dtypes
        token_indices_i32 = token_indices.to(torch.int32)
        expert_fp32 = expert_outputs.to(torch.float32)

        # Launch Triton kernel: one program per update, vectorize across hidden dimension
        grid = (N,)
        # Choose BLOCK_H as 128 (good default). Triton will unroll the loop over H.
        scatter_add_rows_atomic_vec_fp32[grid](
            out_fp32,
            token_indices_i32,
            expert_fp32,
            M, N, H,
            BLOCK_H=128,
            num_warps=4,  # moderate parallelism per program
        )

        # Cast back to bfloat16 to match original dtype
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
