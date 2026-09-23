import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H]
    src_ptr,        # *bf16, pointer to src tensor [M, H]
    indices_ptr,    # *int32, pointer to indices tensor [M]
    M,              # int32, number of source rows (M = num_selected_tokens)
    N,              # int32, number of rows in out (N = batch_seq_len * num_experts_per_tok //? Actually N=batch_seq_len; but run-time N is correct for final_hidden_states)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size for H, e.g., 256
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out for this source row
    dst = tl.load(indices_ptr + pid)
    if dst < 0 or dst >= N:
        return

    # Loop over hidden dimension in chunks
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute flat offsets into the [N, H] and [M, H] tensors
        out_offsets = dst * H + offs
        src_offsets = pid * H + offs

        # Load source values for this chunk (masked for tail)
        vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)  # bf16

        # Atomic add into out for this chunk
        tl.atomic_add(out_ptr + out_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        """
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device"
        out = final_hidden_states.clone()
        # Make sure inputs are contiguous
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton prefers int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        M = expert_outputs.shape[0]
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Launch grid: one program per source row
        BLOCK_SIZE = 256  # modest increase to reduce loop iterations over H
        grid = (M,)

        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,   # keep simple and reliable across workloads
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
