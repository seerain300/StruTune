import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H]
    src_ptr,        # *bf16, pointer to src tensor [M, H]
    indices_ptr,    # *int32, pointer to token indices [M]
    M,              # int32, number of source rows (num_selected_tokens)
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size for unrolling along H
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index for this source row
    dst = tl.load(indices_ptr + pid)
    if (dst < 0) or (dst >= N):
        return

    # Process hidden dimension in chunks
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Load source row slice
        src_row_start = pid * H
        src_vals = tl.load(src_ptr + src_row_start + offs, mask=mask, other=0.0)

        # Atomic add to destination row
        out_row_start = dst * H
        tl.atomic_add(out_ptr + out_row_start + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
        We perform atomic scatter-add in Triton.
        """
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA device"
        out = final_hidden_states.clone().contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Triton expects int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Dimensions
        M = expert_outputs.shape[0]   # number of selected tokens
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Choose a reasonable block size for unrolling; 128 or 256 works well.
        BLOCK_SIZE = 128

        # Launch one program per source row
        grid = (M,)
        scatter_add_chunked_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
