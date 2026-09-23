import torch
import triton
import triton.language as tl


@triton.jit
def index_add_dim0_rows_kernel(
    out_ptr,                # *bf16, shape [M, H]
    in_ptr,                 # *bf16, shape [N, H]
    idx_ptr,                # *int32, shape [N]
    M: tl.constexpr,        # int, out rows (batch_seq_len)
    N: tl.constexpr,        # int, number of expert outputs
    H: tl.constexpr,        # int, hidden size
    BLOCK: tl.constexpr,    # int, chunk size across hidden dimension
    CHUNKS: tl.constexpr,   # int, number of chunks processed per loop iteration
):
    # Each program handles one row i in [0, N)
    i = tl.program_id(0)
    if i >= N:
        return

    # Destination row index in output
    dst_idx = tl.load(idx_ptr + i)  # int32
    if dst_idx < 0 or dst_idx >= M:
        return

    # Base pointers for the i-th input row and destination row
    in_row_ptr = in_ptr + i * H
    out_row_base = out_ptr + dst_idx * H

    # Vectorized offsets for a chunk
    offs = tl.arange(0, BLOCK)

    # Process hidden dimension in steps of CHUNKS * BLOCK
    step = CHUNKS * BLOCK
    for start in range(0, H, step):
        # Unroll over CHUNKS chunks
        for k in range(CHUNKS):
            chunk_start = start + k * BLOCK
            cols = chunk_start + offs
            mask = cols < H

            # Load the chunk of the expert output row as a vector
            vals = tl.load(in_row_ptr + cols, mask=mask, other=0.0)

            # Vectorized atomic add: add 'vals' into output row at columns 'cols'
            out_ptrs = out_row_base + cols
            tl.atomic_add(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton kernel."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Inputs must be bfloat16."
        assert token_indices.dtype == torch.long, "token_indices must be torch.long."

        # Clone to preserve original initialization
        output = final_hidden_states.clone()

        # Ensure contiguity
        out = output.contiguous()
        in_t = expert_outputs.contiguous()
        idx = token_indices.to(torch.int32).contiguous()

        M = out.shape[0]
        N = in_t.shape[0]
        H = out.shape[1]

        # Adaptive tuning
        if H >= 1024:
            BLOCK = 256
            CHUNKS = 2
            num_warps = 8
            num_stages = 3
        elif H >= 512:
            BLOCK = 256
            CHUNKS = 2
            num_warps = 8
            num_stages = 2
        else:
            BLOCK = 128
            CHUNKS = 2
            num_warps = 4
            num_stages = 2

        # Launch one program per row i
        grid = (N,)
        index_add_dim0_rows_kernel[grid](
            out, in_t, idx,
            M, N, H,
            BLOCK, CHUNKS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
