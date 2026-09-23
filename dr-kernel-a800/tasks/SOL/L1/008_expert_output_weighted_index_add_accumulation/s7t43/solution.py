import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H], contiguous
    src_ptr,        # *bf16, pointer to src tensor [M, H], contiguous
    indices_ptr,    # *int32, pointer to indices tensor [M], contiguous
    M,              # int32, number of source rows
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return  # safety if indices are out of bounds (shouldn't happen with provided inputs)

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute flat offsets for out and src rows
        out_offsets = dst * H + offs
        src_row_off = pid * H + start

        # Load source chunk (masked for last partial chunk)
        src_vals = tl.load(
            src_ptr + src_row_off + tl.arange(0, BLOCK_SIZE),
            mask=mask,
            other=0.0,  # ensure masked lanes are zero
        )

        # Atomic add into output
        tl.atomic_add(out_ptr + out_offsets, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device"
        M = expert_outputs.shape[0]
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Triton kernel expects int32 indices
        indices_i32 = token_indices.to(torch.int32)

        # Make sure tensors are contiguous
        out = final_hidden_states.contiguous()
        src = expert_outputs.contiguous()

        # Launch Triton kernel: one program per source row
        BLOCK_SIZE = 512  # chunk size along hidden dimension
        grid = (M,)
        scatter_add_per_row_chunked_kernel[grid](
            out, src, indices_i32, M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
