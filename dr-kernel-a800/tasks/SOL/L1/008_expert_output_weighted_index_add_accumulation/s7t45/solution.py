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
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time constant)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return  # safety check (inputs should be valid)

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute base offsets for the row
        out_row_base = dst * H
        src_row_base = pid * H

        # Load a chunk of src row and atomic add into out row
        vals = tl.load(src_ptr + src_row_base + offs, mask=mask, other=0.0)
        tl.atomic_add(out_ptr + out_row_base + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Triton implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We allocate output as an empty tensor and perform atomic adds using Triton.
        """
        # Allocate output with the same shape and dtype as final_hidden_states
        out = torch.empty_like(final_hidden_states)

        # Ensure contiguity and dtypes
        src = expert_outputs  # shape [M, H]
        indices = token_indices.to(torch.int32)  # shape [M]

        # Shapes
        M = src.shape[0]
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        # Launch Triton kernel: one program per source row
        BLOCK_SIZE = 128  # balanced default; Triton will specialize per BLOCK_SIZE
        grid = (M,)

        scatter_add_per_row_chunked_kernel[grid](
            out,
            src,
            indices,
            M,
            N,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
