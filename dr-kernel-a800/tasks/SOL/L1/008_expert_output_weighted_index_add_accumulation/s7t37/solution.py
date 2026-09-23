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

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)          # [BLOCK_SIZE] column indices
        mask = offs < H                                   # boolean mask for tail handling

        # Compute element offsets within each row
        src_row_offs = pid * H + offs                    # [BLOCK_SIZE] offsets into src
        out_row_offs = dst * H + offs                   # [BLOCK_SIZE] offsets into out

        # Load the chunk of expert_outputs for this row with mask; masked elements contribute zero
        val = tl.load(src_ptr + src_row_offs, mask=mask, other=0.0)

        # Atomic add into out at destination row; masked for tail
        tl.atomic_add(out_ptr + out_row_offs, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized atomic scatter-add:
          out[token_indices[i]] += expert_outputs[i] for all i
        Ensures Triton kernel is used; does not use PyTorch index_add.
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA for Triton execution."
        out = final_hidden_states.clone()  # preserve original semantics

        # Ensure contiguity and dtype for indices
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        # Shapes
        M = expert_outputs.shape[0]  # number of selected tokens
        N = final_hidden_states.shape[0]  # batch_seq_len
        H = final_hidden_states.shape[1]  # hidden size

        # Launch Triton kernel: one program per source row
        BLOCK_SIZE = 128  # moderate chunk size; works across hidden sizes
        grid = (M,)
        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,  # simple parallelism per program
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
