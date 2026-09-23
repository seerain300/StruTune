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

    # Destination row in out
    dst = tl.load(indices_ptr + pid)  # int32

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute pointers for out and src chunks
        out_row_ptr = out_ptr + dst * H
        src_row_ptr = src_ptr + pid * H

        # Load chunk from source row
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)

        # Atomic add into out row at positions offs
        tl.atomic_add(out_row_ptr + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        # Clone to match original behavior (out-of-place accumulation)
        out = final_hidden_states.clone()

        # Ensure tensors are on CUDA and contiguous
        assert out.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Shapes
        M = expert_outputs.shape[0]  # num_selected_tokens = batch_seq_len * num_experts_per_tok
        N = out.shape[0]             # batch_seq_len
        H = out.shape[1]             # hidden_size

        # Launch Triton kernel: one program per source row
        BLOCK_SIZE = 128  # moderate chunk size; adjust if needed
        grid = (M,)
        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,
            num_stages=1,
        )

        return out


def run(*args):
    return ModelNew()(*args)
