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

    # Destination row in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return

    # Process hidden dimension in chunks of BLOCK_SIZE
    # Masking handles cases where H is not divisible by BLOCK_SIZE.
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # out_ptr is [N, H] contiguous; row stride is H
        out_row_ptr = out_ptr + dst * H
        src_row_ptr = src_ptr + pid * H

        # Load source values with mask (dtype bfloat16)
        src_vals = tl.load(src_row_ptr + offs, mask=mask, other=tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16))

        # Atomic add into the destination row
        tl.atomic_add(out_row_ptr + offs, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized atomic scatter-add of expert_outputs into final_hidden_states at positions token_indices.

        Args:
            final_hidden_states: [batch_seq_len, hidden_size], bfloat16
            expert_outputs:       [num_selected_tokens, hidden_size], bfloat16
            token_indices:        [num_selected_tokens], int64 or int32

        Returns:
            Updated final_hidden_states with contributions from expert_outputs.
        """
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA device"
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Triton works best with int32 indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Output buffer: clone to match original semantics and avoid modifying input
        out = final_hidden_states.clone()

        M = expert_outputs.shape[0]
        N = out.shape[0]
        H = out.shape[1]

        # Launch Triton kernel
        BLOCK_SIZE = 256  # larger chunk reduces loop iterations
        grid = (M,)
        scatter_add_per_row_chunked_kernel[grid](
            out, expert_outputs, token_indices,
            M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
