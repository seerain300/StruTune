import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    out_ptr,            # *bf16, shape (N, H)
    src_ptr,            # *bf16, shape (M, H)
    indices_ptr,        # *int32, shape (M,)
    N,                  # int32, number of rows in out (batch_seq_len)
    H,                  # int32, hidden size
    M,                  # int32, number of expert outputs
    BLOCK_SIZE: tl.constexpr,  # rows per program
):
    # 1D grid: each program handles BLOCK_SIZE rows
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    # Process up to BLOCK_SIZE rows in this program
    for i in range(BLOCK_SIZE):
        row = row_start + i
        if row >= N:
            break
        # Destination index for this row
        idx = tl.load(indices_ptr + row)  # int32 index in [0, N)
        # Iterate over hidden dimension and atomically add
        for j in range(0, H):
            src_offset = row * H + j
            dst_offset = idx * H + j
            val = tl.load(src_ptr + src_offset)
            tl.atomic_add(out_ptr + dst_offset, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        # Ensure all tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be CUDA tensors."
        # Clone to match reference behavior
        out = final_hidden_states.clone()
        # Ensure dtype and contiguity
        assert out.dtype == expert_outputs.dtype == torch.bfloat16, "Expected bfloat16 tensors"
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton prefers int32 indices
        indices_i32 = token_indices.to(torch.int32).contiguous()

        N = out.shape[0]
        H = out.shape[1]
        M = expert_outputs.shape[0]

        # Safe BLOCK_SIZE and grid to avoid int32 overflow in program_id arithmetic
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        # Launch Triton kernel
        scatter_add_atomic_kernel[grid](
            out, expert_outputs, indices_i32,
            N, H, M,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
