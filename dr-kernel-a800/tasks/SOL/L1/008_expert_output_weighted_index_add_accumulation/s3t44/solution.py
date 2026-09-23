import torch
import triton
import triton.language as tl

@triton.jit
def _scatter_add_chunks_kernel(
    out_ptr,            # *fp32, shape [batch_seq_len, H]
    token_indices_ptr,  # *int32, shape [N]
    expert_outputs_ptr, # *fp32, shape [N, H]
    N,                  # int: number of updates
    H,                  # int: hidden size
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one update
    if pid >= N:
        return

    # Load token index (row in output)
    idx = tl.load(token_indices_ptr + pid)

    # Iterate over hidden dimension in chunks of BLOCK_H
    h = 0
    while h < H:
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load source vector for this update (fp32)
        v = tl.load(expert_outputs_ptr + pid * H + offs, mask=mask, other=0.0)

        # Compute output pointer for this row and chunk
        out_row_ptr = out_ptr + idx * H
        tl.atomic_add(out_row_ptr + offs, v, mask=mask)

        h += BLOCK_H


def _run_triton_scatter_add(final_hidden_states: torch.Tensor,
                            expert_outputs: torch.Tensor,
                            token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized scatter-add:
      out = final_hidden_states.clone()
      out.index_add_(dim=0, index=token_indices, source=expert_outputs)
    Returns updated out (bfloat16).
    """
    # Ensure tensors are on CUDA
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

    # Make inputs contiguous
    final_hidden_states = final_hidden_states.contiguous()
    expert_outputs = expert_outputs.contiguous()
    token_indices = token_indices.contiguous()

    # Triton requires int32 indices
    token_indices_i32 = token_indices.to(torch.int32)

    # Prepare output buffer in float32 for atomic accumulation
    out_fp32 = final_hidden_states.to(torch.float32).clone()

    # Cast expert outputs to float32 for accumulation
    expert_outputs_fp32 = expert_outputs.to(torch.float32)

    N = expert_outputs_fp32.shape[0]
    H = expert_outputs_fp32.shape[1]

    # Heuristic for BLOCK_H and num_warps based on hidden size
    if H >= 1024:
        BLOCK_H = 1024
        num_warps = 8
    elif H >= 512:
        BLOCK_H = 512
        num_warps = 4
    else:
        BLOCK_H = 256
        num_warps = 2

    grid = (N,)

    _scatter_add_chunks_kernel[grid](
        out_fp32,
        token_indices_i32,
        expert_outputs_fp32,
        N,
        H,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=2,
    )

    # Cast back to bfloat16 to match original API
    return out_fp32.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        # The reference forward clones and uses index_add in bfloat16.
        # Our Triton kernel accumulates in fp32 and returns fp32 cast to bfloat16.
        return _run_triton_scatter_add(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
