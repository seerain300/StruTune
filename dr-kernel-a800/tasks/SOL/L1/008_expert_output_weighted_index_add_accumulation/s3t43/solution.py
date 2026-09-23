import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_bf16_to_fp32_atomic(
    out_fp32_ptr,          # *float32, shape [B, H] (B = batch_seq_len)
    token_indices_i32_ptr, # *int32, shape [N]
    expert_outputs_fp32_ptr,  # *float32, shape [N, H]
    N,                      # int32, number of updates
    H,                      # int32, hidden_size
    BLOCK_H: tl.constexpr,  # compile-time block size along hidden dim
):
    # One program per update
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load token index for this update
    idx = tl.load(token_indices_i32_ptr + pid)  # int32

    # Iterate over hidden dimension in chunks
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        # Load source vector (float32) for this chunk
        v = tl.load(expert_outputs_fp32_ptr + pid * H + offs, mask=mask, other=0.0)  # [BLOCK_H], fp32
        # Atomic add into output row (fp32)
        tl.atomic_add(out_fp32_ptr + idx * H + offs, v, mask=mask)


def _choose_block_h_and_warps(H: int):
    # Heuristics: use larger BLOCK_H for larger H; cap at 1024
    if H >= 1024:
        return 1024, 8
    elif H >= 512:
        return 512, 4
    else:
        return 256, 2


def _triton_scatter_add_bf16(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Triton-optimized scatter-add in bfloat16 API:
      out = final_hidden_states.clone()
      out.index_add_(dim=0, index=token_indices, source=expert_outputs)
    Returns updated out.
    """
    # Ensure CUDA and contiguity
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
    final_hidden_states = final_hidden_states.contiguous()
    expert_outputs = expert_outputs.contiguous()
    token_indices = token_indices.contiguous()

    B, H = final_hidden_states.shape  # batch_seq_len
    N = expert_outputs.shape[0]

    # Prepare buffers:
    # - Keep final_hidden_states in bfloat16 (to match API and original behavior).
    # - We will accumulate into an fp32 copy (for Triton atomic_add), and cast back to bf16 at the end.
    out_fp32 = final_hidden_states.to(torch.float32).clone()

    # Cast expert_outputs to fp32 for kernel computation
    expert_outputs_fp32 = expert_outputs.to(torch.float32)

    # Cast token_indices to int32 for Triton
    token_indices_i32 = token_indices.to(torch.int32)

    # Choose BLOCK_H and num_warps
    BLOCK_H, num_warps = _choose_block_h_and_warps(H)

    # Launch kernel: one program per update
    grid = (N,)
    scatter_add_bf16_to_fp32_atomic[grid](
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
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton-optimized scatter-add with bfloat16 API compatibility.
        return _triton_scatter_add_bf16(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
