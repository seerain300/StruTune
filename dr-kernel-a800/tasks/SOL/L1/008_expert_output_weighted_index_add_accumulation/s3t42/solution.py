import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_chunks_kernel(
    out_ptr: tl.pointer_type(dtype=tl.float32),   # accumulate in fp32
    indices_ptr: tl.pointer_type(dtype=tl.int32),  # token_indices as int32
    src_ptr: tl.pointer_type(dtype=tl.bfloat16),    # expert_outputs in bfloat16
    N: tl.int32,  # number of updates
    H: tl.int32,  # hidden size
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per update
    # Guard in case grid > N (not used here, but safe)
    if pid >= N:
        return

    # Load the target row index for this update
    idx = tl.load(indices_ptr + pid)  # int32

    # Iterate over hidden dimension in BLOCK_H chunks
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load a chunk of the expert output (bfloat16), cast to float32
        v = tl.load(src_ptr + pid * H + offs, mask=mask, other=0.0)
        v = v.to(tl.float32)

        # Atomic add into the output at row 'idx'
        out_row_ptr = out_ptr + idx * H
        tl.atomic_add(out_row_ptr + offs, v, mask=mask)


def _choose_block_and_warps(hidden_size: int):
    # Heuristic selection: larger BLOCK_H reduces atomics, smaller increases per-program work.
    # 1024 is ideal for common hidden_size=1024; 512/256 for smaller H.
    if hidden_size >= 1024:
        return 1024, 8
    elif hidden_size >= 512:
        return 512, 4
    else:
        return 256, 2


class Model(torch.nn.Module):
    """
    Triton-optimized version of the original run:
      output[token_indices[i]] += expert_outputs[i, :]
    This avoids torch.index_add_ and performs the scatter-add via Triton atomics in fp32,
    then casts back to bfloat16 to match the original interface.
    """
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure all tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        # Ensure contiguity
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Prepare dtypes
        token_indices_i32 = token_indices.to(torch.int32)

        # Accumulate in float32 for atomic support
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        BLOCK_H, num_warps = _choose_block_and_warps(H)
        grid = (N,)

        _scatter_add_chunks_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs,  # bfloat16
            N,
            H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


# Provide ModelNew for parity, identical behavior
class ModelNew(Model):
    pass


def run(*args):
    return ModelNew()(*args)
