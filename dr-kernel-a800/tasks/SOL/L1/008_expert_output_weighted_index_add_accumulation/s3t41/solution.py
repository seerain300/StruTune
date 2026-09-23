import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_bf16_kernel(
    out_ptr,          # *bf16, shape [batch_seq_len, H]
    token_indices_ptr,  # *int32, shape [N]
    expert_ptr,       # *bf16, shape [N, H]
    N: tl.constexpr,  # number of updates
    H: tl.constexpr,  # hidden size
    BLOCK_H: tl.constexpr,  # vectorization width across hidden dim
):
    pid = tl.program_id(axis=0)
    # one program per update
    if pid >= N:
        return

    # Load the target token index for this update (int32)
    idx = tl.load(token_indices_ptr + pid)

    # Vectorized loop across hidden dimension in chunks of BLOCK_H
    h = 0
    while h < H:
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load v chunk from expert outputs
        # Address: expert_ptr + pid * H + offs
        v = tl.load(expert_ptr + pid * H + offs, mask=mask, other=0.0)

        # Atomic add into output row idx
        tl.atomic_add(out_ptr + idx * H + offs, v, mask=mask)

        h += BLOCK_H


def _triton_scatter_add_bf16(out: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Triton scatter-add in bfloat16:
      out[token_indices[i]] += expert_outputs[i, :]
    out: (B*seq_len, H) bfloat16
    expert_outputs: (N, H) bfloat16
    token_indices: (N,) int64 or int32
    Returns: updated out
    """
    assert out.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
    N = expert_outputs.shape[0]
    H = expert_outputs.shape[1]

    # Triton prefers int32 indices
    token_indices_i32 = token_indices.to(torch.int32)

    # Ensure contiguity
    out = out.contiguous()
    expert_outputs = expert_outputs.contiguous()

    # Choose BLOCK_H based on H
    BLOCK_H = 512 if H >= 512 else 256
    num_warps = 4 if BLOCK_H >= 512 else 2
    num_stages = 2

    grid = (N,)

    scatter_add_bf16_kernel[grid](
        out, token_indices_i32, expert_outputs,
        N, H,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
          final_hidden_states[token_indices[i]] += expert_outputs[i, :]
        Returns updated final_hidden_states.
        """
        # Ensure device and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton kernel operates in bfloat16; keep dtype consistent
        # The evaluation harness provides bfloat16 from get_inputs; ensure expert_outputs is bfloat16
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)

        # Launch Triton kernel
        out = _triton_scatter_add_bf16(final_hidden_states, expert_outputs, token_indices)

        return out


def run(*args):
    return ModelNew()(*args)
