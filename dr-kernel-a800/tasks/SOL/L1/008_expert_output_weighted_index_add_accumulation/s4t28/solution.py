import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_kernel(out_ptr, in_ptr, B, H: tl.constexpr):
    """
    Copy one row i: out[i, :] = in[i, :]
    Launch: grid = (B,), each program copies a single row.
    H must be passed as a compile-time constant for loop unrolling.
    """
    i = tl.program_id(0)
    if i < B:
        # Loop over columns; H is constexpr so Triton can unroll or optimize
        for h in range(H):
            val = tl.load(in_ptr + i * H + h)
            tl.store(out_ptr + i * H + h, val)


@triton.jit
def _scatter_add_rows_kernel(out_ptr, exp_ptr, idx_ptr, T, H: tl.constexpr):
    """
    Scatter-add along rows:
    For each i in [0, T):
      idx = token_indices[i] (int64, cast to int32)
      For each h in [0, H):
        out[idx, h] += exp[i, h]
    Launch: grid = (T,), each program handles one source row and loops over H.
    H must be passed as a compile-time constant for loop unrolling.
    """
    i = tl.program_id(0)
    if i < T:
        idx64 = tl.load(idx_ptr + i)
        idx = idx64.to(tl.int32)
        for h in range(H):
            val = tl.load(exp_ptr + i * H + h)
            tl.store(out_ptr + idx * H + h, val, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized implementation of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        T = token_indices.shape[0]

        # Allocate output tensor (same shape/dtype/device)
        output = torch.empty_like(final_hidden_states)

        # Ensure contiguity
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()
        output = output.contiguous()

        # Triton copy: one program per row
        # We pass H as a constexpr meta-parameter. Triton allows this in @triton.jit signature.
        grid_copy = (B,)
        _copy_rows_kernel[grid_copy](output, final_hidden_states, B, H)

        # Triton scatter-add: one program per source row
        grid_scatter = (T,)
        _scatter_add_rows_kernel[grid_scatter](output, expert_outputs, token_indices, T, H)

        return output


def run(*args):
    return ModelNew()(*args)
