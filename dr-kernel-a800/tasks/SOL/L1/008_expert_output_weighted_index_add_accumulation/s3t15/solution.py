import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_fp32_atomic(
    out_ptr,                # *float32, [M, H] output buffer (M = batch_seq_len)
    idx_ptr,                # *int32,   [N] token indices
    A_ptr,                  # *float32, [N, H] expert outputs
    M: tl.constexpr,        # int, number of rows (batch_seq_len)
    N: tl.constexpr,        # int, number of updates
    H: tl.constexpr,        # int, hidden size
):
    # One program per update
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load index and vector
    idx = tl.load(idx_ptr + pid)  # int32
    # For hidden_size H, vectorized load of expert_outputs[pid, :]
    a = tl.load(A_ptr + pid * H + tl.arange(0, H))  # float32 vector of length H

    # Atomic add into the corresponding row in out_ptr
    row_base = out_ptr + idx * H
    tl.atomic_add(row_base + tl.arange(0, H), a)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward that performs:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, token_indices, expert_outputs)
        All computation is done by Triton kernels. Accumulation is in float32, final is cast back to bfloat16.
        """
        # Ensure tensors are on the same device and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device for Triton."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Output buffer in fp32 (clone initial state)
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Convert expert_outputs to fp32
        A = expert_outputs.to(torch.float32)

        # Convert token_indices to int32 for Triton
        idx_int32 = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per update
        grid = (N,)

        scatter_add_rows_fp32_atomic[grid](
            out_fp32, idx_int32, A,
            M=M, N=N, H=H,
            num_warps=4, num_stages=2,
        )

        # Cast back to bfloat16 to match original dtype
        result = out_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
