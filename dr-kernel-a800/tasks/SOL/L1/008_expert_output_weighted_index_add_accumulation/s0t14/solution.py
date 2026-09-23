import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_unrolled_kernel(
    output_ptr,            # *output* (M, H), will be treated as float32 for kernel math
    expert_outputs_ptr,    # *expert_outputs* (N, H), float32
    token_indices_ptr,     # *token_indices* (N,), int64
    N: tl.constexpr,       # number of expert outputs
    H: tl.constexpr,       # hidden size
    UNROLL: tl.constexpr,  # number of rows processed per iteration
):
    # Each program handles UNROLL rows sequentially
    pid = tl.program_id(axis=0)
    base_i = pid * UNROLL

    # Loop over unrolled rows
    for u in range(UNROLL):
        i = base_i + u
        # Mask to avoid out-of-bounds when i >= N
        if i >= N:
            break

        # Load destination row index
        idx = tl.load(token_indices_ptr + i)  # int64
        # Compute base offsets
        row_offset = idx * H

        # Load and add expert_outputs[i, :] into output[idx, :]
        # Iterate over H in chunks of 1 (simple and robust)
        for h in range(H):
            val = tl.load(expert_outputs_ptr + i * H + h)  # float32
            out_val = tl.load(output_ptr + row_offset + h)  # float32
            out_val = out_val + val
            tl.store(output_ptr + row_offset + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-based scatter-add that replaces torch.index_add along dim=0.
        We operate in float32 inside the kernel for robustness and cast back to bfloat16 at the end.
        """
        # Ensure tensors are on CUDA
        if final_hidden_states.device.type != 'cuda':
            raise RuntimeError("ModelNew requires CUDA tensors. Move inputs to CUDA.")
        if expert_outputs.device.type != 'cuda':
            expert_outputs = expert_outputs.to(final_hidden_states.device)
        if token_indices.device.type != 'cuda':
            token_indices = token_indices.to(final_hidden_states.device)

        # Make sure tensors are contiguous
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Cast to float32 for Triton math (Triton is robust with float32)
        output = final_hidden_states.clone().to(torch.float32)
        expert_outputs_fp32 = expert_outputs.to(torch.float32)
        token_indices_fp64 = token_indices.to(torch.long)

        # Compute grid: 1D over N rows with unrolling
        N = expert_outputs_fp32.shape[0]
        # We use UNROLL=4 to increase throughput while keeping kernel simple and safe
        UNROLL = 4
        grid = (triton.cdiv(N, UNROLL),)

        # Launch Triton kernel
        scatter_add_rows_unrolled_kernel[grid](
            output, expert_outputs_fp32, token_indices_fp64,
            N=N, H=final_hidden_states.shape[1],
            UNROLL=UNROLL,
            num_warps=4, num_stages=2,
        )

        # Cast back to bfloat16 to match the original dtype
        return output.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
