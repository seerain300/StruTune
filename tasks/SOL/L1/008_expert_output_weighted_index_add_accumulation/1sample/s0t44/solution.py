import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel_1d(
    out_ptr, out_stride0, out_stride1,
    src_ptr, src_stride0, src_stride1,
    token_idx_ptr,  # int32 indices
    N_rows, H, N_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D launch: one program per token
    pid = tl.program_id(0)
    # Load the destination row index for this token
    tok = tl.load(token_idx_ptr + pid)
    # Guard against out-of-range tokens (defensive; grid should ensure pid < N_tokens)
    valid_token = pid < N_tokens

    # Iterate over hidden dimension in chunks of BLOCK_SIZE
    for off in tl.static_range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = (cols < H) & valid_token

        # Compute source and output pointers for this token and column chunk
        src_ptrs = src_ptr + pid * src_stride0 + cols * src_stride1
        out_ptrs = out_ptr + tok * out_stride0 + cols * out_stride1

        # Load source chunk (masked) and atomic add to output
        vals = tl.load(src_ptrs, mask=mask, other=0)
        # Atomic add only for valid lanes
        tl.atomic_add(out_ptrs, vals, mask=mask)


def run(final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Performs atomic accumulation of expert outputs back to token positions.
    This matches the original PyTorch behavior: output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    """
    # Clone to avoid modifying input in-place and preserve initial random values
    output = final_hidden_states.clone()

    # Ensure contiguity and dtypes
    batch_seq_len = output.shape[0]
    hidden_size = output.shape[1]
    n_tokens = expert_outputs.shape[0]

    # Triton expects indices as int32 and contiguous
    token_indices_i32 = token_indices.to(torch.int32).contiguous()
    expert_outputs = expert_outputs.contiguous()
    output = output.contiguous()

    # Kernel launch configuration: 1D grid over tokens, iterate hidden dimension inside the kernel
    grid = (n_tokens,)

    # Choose block size and warps (balanced for a wide range of GPUs)
    BLOCK_SIZE = 256
    num_warps = 4

    scatter_add_atomic_kernel_1d[grid](
        output, output.stride(0), output.stride(1),
        expert_outputs, expert_outputs.stride(0), expert_outputs.stride(1),
        token_indices_i32,
        batch_seq_len, hidden_size, n_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )

    return output


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same argument order as the original: (final_hidden_states, expert_outputs, token_indices)
        if len(args) != 3:
            raise RuntimeError("ModelNew.forward expects three tensors: final_hidden_states, expert_outputs, token_indices.")
        final_hidden_states, expert_outputs, token_indices = args
        return run(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
