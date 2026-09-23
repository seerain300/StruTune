import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_kernel(src_ptr, dst_ptr, M, N, stride_src, stride_dst, BLOCK_N: tl.constexpr):
    # One Triton program handles one row and writes it in BLOCK_N-sized chunks.
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    # Iterate over columns in tiles of BLOCK_N with mask for tail
    for col_start in range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        src_row_ptr = src_ptr + row_id * stride_src + cols
        dst_row_ptr = dst_ptr + row_id * stride_dst + cols
        vals = tl.load(src_row_ptr, mask=mask, other=0.0)
        tl.store(dst_row_ptr, vals, mask=mask)


@triton.jit
def _scatter_add_rows_kernel(output_ptr, expert_ptr, indices_ptr, K, N, BLOCK_N: tl.constexpr):
    # One Triton program per token i; add the expert row to the selected output row.
    i = tl.program_id(0)
    if i >= K:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + i)
    # Loop over hidden dimension in BLOCK_N-sized chunks
    for col_start in range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        out_row_ptr = output_ptr + idx * N + cols
        expert_row_ptr = expert_ptr + i * N + cols
        add_vals = tl.load(expert_row_ptr, mask=mask, other=0.0)
        # Read current output, add, and write back
        out_vals = tl.load(out_row_ptr, mask=mask, other=0.0)
        out_vals = out_vals + add_vals
        tl.store(out_row_ptr, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized implementation:
        - Clone final_hidden_states into output via Triton kernel.
        - Perform scatter-add into output via Triton kernel (one program per token).
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton kernels."

        # Shapes
        M, N = final_hidden_states.shape
        K = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == N, "expert_outputs second dim must match hidden_size"
        assert token_indices.shape[0] == K, "token_indices length must match num_selected_tokens"

        # 1) Clone via Triton kernel
        output = torch.empty_like(final_hidden_states)

        # Heuristic tuning for BLOCK_N and num_warps based on N
        if N >= 4096:
            BLOCK_N_COPY = 512
            num_warps_copy = 8
        elif N >= 1024:
            BLOCK_N_COPY = 256
            num_warps_copy = 8
        elif N >= 256:
            BLOCK_N_COPY = 256
            num_warps_copy = 4
        elif N >= 128:
            BLOCK_N_COPY = 128
            num_warps_copy = 4
        else:
            BLOCK_N_COPY = 64
            num_warps_copy = 2

        grid_copy = (M,)
        _copy_rows_kernel[grid_copy](
            final_hidden_states, output,
            M, N,
            final_hidden_states.stride(0), output.stride(0),
            BLOCK_N=BLOCK_N_COPY,
            num_warps=num_warps_copy,
            num_stages=2
        )

        # 2) Scatter-add via Triton kernel (one program per token)
        indices32 = token_indices.to(torch.int32)
        grid_scatter = (K,)
        BLOCK_N_SCATTER = BLOCK_N_COPY  # reuse similar tiling
        num_warps_scatter = num_warps_copy

        _scatter_add_rows_kernel[grid_scatter](
            output, expert_outputs, indices32,
            K, N,
            BLOCK_N=BLOCK_N_SCATTER,
            num_warps=num_warps_scatter,
            num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
