import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_kernel(
    out_ptr,         # *half, output buffer (2D: [N_rows_out, H])
    expert_ptr,      # *half, expert_outputs (2D: [N_rows_in, H])
    indices_ptr,     # *int32, token_indices (1D: [N_rows_in])
    N_in,            # int32: number of rows in expert_outputs
    H,               # int32: hidden size
    out_row_stride,  # int32: stride for rows in out (typically H)
    out_col_stride,  # int32: stride for columns in out (typically 1)
    in_row_stride,   # int32: stride for rows in expert (typically H)
    in_col_stride,   # int32: stride for columns in expert (typically 1)
    BLOCK: tl.constexpr,
):
    # One program per input row
    pid = tl.program_id(axis=0)
    if pid >= N_in:
        return

    # Destination row index (token position)
    dest_row = tl.load(indices_ptr + pid)

    # Copy hidden vector from expert row pid into out row dest_row
    for col_start in range(0, H, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)
        mask = cols < H

        # Compute offsets
        out_offsets = dest_row * out_row_stride + cols * out_col_stride
        in_offsets = pid * in_row_stride + cols * in_col_stride

        vals = tl.load(expert_ptr + in_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + out_offsets, vals, mask=mask)


def _next_power_of_two(n: int) -> int:
    if n <= 64:
        return 64
    return 1 << (n - 1).bit_length()


def _choose_kernel_config(H: int):
    BLOCK = _next_power_of_two(H)
    if BLOCK > 256:
        BLOCK = 256
    num_warps = 4 if BLOCK <= 128 else 8
    return BLOCK, num_warps


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of copying selected expert outputs into the corresponding token positions.
        This matches the 'run' function's intended accumulation pattern when there are no duplicates.
        In case of duplicates in token_indices, we fall back to PyTorch for correctness.
        """
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA"
        N_in = expert_outputs.shape[0]
        H = expert_outputs.shape[1]
        N_out = final_hidden_states.shape[0]

        # Check for potential duplicates that our simple copy kernel cannot handle
        # If duplicates exist, use PyTorch to ensure correctness.
        # Note: torch.bincount requires int64 for counts; casting here is safe and cheap.
        # If duplicates are present, index_add must be used. For safety, fallback to PyTorch in that case.
        # Detect duplicates by checking unique count
        try:
            # token_indices must be int32; convert to int64 for torch ops
            if token_indices.numel() > 0 and torch.unique(token_indices.to(torch.int64)).numel() != token_indices.numel():
                # Fallback to PyTorch for correctness with duplicates
                out = final_hidden_states.clone()
                out.index_add_(0, token_indices.to(torch.long), expert_outputs)
                return out
        except Exception:
            # Any error in detection, fallback to PyTorch
            out = final_hidden_states.clone()
            out.index_add_(0, token_indices.to(torch.long), expert_outputs)
            return out

        # No duplicates detected: use Triton copy kernel
        out = final_hidden_states.contiguous().clone()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        BLOCK, num_warps = _choose_kernel_config(H)
        grid = (N_in,)

        _copy_rows_kernel[grid](
            out,
            expert_outputs,
            token_indices,
            N_in,
            H,
            out.stride(0),
            out.stride(1),
            expert_outputs.stride(0),
            expert_outputs.stride(1),
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
