import torch
import triton
import triton.language as tl


@triton.jit
def _atomically_add_experts_kernel(
    out_ptr,        # *half, output to be accumulated into (shape [N, H], contiguous)
    expert_ptr,     # *half, expert_outputs (shape [N, H], contiguous)
    indices_ptr,    # *int32, token_indices (shape [N], contiguous)
    N,              # int32, number of selected tokens
    H,              # int32, hidden size (columns)
    row_stride: tl.constexpr,  # typically H
    col_stride: tl.constexpr,  # typically 1
    BLOCK: tl.constexpr,       # chunk size along hidden dimension
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load destination row index (int32)
    idx = tl.load(indices_ptr + pid)

    # Loop over hidden dimension in chunks of BLOCK
    for col_start in range(0, H, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)
        # Mask for valid columns
        mask_cols = cols < H

        # Compute source and destination pointers for this chunk
        # Source: expert row pid
        src_row_ptr = expert_ptr + pid * H
        src_ptrs = src_row_ptr + cols * col_stride

        # Destination: row idx, columns cols
        dst_row_ptr = out_ptr + idx * row_stride
        dst_ptrs = dst_row_ptr + cols * col_stride

        # Load chunk from expert_outputs; dtype is bfloat16
        vals = tl.load(src_ptrs, mask=mask_cols, other=0.0)

        # Atomically add into output
        tl.atomic_add(dst_ptrs, vals, mask=mask_cols)


def _next_power_of_two(x: int) -> int:
    # Returns the next power of two >= x, capped at 256
    if x <= 64:
        return 64
    elif x <= 128:
        return 128
    else:
        return 256


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on same device and dtype; we assume get_inputs provided bfloat16 and long for indices
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All inputs must be on CUDA device"
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "dtype must be bfloat16"

        # Clone the output to avoid modifying the input; necessary to match reference behavior
        out = final_hidden_states.clone()
        # Ensure contiguous memory layout
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton kernel launch
        N = expert_outputs.shape[0]  # number of selected tokens
        H = out.shape[1]             # hidden size

        # Choose BLOCK as next power-of-two of H, capped at 256 (to minimize chunks)
        BLOCK = _next_power_of_two(H)
        # Tune num_warps based on BLOCK
        num_warps = 4 if BLOCK <= 128 else 8
        grid = (N,)

        _atomically_add_experts_kernel[grid](
            out,                      # out_ptr
            expert_outputs,           # expert_ptr
            token_indices.to(torch.int32),  # indices_ptr (convert to int32 for Triton)
            N, H,
            row_stride=H,             # second dimension stride for contiguous [N,H] tensor
            col_stride=1,             # contiguous along columns
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
