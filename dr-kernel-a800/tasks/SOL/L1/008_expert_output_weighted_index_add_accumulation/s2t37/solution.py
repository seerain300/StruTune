import torch
import triton
import triton.language as tl


@triton.jit
def _init_output_from_clone_kernel(
    out_ptr,        # *half, output tensor to initialize
    src_ptr,        # *half, source tensor (final_hidden_states)
    M,              # int32, number of rows (batch_seq_len)
    H,              # int32, hidden size (columns)
    row_stride,     # int32, stride between rows (typically H)
    col_stride,     # int32, stride between columns (typically 1)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Copy row pid from src to out using 2D addressing: addr = base + row * row_stride + col * col_stride
    for col_start in range(0, H, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)
        mask = cols < H
        out_row = out_ptr + pid * row_stride + cols * col_stride
        src_row = src_ptr + pid * row_stride + cols * col_stride
        vals = tl.load(src_row, mask=mask, other=0.0)
        tl.store(out_row, vals, mask=mask)


@triton.jit
def _atomic_add_rows_kernel(
    out_ptr,         # *half, output tensor (already initialized as clone)
    expert_ptr,      # *half, expert_outputs [N, H]
    indices_ptr,     # *int32, token_indices [N]
    N,               # int32, number of selected tokens
    H,               # int32, hidden size
    row_stride,      # int32, stride between rows in out (typically H)
    col_stride,      # int32, stride between columns (typically 1)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Load destination row index for this selected token
    idx = tl.load(indices_ptr + pid)  # int32

    # Iterate over hidden dimension in chunks and atomically add
    for col_start in range(0, H, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)
        mask = cols < H

        # Load the expert vector chunk: contiguous across columns for row pid
        src_row = expert_ptr + pid * H + cols
        vals = tl.load(src_row, mask=mask, other=0.0)

        # Destination addresses in output using strides
        dest_row = out_ptr + idx * row_stride + cols * col_stride

        # Atomic add into output
        tl.atomic_add(dest_row, vals, mask=mask)


def _choose_block(H: int) -> int:
    # Next power-of-two of H, capped at 256. Ensure at least 64.
    if H <= 64:
        return 64
    elif H <= 128:
        return 128
    else:
        return 256


def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    """Returns the input arguments for the Triton forward pass. Triton-only version."""
    batch_size, seq_len, hidden_size = (
        axes_and_scalars["batch_size"],
        axes_and_scalars["seq_len"],
        axes_and_scalars["hidden_size"],
    )
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len
    num_selected_tokens = batch_size * seq_len * num_experts_per_tok

    # Initialize accumulation buffer with random values (not zeros) to detect no-op
    final_hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)

    # Expert outputs (weighted outputs from expert computation)
    expert_outputs = torch.randn(num_selected_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Token indices (which token position each expert output belongs to)
    token_indices = torch.randint(0, batch_seq_len, (num_selected_tokens,), dtype=torch.int32, device=device)

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        All data movement and accumulation are done via Triton kernels.
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton kernels."

        # Shapes
        batch_seq_len = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs second dim must equal hidden_size"
        assert token_indices.shape[0] == N, "token_indices length must equal num_selected_tokens"

        # Ensure contiguous and dtype
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Output: initialize as clone of final_hidden_states
        output = torch.empty_like(final_hidden_states)

        # Launch init kernel: copy final_hidden_states -> output (one program per row)
        BLOCK = _choose_block(H)
        _init_output_from_clone_kernel[(batch_seq_len,)](
            output, final_hidden_states,
            batch_seq_len, H,
            H, 1,
            BLOCK=BLOCK,
            num_warps=4 if BLOCK <= 128 else 8,
            num_stages=2,
        )

        # Launch atomic add kernel: add expert_outputs into selected rows of output using token_indices
        _atomic_add_rows_kernel[(N,)](
            output, expert_outputs, token_indices,
            N, H,
            H, 1,
            BLOCK=BLOCK,
            num_warps=4 if BLOCK <= 128 else 8,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
