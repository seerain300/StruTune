import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel(
    out_ptr,          # *const bfloat16, shape [M, H]
    src_ptr,          # *const bfloat16, shape [N, H]
    indices_ptr,      # *const int32,    shape [N]
    N,                # number of rows in src_ptr/indices_ptr (same as number of adds)
    H: tl.constexpr,  # hidden size (compile-time for vectorization)
    BLOCK_H: tl.constexpr,  # tile size across hidden dimension
):
    pid = tl.program_id(0)  # program processes one source row
    if pid >= N:
        return

    # Compute row base offsets
    src_row_ptr = src_ptr + pid * H
    out_row = tl.load(indices_ptr + pid)  # token index
    out_row = out_row  # use as int32 index

    # Iterate over hidden dimension in tiles
    for j in range(0, H, BLOCK_H):
        offs = j + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Load source values as bfloat16 (masked)
        vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)  # bfloat16

        # Compute output pointer for this row at those hidden positions
        out_col_ptr = out_ptr + out_row * H + offs

        # Atomic add into output
        tl.atomic_add(out_col_ptr, vals, mask=mask)


def _select_block_h_and_warps(H: int):
    # Choose BLOCK_H as next power-of-two of H, clamped to [128, 1024]
    if H <= 128:
        block_h = 128
    elif H <= 256:
        block_h = 256
    elif H <= 512:
        block_h = 512
    else:
        block_h = 1024

    # Select num_warps based on tile size
    num_warps = 4 if block_h <= 256 else 8
    return block_h, num_warps


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of:
          output = final_hidden_states.clone()
          output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expected bfloat16 tensors."
        assert token_indices.dtype == torch.int32, "token_indices must be int32."

        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs must have shape [N, H]."
        assert token_indices.shape[0] == N, "token_indices must have shape [N]."
        assert token_indices.max().item() < M and token_indices.min().item() >= 0, "token_indices must be in [0, M)."

        # Ensure contiguity
        out = final_hidden_states.contiguous().clone()

        # Prepare source and indices (contiguous)
        src = expert_outputs.contiguous()
        idx = token_indices.contiguous()

        # Select kernel launch parameters
        BLOCK_H, num_warps = _select_block_h_and_warps(H)

        # Launch one program per source row
        grid = (N,)
        scatter_add_rows_kernel[grid](
            out, src, idx,
            N=N, H=H, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=2,
        )
        return out


# For completeness, the original helper functions can be reused as-is.
def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Returns the input arguments for the reference forward pass. Required method."""
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
    # These should be in range [0, batch_seq_len)
    token_indices = torch.randint(
        0, batch_seq_len, (num_selected_tokens,), dtype=torch.int32, device=device
    )

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Performs atomic accumulation of expert outputs back to token positions.
    """
    output = final_hidden_states.clone()
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    return output


class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
