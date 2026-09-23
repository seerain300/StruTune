import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_2d_kernel(
    output_ptr,           # *pointer to output [M, H], dtype bfloat16/float16
    expert_ptr,           # *pointer to expert_outputs [N, H]
    idx_ptr,              # *int64 pointer to token_indices [N]
    N: tl.constexpr,      # number of source rows
    H: tl.constexpr,      # hidden dimension
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2D grid: pid0 over N tiles, pid1 over H tiles
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Offsets for this program
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # Masks for bounds
    mask_n = n_offsets < N
    mask_h = h_offsets < H

    # Load token indices (int64) for these rows
    idx = tl.load(idx_ptr + n_offsets, mask=mask_n, other=0)  # int64 vector

    # Compute 2D pointer to expert outputs: expert_ptr + n_offsets[:, None] * H + h_offsets[None, :]
    expert_ptrs = expert_ptr + (n_offsets[:, None] * H + h_offsets[None, :])
    # 2D mask
    mask_2d = mask_n[:, None] & mask_h[None, :]
    vals = tl.load(expert_ptrs, mask=mask_2d, other=0.0)  # bfloat16/float16

    # Compute destination pointers into output: output_ptr + idx[:, None] * H + h_offsets[None, :]
    dest_ptrs = output_ptr + (idx[:, None] * H + h_offsets[None, :])

    # Atomic add into output
    tl.atomic_add(dest_ptrs, vals, mask=mask_2d)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-only scatter-add: output[token_indices[i]] += expert_outputs[i]
        """
        # Initialize output by cloning the input buffer
        output = final_hidden_states.clone()

        # Shapes
        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Heuristic tuning for tile sizes and warps
        if H >= 1024:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 3
        elif H >= 512:
            BLOCK_H = 256
            num_warps = 4
            num_stages = 2
        elif H >= 256:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        BLOCK_N = 128 if N >= 128 else 64

        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

        # Launch Triton kernel
        scatter_add_2d_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


# Optional: helper to mirror the original get_inputs for local testing.
def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    """Returns the input arguments for the reference forward pass. Required method."""
    batch_size, seq_len, hidden_size = (
        axes_and_scalars["batch_size"],
        axes_and_scalars["seq_len"],
        axes_and_scalars["hidden_size"],
    )
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len
    num_selected_tokens = batch_seq_len * num_experts_per_tok  # each token can have multiple experts

    # Initialize accumulation buffer with random values (not zeros) to detect no-op
    final_hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)

    # Expert outputs (weighted outputs from expert computation)
    expert_outputs = torch.randn(num_selected_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Token indices (which token position each expert output belongs to)
    token_indices = torch.randint(
        0, batch_seq_len, (num_selected_tokens,), dtype=torch.long, device=device
    )

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }


def run(*args):
    return ModelNew()(*args)
