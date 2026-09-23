import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_bf16(
    output_ptr,            # *ptr to (M, H) output tensor
    expert_outputs_ptr,    # *ptr to (N, H) expert_outputs tensor
    token_indices_ptr,     # *ptr to (N,) indices
    M, H, N,               # sizes: M rows, H hidden dim, N number of updates
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid: each program handles BLOCK_SIZE rows (updates)
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load token indices for these rows
    idx = tl.load(token_indices_ptr + offs, mask=mask, other=0)  # int32/int64

    # Loop over hidden dimension H; vectorize across rows handled by this program
    for h in range(0, H):
        # Load expert_outputs row h across BLOCK_SIZE rows
        vals = tl.load(expert_outputs_ptr + offs * H + h, mask=mask, other=0).to(tl.bfloat16)

        # Destination address: idx * H + h
        dest_addr = idx * H + h
        # Atomic add into output
        tl.atomic_add(output_ptr + dest_addr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along dim=0:
        output[token_indices[i]] += expert_outputs[i] for all i
        """
        # Ensure CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "Triton kernels require CUDA tensors. Move inputs to CUDA device."
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64"

        # Compute sizes
        M = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]
        assert expert_outputs.shape[1] == H, "expert_outputs second dimension must equal hidden_size"
        assert token_indices.shape[0] == N, "token_indices length must equal number of expert outputs"

        # Clone to keep original buffer intact (matching reference semantics)
        output = final_hidden_states.clone()

        # Choose block size and num_warps
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        # Launch Triton kernel
        scatter_add_dim0_bf16[grid](
            output, expert_outputs, token_indices,
            M, H, N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        return output


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
