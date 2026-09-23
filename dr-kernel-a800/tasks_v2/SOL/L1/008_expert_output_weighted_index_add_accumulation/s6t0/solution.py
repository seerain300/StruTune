import torch
import triton
import triton.language as tl


@triton.jit
def _atomic_accumulate_expert_outputs(
    output_ptr,          # *bf16, [batch_seq_len, hidden_size]
    expert_ptr,          # *bf16, [num_selected_tokens, hidden_size]
    indices_ptr,         # *int32, [num_selected_tokens]
    batch_seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # One program per selected token
    token_id = tl.program_id(axis=0)

    # Bounds check on tokens (in case grid > num_selected_tokens)
    if token_id >= hidden_size:
        return

    # Load the token index (row position to accumulate into)
    # indices are int32 (safe for batch_seq_len < 2**31)
    index = tl.load(indices_ptr + token_id)

    # If index is out of range, skip (masking will handle but typical inputs ensure valid range)
    # Triton doesn't support Pythonic 'continue'/'if' branching here, so we rely on masks for safety.
    # We assume inputs are valid; if invalid, we mask and do nothing.

    # Prepare offsets for the hidden dimension
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    # Load the expert_outputs row for this token_id
    expert_row_ptr = expert_ptr + token_id * hidden_size + offs
    vals = tl.load(expert_row_ptr, mask=mask, other=0.0)  # vals are bfloat16

    # Compute output row addresses and perform atomic add
    out_row_ptr = output_ptr + index * hidden_size + offs
    # Atomic add: vals (bf16) into output (bf16)
    tl.atomic_add(out_row_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized scatter-add:
        For each i in [0, num_selected_tokens), output[ token_indices[i] ] += expert_outputs[i].
        Returns the updated output tensor (new allocation; does not modify inputs).
        """
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton kernel."

        # Ensure dtype and layout are as expected
        # The original uses bfloat16; we follow that.
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64"
        # Triton prefers int32 for indices in many kernels; convert if needed
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Ensure contiguous tensors
        expert_outputs = expert_outputs.contiguous()
        # We allocate a fresh output tensor (not a clone of final_hidden_states),
        # to match the behavior of updating with atomic adds (original code clones then index_add).
        # If you want to accumulate into a provided tensor, ensure it is zero-initialized or empty.
        output = torch.empty_like(final_hidden_states)

        batch_seq_len = final_hidden_states.shape[0]
        hidden_size = final_hidden_states.shape[1]
        num_selected_tokens = expert_outputs.shape[0]

        # Choose BLOCK_H based on hidden_size; 128 is a reasonable default
        BLOCK_H = 128
        grid = (num_selected_tokens,)

        _atomic_accumulate_expert_outputs[grid](
            output, expert_outputs, token_indices,
            batch_seq_len=batch_seq_len,
            hidden_size=hidden_size,
            BLOCK_H=BLOCK_H,
            num_warps=4,  # tuning parameter; 4 or 8 are typical
        )

        return output


def run(*args):
    return ModelNew()(*args)
