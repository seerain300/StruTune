# solution=GPT-5.6-Sol_008_expert_output_weighted_index_add_accumulation_triton_optimized_r12 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    batch_size, seq_len, hidden_size = (
        axes_and_scalars["batch_size"],
        axes_and_scalars["seq_len"],
        axes_and_scalars["hidden_size"],
    )
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len
    num_selected_tokens = batch_seq_len * num_experts_per_tok

    final_hidden_states = torch.randn(
        batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device
    )
    expert_outputs = torch.randn(
        num_selected_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    token_indices = torch.randint(
        0,
        batch_seq_len,
        (num_selected_tokens,),
        dtype=torch.long,
        device=device,
    )

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }


@triton.jit
def _chunk_collated_index_add(
    output_ptr,
    expert_ptr,
    indices_ptr,
    num_rows,
    hidden_size: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    chunk_id = tl.program_id(0)
    hidden_id = tl.program_id(1)

    row_offsets = tl.arange(0, CHUNK)
    hidden_offsets = hidden_id * BLOCK_H + tl.arange(0, BLOCK_H)

    rows = chunk_id * CHUNK + row_offsets
    row_mask = rows < num_rows
    hidden_mask = hidden_offsets < hidden_size

    indices = tl.load(indices_ptr + rows, mask=row_mask, other=0)
    source = tl.load(
        expert_ptr
        + rows[:, None] * hidden_size
        + hidden_offsets[None, :],
        mask=row_mask[:, None] & hidden_mask[None, :],
        other=0.0,
    )

    for owner in range(CHUNK):
        owner_valid = row_mask[owner]
        owner_index = indices[owner]

        is_first = owner_valid
        for previous in range(owner):
            is_first &= (~row_mask[previous]) | (
                indices[previous] != owner_index
            )

        matches = row_mask & (indices == owner_index)
        accumulated = tl.sum(
            tl.where(matches[:, None], source, 0.0),
            axis=0,
        )

        tl.atomic_add(
            output_ptr + owner_index * hidden_size + hidden_offsets,
            accumulated.to(tl.bfloat16),
            mask=is_first & hidden_mask,
        )


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    output = final_hidden_states.clone()

    hidden_size = output.shape[1]
    num_rows = expert_outputs.shape[0]
    chunk = 8
    block_h = 128

    grid = (
        triton.cdiv(num_rows, chunk),
        triton.cdiv(hidden_size, block_h),
    )

    _chunk_collated_index_add[grid](
        output,
        expert_outputs,
        token_indices,
        num_rows,
        hidden_size,
        CHUNK=chunk,
        BLOCK_H=block_h,
        num_warps=4,
    )

    return output