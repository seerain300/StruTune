# solution=GPT-5.6-Sol_008_expert_output_weighted_index_add_accumulation_triton_optimized_r17 score=-1.0 passed=False
I’m keeping the packed BF16 atomic scatter path intact and focusing the change on initialization overhead, which is the only independent phase left in this implementation. The output must remain distinct from the input before atomics begin, so I’m evaluating a more efficient contiguous clone path while preserving the existing launch tuning.import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 3072


@triton.jit
def _direct_scatter_kernel(
    expert_outputs_ptr,
    token_indices_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    source_row = tl.program_id(0)
    destination_row = tl.load(token_indices_ptr + source_row)
    pair_columns = tl.arange(0, BLOCK_SIZE)

    for chunk in tl.static_range(0, HIDDEN_SIZE // (4 * BLOCK_SIZE)):
        first_columns = 2 * (2 * chunk * BLOCK_SIZE + pair_columns)
        second_columns = first_columns + 2 * BLOCK_SIZE

        first_source_offsets = source_row * HIDDEN_SIZE + first_columns
        second_source_offsets = source_row * HIDDEN_SIZE + second_columns
        first_destination_offsets = destination_row * HIDDEN_SIZE + first_columns
        second_destination_offsets = destination_row * HIDDEN_SIZE + second_columns

        first_values = tl.inline_asm_elementwise(
            asm="ld.global.cs.b32 $0, [$1];",
            constraints="=r,l",
            args=[expert_outputs_ptr + first_source_offsets],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        second_values = tl.inline_asm_elementwise(
            asm="ld.global.cs.b32 $0, [$1];",
            constraints="=r,l",
            args=[expert_outputs_ptr + second_source_offsets],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

        tl.inline_asm_elementwise(
            asm="""
            red.global.relaxed.gpu.add.noftz.bf16x2 [$1], $2;
            mov.b32 $0, $2;
            """,
            constraints="=r,l,r",
            args=[output_ptr + first_destination_offsets, first_values],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.inline_asm_elementwise(
            asm="""
            red.global.relaxed.gpu.add.noftz.bf16x2 [$1], $2;
            mov.b32 $0, $2;
            """,
            constraints="=r,l,r",
            args=[output_ptr + second_destination_offsets, second_values],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    if HIDDEN_SIZE % (4 * BLOCK_SIZE):
        columns = 2 * (
            (HIDDEN_SIZE // (4 * BLOCK_SIZE)) * 2 * BLOCK_SIZE
            + pair_columns
        )
        source_offsets = source_row * HIDDEN_SIZE + columns
        destination_offsets = destination_row * HIDDEN_SIZE + columns

        packed_values = tl.inline_asm_elementwise(
            asm="ld.global.cs.b32 $0, [$1];",
            constraints="=r,l",
            args=[expert_outputs_ptr + source_offsets],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

        tl.inline_asm_elementwise(
            asm="""
            red.global.relaxed.gpu.add.noftz.bf16x2 [$1], $2;
            mov.b32 $0, $2;
            """,
            constraints="=r,l,r",
            args=[output_ptr + destination_offsets, packed_values],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    output = final_hidden_states.clone()
    num_selected_tokens = expert_outputs.shape[0]

    if num_selected_tokens <= 4096:
        block_size = 256
        num_warps = 4
    elif num_selected_tokens <= 65536:
        block_size = 128
        num_warps = 4
    else:
        block_size = 128
        num_warps = 2

    _direct_scatter_kernel[(num_selected_tokens,)](
        expert_outputs,
        token_indices,
        output,
        HIDDEN_SIZE=_HIDDEN_SIZE,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output