# solution=GPT-5.6-Sol_008_expert_output_weighted_index_add_accumulation_triton_optimized_r27 score=3.8527163843778887 passed=True
import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 3072
_COPY_BLOCK_SIZE = 2048


@triton.jit
def _copy_kernel(
    src_ptr,
    dst_ptr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    values = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["num_selected_tokens"],
    restore_value=["output_ptr"],
)
@triton.jit
def _direct_scatter_kernel(
    expert_outputs_ptr,
    token_indices_ptr,
    output_ptr,
    num_selected_tokens,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    source_row = tl.program_id(0)
    destination_row = tl.load(token_indices_ptr + source_row)
    pair_columns = tl.arange(0, BLOCK_SIZE)

    for chunk in tl.static_range(0, HIDDEN_SIZE // (2 * BLOCK_SIZE)):
        even_columns = 2 * (chunk * BLOCK_SIZE + pair_columns)
        source_offsets = source_row * HIDDEN_SIZE + even_columns
        destination_offsets = destination_row * HIDDEN_SIZE + even_columns

        even_values = tl.load(
            expert_outputs_ptr + source_offsets,
            cache_modifier=".cg",
        )
        odd_values = tl.load(
            expert_outputs_ptr + source_offsets + 1,
            cache_modifier=".cg",
        )

        packed_values = tl.inline_asm_elementwise(
            asm="mov.b32 $0, {$1, $2};",
            constraints="=r,h,h",
            args=[even_values, odd_values],
            dtype=tl.int32,
            is_pure=True,
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
    output = torch.empty_like(final_hidden_states)

    num_output_elements = final_hidden_states.numel()
    _copy_kernel[(triton.cdiv(num_output_elements, _COPY_BLOCK_SIZE),)](
        final_hidden_states,
        output,
        num_elements=num_output_elements,
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
        num_warps=8,
    )

    num_selected_tokens = expert_outputs.shape[0]
    _direct_scatter_kernel[(num_selected_tokens,)](
        expert_outputs,
        token_indices,
        output,
        num_selected_tokens,
        HIDDEN_SIZE=_HIDDEN_SIZE,
    )

    return output