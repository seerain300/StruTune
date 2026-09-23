# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r12 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _resident_composite_sort_kernel(
    topk_idx_ptr,
    sorted_token_indices_ptr,
    expert_offsets_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    mask = lane < n_elements
    values = tl.load(topk_idx_ptr + lane, mask=mask, other=0)

    keys = values * BLOCK + lane
    keys = tl.where(mask, keys, 0x7FFFFFFF)
    sorted_keys = tl.sort(keys, dim=0)

    tl.store(
        sorted_token_indices_ptr + lane,
        sorted_keys % BLOCK,
        mask=mask,
    )

    counts = tl.histogram(values, 256)
    expert = tl.arange(0, 256)
    counts = tl.where(
        expert == 0,
        counts - (BLOCK - n_elements),
        counts,
    )
    cumulative = tl.cumsum(counts, axis=0)

    tl.store(expert_offsets_ptr, 0)
    tl.store(expert_offsets_ptr + expert + 1, cumulative)


@triton.jit
def _expert_major_sort_kernel_impl(
    topk_idx_ptr,
    sorted_token_indices_ptr,
    expert_offsets_ptr,
    n_elements: tl.constexpr,
    token_bucket,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    expert0 = pair * 2
    expert1 = expert0 + 1
    lane = tl.arange(0, BLOCK)

    RADIX: tl.constexpr = 2 * BLOCK
    N_FULL: tl.constexpr = n_elements // BLOCK
    REM: tl.constexpr = n_elements % BLOCK

    before0 = tl.zeros((), tl.int32)
    count0 = tl.zeros((), tl.int