# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r5 score=8.240445315392883 passed=True
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
def _expert_major_sort_kernel(
    topk_idx_ptr,
    sorted_token_indices_ptr,
    expert_offsets_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    expert0 = pair * 2
    expert1 = expert0 + 1
    lane = tl.arange(0, BLOCK)
    RADIX: tl.constexpr = 2 * BLOCK

    before0 = tl.zeros((), tl.int32)
    count0 = tl.zeros((), tl.int32)

    for start in tl.range(0, n_elements, BLOCK, num_stages=1):
        offsets = start + lane
        mask = offsets < n_elements
        values = tl.load(topk_idx_ptr + offsets, mask=mask, other=256)

        is_before = (values < expert0).to(tl.int32)
        is_expert0 = (values == expert0).to(tl.int32)
        packed = is_before + is_expert0 * RADIX
        packed_count = tl.sum(packed, axis=0)

        before0 += packed_count % RADIX
        count0 += packed_count // RADIX

    tl.store(expert_offsets_ptr + expert0, before0)
    tl.store(expert_offsets_ptr + expert1, before0 + count0)

    if pair == 127:
        tl.store(expert_offsets_ptr + 256, n_elements)

    emitted0 = tl.zeros((), tl.int32)
    emitted1 = tl.zeros((), tl.int32)

    for start in tl.range(0, n_elements, BLOCK, num_stages=1):
        offsets = start + lane
        mask = offsets < n_elements
        values = tl.load(topk_idx_ptr + offsets, mask=mask, other=256)

        selected0 = values == expert0
        selected1 = values == expert1

        packed = (
            selected0.to(tl.int32)
            + selected1.to(tl.int32) * RADIX
        )
        packed_prefix = tl.cumsum(packed, axis=0)

        ranks0 = packed_prefix % RADIX - 1
        ranks1 = packed_prefix // RADIX - 1

        tl.store(
            sorted_token_indices_ptr + before0 + emitted0 + ranks0,
            offsets,
            mask=selected0,
        )
        tl.store(
            sorted_token_indices_ptr + before0 + count0 + emitted1 + ranks1,
            offsets,
            mask=selected1,
        )

        packed_count = tl.sum(packed, axis=0)
        emitted0 += packed_count % RADIX
        emitted1 += packed_count // RADIX


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    num_tokens = flat.numel()

    sorted_token_indices = torch.empty(
        num_tokens,
        dtype=torch.int32,
        device=flat.device,
    )
    expert_offsets = torch.empty(
        257,
        dtype=torch.int32,
        device=flat.device,
    )

    if num_tokens <= 4096:
        block = triton.next_power_of_2(num_tokens)
        num_warps = 8 if block >= 2048 else 4

        _resident_composite_sort_kernel[(1,)](
            flat,
            sorted_token_indices,
            expert_offsets,
            n_elements=num_tokens,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        block = 2048 if num_tokens >= 32768 else 1024

        _expert_major_sort_kernel[(128,)](
            flat,
            sorted_token_indices,
            expert_offsets,
            num_tokens,
            BLOCK=block,
            num_warps=8,
            num_stages=1,
        )

    return sorted_token_indices, expert_offsets