# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r46 score=15.146392716453613 passed=True
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
        sorted_keys & (BLOCK - 1),
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
def _resident_composite_sort_exact_1024_kernel(
    topk_idx_ptr,
    sorted_token_indices_ptr,
    expert_offsets_ptr,
):
    lane = tl.arange(0, 1024)
    values = tl.load(topk_idx_ptr + lane)

    keys = values * 1024 + lane
    sorted_keys = tl.sort(keys, dim=0)

    tl.store(
        sorted_token_indices_ptr + lane,
        sorted_keys & 1023,
    )

    counts = tl.histogram(values, 256)
    expert = tl.arange(0, 256)
    cumulative = tl.cumsum(counts, axis=0)

    tl.store(expert_offsets_ptr, 0)
    tl.store(expert_offsets_ptr + expert + 1, cumulative)


@triton.jit
def _resident_composite_sort_large_kernel(
    topk_idx_ptr,
    sorted_token_indices_ptr,
    expert_offsets_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)

    if n_elements == 2048 or n_elements == 4096:
        counts = tl.histogram(
            tl.load(topk_idx_ptr + lane),
            256,
        )
    else:
        mask = lane < n_elements
        values = tl.load(topk_idx_ptr + lane, mask=mask, other=0)
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

    if n_elements == 2048 or n_elements == 4096:
        tl.store(
            sorted_token_indices_ptr + lane,
            tl.sort(
                tl.load(topk_idx_ptr + lane) * BLOCK + lane,
                dim=0,
            )
            & (BLOCK - 1),
        )
    else:
        keys = values * BLOCK + lane
        keys = tl.where(mask, keys, 0x7FFFFFFF)
        sorted_keys = tl.sort(keys, dim=0)

        tl.store(
            sorted_token_indices_ptr + lane,
            sorted_keys & (BLOCK - 1),
            mask=mask,
        )


@triton.jit
def _expert_major_sort_kernel_impl(
    topk_idx_ptr,
    sorted_token_indices_ptr,
    expert_offsets_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    expert0 = pair * 2
    expert1 = expert0 + 1
    lane = tl.arange(0, BLOCK)

    RADIX: tl.constexpr = 2 * BLOCK
    RADIX_MASK: tl.constexpr = RADIX - 1
    N_FULL: tl.constexpr = n_elements // BLOCK
    REM: tl.constexpr = n_elements % BLOCK

    before0 = tl.zeros((), tl.int32)
    count0 = tl.zeros((), tl.int32)

    for tile in tl.range(0, N_FULL, num_stages=1):
        offsets = tile * BLOCK + lane
        values = tl.load(topk_idx_ptr + offsets)

        is_before = (values < expert0).to(tl.int32)
        is_expert0 = (values == expert0).to(tl.int32)
        packed = is_before + is_expert0 * RADIX
        packed_count = tl.sum(packed, axis=0)

        before0 += packed_count & RADIX_MASK
        count0 += packed_count // RADIX

    if REM:
        offsets = N_FULL * BLOCK + lane
        values = tl.load(
            topk_idx_ptr + offsets,
            mask=lane < REM,
            other=256,
        )

        is_before = (values < expert0).to(tl.int32)
        is_expert0 = (values == expert0).to(tl.int32)
        packed = is_before + is_expert0 * RADIX
        packed_count = tl.sum(packed, axis=0)

        before0 += packed_count & RADIX_MASK
        count0 += packed_count // RADIX

    tl.store(expert_offsets_ptr + expert0, before0)
    tl.store(expert_offsets_ptr + expert1, before0 + count0)

    if pair == 127:
        tl.store(expert_offsets_ptr + 256, n_elements)

    emitted0 = tl.zeros((), tl.int32)
    emitted1 = tl.zeros((), tl.int32)

    for tile in tl.range(0, N_FULL, num_stages=1):
        offsets = tile * BLOCK + lane
        values = tl.load(topk_idx_ptr + offsets)

        selected0 = values == expert0
        selected1 = values == expert1

        packed = (
            selected0.to(tl.int32)
            + selected1.to(tl.int32) * RADIX
        )
        packed_prefix = tl.cumsum(packed, axis=0)

        ranks0 = (packed_prefix & RADIX_MASK) - 1
        ranks1 = packed_prefix // RADIX - 1

        tl.store(
            sorted_token_indices_ptr + before0 + emitted0 + ranks0,
            offsets,
            mask=selected0,
        )
        tl.store(
            sorted_token_indices_ptr
            + before0
            + count0
            + emitted1
            + ranks1,
            offsets,
            mask=selected1,
        )

        packed_count = tl.sum(packed, axis=0)
        emitted0 += packed_count & RADIX_MASK
        emitted1 += packed_count // RADIX

    if REM:
        offsets = N_FULL * BLOCK + lane
        values = tl.load(
            topk_idx_ptr + offsets,
            mask=lane < REM,
            other=256,
        )

        selected0 = values == expert0
        selected1 = values == expert1

        packed = (
            selected0.to(tl.int32)
            + selected1.to(tl.int32) * RADIX
        )
        packed_prefix = tl.cumsum(packed, axis=0)

        ranks0 = (packed_prefix & RADIX_MASK) - 1
        ranks1 = packed_prefix // RADIX - 1

        tl.store(
            sorted_token_indices_ptr + before0 + emitted0 + ranks0,
            offsets,
            mask=selected0,
        )
        tl.store(
            sorted_token_indices_ptr
            + before0
            + count0
            + emitted1
            + ranks1,
            offsets,
            mask=selected1,
        )


_pair_scan_winners = (
    (1024, 4),
    (2048, 8),
    (4096, 8),
    (8192, 8),
    (8192, 8),
)


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

        if block < 1024:
            num_warps = 4
        else:
            num_warps = 8

        if num_tokens == 1024:
            _resident_composite_sort_exact_1024_kernel[(1,)](
                flat,
                sorted_token_indices,
                expert_offsets,
                num_warps=8,
                num_stages=1,
            )
        elif block >= 2048:
            _resident_composite_sort_large_kernel[(1,)](
                flat,
                sorted_token_indices,
                expert_offsets,
                n_elements=num_tokens,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
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
        if num_tokens < 8192:
            token_bucket = 0
        elif num_tokens < 16384:
            token_bucket = 1
        elif num_tokens < 32768:
            token_bucket = 2
        elif num_tokens < 65536:
            token_bucket = 3
        else:
            token_bucket = 4

        block, num_warps = _pair_scan_winners[token_bucket]

        _expert_major_sort_kernel_impl[(128,)](
            flat,
            sorted_token_indices,
            expert_offsets,
            n_elements=num_tokens,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )

    return sorted_token_indices, expert_offsets