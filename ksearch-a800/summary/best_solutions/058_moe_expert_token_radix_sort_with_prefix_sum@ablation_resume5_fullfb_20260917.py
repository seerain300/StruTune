# task: 058_moe_expert_token_radix_sort_with_prefix_sum
# batch: ablation_resume5_fullfb_20260917 (ablation)
# final eval: valid=True pass=16/16 geomean=10.324x

import torch
import triton
import triton.language as tl


@triton.jit
def _composite_sort_kernel(
    topk_idx_ptr,
    sorted_indices_ptr,
    expert_offsets_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    positions = tl.arange(0, BLOCK_SIZE)
    mask = positions < n_elements

    experts = tl.load(topk_idx_ptr + positions, mask=mask, other=0)
    composite_keys = (
        experts << tl.constexpr(BLOCK_SIZE.bit_length() - 1)
    ) | positions
    composite_keys = tl.where(mask, composite_keys, 0x7FFFFFFF)

    sorted_keys = tl.sort(composite_keys, dim=0)
    sorted_positions = sorted_keys & (BLOCK_SIZE - 1)
    tl.store(sorted_indices_ptr + positions, sorted_positions, mask=mask)

    bins = tl.arange(0, 256)
    counts = tl.histogram(experts, 256)
    invalid_count = BLOCK_SIZE - n_elements
    counts -= tl.where(bins == 0, invalid_count, 0)
    cumulative_counts = tl.cumsum(counts, axis=0)

    tl.store(expert_offsets_ptr, 0)
    tl.store(expert_offsets_ptr + bins + 1, cumulative_counts)


@triton.jit
def _local_sort_kernel(
    topk_idx_ptr,
    temporary_keys_ptr,
    metadata_ptr,
    n_elements,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KEY_SHIFT: tl.constexpr,
    METADATA_SHIFT: tl.constexpr,
    FULL_CHUNKS: tl.constexpr,
):
    chunk = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    positions = chunk * BLOCK_SIZE + lanes
    bins = tl.arange(0, 256)

    if FULL_CHUNKS:
        experts = tl.load(topk_idx_ptr + positions)
        keys = (experts << KEY_SHIFT) | lanes
        sorted_keys = tl.sort(keys, dim=0)
        tl.store(temporary_keys_ptr + positions, sorted_keys)
        counts = tl.histogram(experts, 256)
    else:
        if chunk == N_CHUNKS - 1:
            mask = positions < n_elements
            experts = tl.load(topk_idx_ptr + positions, mask=mask, other=0)
            keys = (experts << KEY_SHIFT) | lanes
            keys = tl.where(mask, keys, 0x7FFFFFFF)
            sorted_keys = tl.sort(keys, dim=0)
            tl.store(temporary_keys_ptr + positions, sorted_keys, mask=mask)

            counts = tl.histogram(experts, 256)
            invalid_count = BLOCK_SIZE - (n_elements - chunk * BLOCK_SIZE)
            counts -= tl.where(bins == 0, invalid_count, 0)
        else:
            experts = tl.load(topk_idx_ptr + positions)
            keys = (experts << KEY_SHIFT) | lanes
            sorted_keys = tl.sort(keys, dim=0)
            tl.store(temporary_keys_ptr + positions, sorted_keys)
            counts = tl.histogram(experts, 256)

    local_starts = tl.cumsum(counts, axis=0) - counts
    packed_metadata = counts | (local_starts << METADATA_SHIFT)
    tl.store(metadata_ptr + chunk * 256 + bins, packed_metadata)


@triton.jit
def _finalize_sorted_kernel(
    temporary_keys_ptr,
    metadata_ptr,
    expert_offsets_ptr,
    sorted_indices_ptr,
    n_elements,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KEY_SHIFT: tl.constexpr,
    METADATA_SHIFT: tl.constexpr,
    FULL_CHUNKS: tl.constexpr,
):
    chunk = tl.program_id(0)
    expert_bins = tl.arange(0, 256)

    total_counts = tl.zeros((256,), dtype=tl.int32)
    preceding_counts = tl.zeros((256,), dtype=tl.int32)

    for source_chunk in tl.range(0, N_CHUNKS):
        packed = tl.load(metadata_ptr + source_chunk * 256 + expert_bins)
        counts = packed & (BLOCK_SIZE * 2 - 1)
        total_counts += counts
        preceding_counts += tl.where(source_chunk < chunk, counts, 0)

    own_metadata = tl.load(metadata_ptr + chunk * 256 + expert_bins)

    cumulative_counts = tl.cumsum(total_counts, axis=0)
    expert_starts = cumulative_counts - total_counts

    if chunk == 0:
        tl.store(expert_offsets_ptr, 0)
        tl.store(expert_offsets_ptr + expert_bins + 1, cumulative_counts)

    lanes = tl.arange(0, BLOCK_SIZE)
    slots = chunk * BLOCK_SIZE + lanes

    if FULL_CHUNKS:
        sorted_keys = tl.load(temporary_keys_ptr + slots)
    else:
        if chunk == N_CHUNKS - 1:
            mask = slots < n_elements
            sorted_keys = tl.load(
                temporary_keys_ptr + slots,
                mask=mask,
                other=0,
            )
        else:
            sorted_keys = tl.load(temporary_keys_ptr + slots)

    experts = sorted_keys >> KEY_SHIFT
    local_positions = sorted_keys & (BLOCK_SIZE - 1)
    local_starts = own_metadata >> METADATA_SHIFT

    starts_for_tokens = tl.gather(expert_starts, experts, axis=0)
    preceding_for_tokens = tl.gather(preceding_counts, experts, axis=0)
    local_starts_for_tokens = tl.gather(local_starts, experts, axis=0)

    destinations = (
        starts_for_tokens
        + preceding_for_tokens
        + lanes
        - local_starts_for_tokens
    )
    token_indices = chunk * BLOCK_SIZE + local_positions

    if FULL_CHUNKS:
        tl.store(sorted_indices_ptr + destinations, token_indices)
    else:
        if chunk == N_CHUNKS - 1:
            mask = slots < n_elements
            tl.store(
                sorted_indices_ptr + destinations,
                token_indices,
                mask=mask,
            )
        else:
            tl.store(sorted_indices_ptr + destinations, token_indices)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    num_tokens = flat.numel()

    if num_tokens == 0:
        return (
            torch.empty((0,), dtype=torch.int32, device=flat.device),
            torch.zeros((257,), dtype=torch.int32, device=flat.device),
        )

    block_size = triton.next_power_of_2(num_tokens)

    if block_size <= 2048:
        storage = torch.empty(
            (num_tokens + 257,),
            dtype=torch.int32,
            device=flat.device,
        )
        sorted_token_indices = storage[:num_tokens]
        expert_offsets = storage[num_tokens:]

        if block_size <= 128:
            num_warps = 1
        elif block_size <= 1024:
            num_warps = 2
        elif num_tokens <= 1792:
            num_warps = 2
        else:
            num_warps = 4

        _composite_sort_kernel[(1,)](
            flat,
            sorted_token_indices,
            expert_offsets,
            num_tokens,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return sorted_token_indices, expert_offsets

    chunk_size = 512
    key_shift = 9
    metadata_shift = 10
    num_chunks = triton.cdiv(num_tokens, chunk_size)
    metadata_size = num_chunks * 256
    full_chunks = (num_tokens & (chunk_size - 1)) == 0

    storage = torch.empty(
        2 * num_tokens + metadata_size + 257,
        dtype=torch.int32,
        device=flat.device,
    )

    sorted_token_indices = storage[:num_tokens]
    temporary_keys = storage[num_tokens:2 * num_tokens]
    metadata = storage[
        2 * num_tokens:2 * num_tokens + metadata_size
    ]
    expert_offsets = storage[
        2 * num_tokens + metadata_size:
    ]

    _local_sort_kernel[(num_chunks,)](
        flat,
        temporary_keys,
        metadata,
        num_tokens,
        N_CHUNKS=num_chunks,
        BLOCK_SIZE=chunk_size,
        KEY_SHIFT=key_shift,
        METADATA_SHIFT=metadata_shift,
        FULL_CHUNKS=full_chunks,
        num_warps=4,
    )
    _finalize_sorted_kernel[(num_chunks,)](
        temporary_keys,
        metadata,
        expert_offsets,
        sorted_token_indices,
        num_tokens,
        N_CHUNKS=num_chunks,
        BLOCK_SIZE=chunk_size,
        KEY_SHIFT=key_shift,
        METADATA_SHIFT=metadata_shift,
        FULL_CHUNKS=full_chunks,
        num_warps=8,
    )

    return sorted_token_indices, expert_offsets