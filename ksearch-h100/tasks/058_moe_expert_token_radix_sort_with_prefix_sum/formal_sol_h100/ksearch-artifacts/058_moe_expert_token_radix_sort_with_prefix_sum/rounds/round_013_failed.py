# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r13 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _count_experts_kernel(
    topk_ptr,
    tile_counts_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
):
    tile_id = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offsets = tile_id * BLOCK + lane
    mask = offsets < num_tokens

    expert_ids = tl.load(
        topk_ptr + offsets,
        mask=mask,
        other=0,
    ).to(tl.int32)

    tl.atomic_add(
        tile_counts_ptr + tile_id * 256 + expert_ids,
        1,
        mask=mask,
    )


@triton.jit
def _stable_tile_scatter_kernel(
    topk_ptr,
    tile_counts_ptr,
    tile_prefix_ptr,
    expert_offsets_ptr,
    sorted_token_indices_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
):
    tile_id = tl.program_id(0)
    lane = tl.arange(0, BLOCK)

    tile_start = tile_id * BLOCK
    source_offsets = tile_start + lane
    source_mask = source_offsets < num_tokens

    expert_ids = tl.load(
        topk_ptr + source_offsets,
        mask=source_mask,
        other=256,
    ).to(tl.int32)

    sort_keys = tl.sort(expert_ids * BLOCK + lane)
    sorted_experts = sort_keys // BLOCK
    sorted_lanes = sort_keys % BLOCK

    valid = sorted_experts < 256
    safe_experts = tl.minimum(sorted_experts, 255)

    local_counts = tl.load(
        tile_counts_ptr + tile_id * 256 + lane,
        mask=lane < 256,
        other=0,
    ).to(tl.int32)

    local_starts = tl.cumsum(local_counts, axis=0) - local_counts
    rank_in_tile = lane - tl.load(
        local_starts + safe_experts,
        mask=valid,
        other=0,
    ).to(tl.int32)

    previous_tile_count = tl.load(
        tile_prefix_ptr + tile_id * 256 + safe_experts,
        mask=valid,
        other=0,
    ).to(tl.int32)

    expert_base = tl.load(
        expert_offsets_ptr + safe_experts,
        mask=valid,
        other=0,
    ).to(tl.int32)

    destination = expert_base + previous_tile_count + rank_in_tile
    source_index = tile_start + sorted_lanes

    tl.store(
        sorted_token_indices_ptr + destination,
        source_index.to(tl.int32),
        mask=valid & (source_index < num_tokens),
    )


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    num_experts = 256
    flat = topk_idx.reshape(-1)
    num_tokens = flat.numel()

    block_size = 256
    num_tiles = triton.cdiv(num_tokens, block_size)

    tile_counts = torch.zeros(
        (num_tiles, num_experts),
        dtype=torch.int32,
        device=flat.device,
    )

    _count_experts_kernel[(num_tiles,)](
        flat,
        tile_counts,
        num_tokens,
        BLOCK=block_size,
    )

    tile_prefix = torch.cumsum(
        tile_counts,
        dim=0,
        dtype=torch.int32,
    ) - tile_counts

    expert_counts = tile_counts.sum(dim=0, dtype=torch.int32)

    expert_offsets = torch.empty(
        num_experts + 1,
        dtype=torch.int32,
        device=flat.device,
    )
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(
        expert_counts,
        dim=0,
        dtype=torch.int32,
    )

    sorted_token_indices = torch.empty(
        num_tokens,
        dtype=torch.int32,
        device=flat.device,
    )

    _stable_tile_scatter_kernel[(num_tiles,)](
        flat,
        tile_counts,
        tile_prefix,
        expert_offsets,
        sorted_token_indices,
        num_tokens,
        BLOCK=block_size,
    )

    return sorted_token_indices, expert_offsets