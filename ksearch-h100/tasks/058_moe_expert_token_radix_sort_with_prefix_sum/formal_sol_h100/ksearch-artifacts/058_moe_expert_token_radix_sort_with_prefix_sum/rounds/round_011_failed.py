# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r11 score=-1.0 passed=False
I’m implementing the requested tiled path directly: per-tile expert histograms, a tile-prefix buffer, then tile-local stable sorting whose positions are translated through global expert bases. The returned permutation remains stable with respect to the flattened input assignment order.import torch
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
    offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_tokens

    expert_ids = tl.load(topk_ptr + offsets, mask=mask, other=0).to(tl.int32)
    counts_ptrs = tile_counts_ptr + tile_id * 256 + expert_ids
    tl.atomic_add(counts_ptrs, 1, mask=mask)


@triton.jit
def _stable_tile_scatter_kernel(
    topk_ptr,
    tile_prefix_ptr,
    expert_bases_ptr,
    sorted_token_indices_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tile_start = tile_id * BLOCK
    lane = tl.arange(0, BLOCK)
    offsets = tile_start + lane
    mask = offsets < num_tokens

    expert_ids = tl.load(topk_ptr + offsets, mask=mask, other=256).to(tl.int32)

    sort_keys = expert_ids * BLOCK + lane
    sorted_keys, sorted_lanes = tl.sort(sort_keys, return_indices=True)

    sorted_experts = sorted_keys // BLOCK
    valid = sorted_experts < 256

    tile_counts = tl.load(
        tile_prefix_ptr + tile_id * 256 + lane,
        mask=lane < 256,
        other=0,
    ).to(tl.int32)
    tile_exclusive_prefix = tl.cumsum(tile_counts, axis=0) - tile_counts

    local_prefix = tl.load(
        tile_prefix_ptr + tile_id * 256 + sorted_experts,
        mask=valid,
        other=0,
    ).to(tl.int32)
    expert_base = tl.load(
        expert_bases_ptr + sorted_experts,
        mask=valid,
        other=0,
    ).to(tl.int32)

    rank_in_tile = lane - local_prefix
    destination = expert_base + tile_exclusive_prefix[sorted_experts] + rank_in_tile

    source_index = tile_start + sorted_lanes
    values = source_index.to(tl.int32)

    tl.store(
        sorted_token_indices_ptr + destination,
        values,
        mask=valid & (source_index < num_tokens),
    )


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    num_experts = 256
    flat = topk_idx.reshape(-1)
    num_tokens = flat.numel()

    block_size = 256
    num_tiles = max(1, (num_tokens + block_size - 1) // block_size)

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

    tile_prefix = (
        torch.cumsum(tile_counts, dim=0, dtype=torch.int32) - tile_counts
    )

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
        tile_prefix,
        expert_offsets,
        sorted_token_indices,
        num_tokens,
        BLOCK=block_size,
    )

    return sorted_token_indices, expert_offsets