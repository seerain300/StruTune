# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r23 score=-1.0 passed=False
I’m targeting the clear latency pattern in the supplied workloads: the current expert-major kernel scales with the number of 256-element loop iterations, with the largest cases paying heavily. I’ll keep the exact stable algorithm and test a larger scan tile selected from `num_tokens`, so the change stays localized to launch geometry and preserves output semantics.import torch
import triton
import triton.language as tl


@triton.jit
def _expert_major_sort_kernel(
    topk_idx_ptr,
    sorted_token_indices_ptr,
    expert_offsets_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    lane = tl.arange(0, BLOCK)

    before = tl.zeros((), tl.int32)
    count = tl.zeros((), tl.int32)

    for start in tl.range(0, n_elements, BLOCK, num_stages=1):
        offsets = start + lane
        mask = offsets < n_elements
        values = tl.load(topk_idx_ptr + offsets, mask=mask, other=0)

        before += tl.sum(
            ((values < expert) & mask).to(tl.int32), axis=0
        )
        count += tl.sum(
            ((values == expert) & mask).to(tl.int32), axis=0
        )

    tl.store(expert_offsets_ptr + expert, before)
    if expert == 255:
        tl.store(expert_offsets_ptr + 256, before + count)

    emitted = tl.zeros((), tl.int32)

    for start in tl.range(0, n_elements, BLOCK, num_stages=1):
        offsets = start + lane
        mask = offsets < n_elements
        values = tl.load(topk_idx_ptr + offsets, mask=mask, other=0)

        selected = (values == expert) & mask
        selected_i32 = selected.to(tl.int32)
        ranks = tl.cumsum(selected_i32, axis=0) - 1

        tl.store(
            sorted_token_indices_ptr + before + emitted + ranks,
            offsets,
            mask=selected,
        )
        emitted += tl.sum(selected_i32, axis=0)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    num_tokens = flat.numel()

    sorted_token_indices = torch.empty(
        num_tokens, dtype=torch.int32, device=flat.device
    )
    expert_offsets = torch.empty(
        257, dtype=torch.int32, device=flat.device
    )

    if num_tokens >= 32768:
        block = 512
        num_warps = 8
    else:
        block = 256
        num_warps = 4

    _expert_major_sort_kernel[(256,)](
        flat,
        sorted_token_indices,
        expert_offsets,
        num_tokens,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )

    return sorted_token_indices, expert_offsets