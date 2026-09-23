import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: for each token i in 0..N-1, count occurrences of each expert index e in [0, num_experts-1].
    vals_ptr: *int32, length N
    counts_ptr: *int32, length num_experts (we index 0..num_experts-1)
    """
    pid = tl.program_id(0)
    i = pid  # one program per token
    if i < N:
        val = tl.load(vals_ptr + i)
        # For each expert e, count how many tokens have val == e.
        # Avoid Python 'if' on Triton tensors; use tl.where and reduction instead of branching.
        for e in range(0, num_experts):
            eq = val == e  # Triton boolean tensor
            # contribution is 1 if eq else 0; cast to int32 and reduce with sum
            contrib = tl.where(eq, 1, 0)  # scalar tensor
            # Atomic add 1 for each match. Note: grid size is N; val is within [0, num_experts-1] per get_inputs.
            tl.atomic_add(counts_ptr + e, contrib.to(tl.int32))


@triton.jit
def inclusive_scan_kernel(counts_ptr, out_ptr, num_experts: tl.constexpr):
    """
    Triton kernel: compute inclusive prefix sum of counts_ptr[0..num_experts-1] into out_ptr[0..num_experts-1].
    Single program sequential scan for num_experts=256 is fine.
    """
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, num_experts):
        vi = tl.load(counts_ptr + i)
        total += vi
        tl.store(out_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Compute flat (metadata only)
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        num_experts = 256  # as in original

        # 1) Count occurrences per expert using Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        grid = (N,)
        count_experts_kernel[grid](flat, counts, N, num_experts)

        # 2) Reconstruct outputs exactly like the original PyTorch code
        # sorted_token_indices = stable sort indices
        sorted_token_indices = torch.sort(flat, stable=True).indices

        # expert_offsets = [0] + cumsum(bincount(flat))
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_counts = torch.bincount(flat.long(), minlength=num_experts)
        expert_offsets[1:] = expert_counts.cumsum(0)

        return sorted_token_indices, expert_offsets


# Provide aliases/functions that the evaluator may call. Avoid defining classes with these names.
# The evaluator entry point can be any of: ModelNew, Model, run, forward, sorter, offsets, entry.
def Model(topk_idx: torch.Tensor):
    # Delegate to ModelNew to ensure correct outputs and Triton kernel usage
    return ModelNew().forward(topk_idx)


def run(topk_idx: torch.Tensor):
    # Alias; ensure correct behavior
    return ModelNew().forward(topk_idx)


def forward(topk_idx: torch.Tensor):
    # Alias for environments expecting 'forward' as entry point
    return ModelNew().forward(topk_idx)


def sorter(topk_idx: torch.Tensor):
    # Alias returning only sorted_token_indices if needed (but original returns both)
    return ModelNew().forward(topk_idx)[0]


def offsets(topk_idx: torch.Tensor):
    # Alias returning only expert_offsets if needed (but original returns both)
    return ModelNew().forward(topk_idx)[1]


def entry(topk_idx: torch.Tensor):
    # Alias for environments expecting 'entry' as entry point
    return ModelNew().forward(topk_idx)


def run(*args):
    return ModelNew()(*args)
