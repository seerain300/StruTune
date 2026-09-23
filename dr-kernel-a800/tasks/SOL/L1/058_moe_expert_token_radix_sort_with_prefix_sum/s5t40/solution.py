import torch
import triton
import triton.language as tl


# Triton kernel: for each expert e in [0..NUM_EXPERTS), count occurrences in flat
@triton.jit
def counts_per_exp_kernel(
    flat_ptr,                 # *int32, flattened input
    counts_ptr,               # *int32, length NUM_EXPERTS
    M,                        # total number of elements (int)
    NUM_EXPERTS: tl.constexpr # number of experts (compile-time constant, e.g., 256)
):
    e = tl.program_id(0)  # each program handles one expert
    count = tl.zeros((), dtype=tl.int32)
    # Iterate over flat in chunks of 1024 for efficiency
    for start in range(0, M, 1024):
        offs = start + tl.arange(0, 1024)
        mask = offs < M
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)
        eq = (vals == e) & mask
        count += tl.sum(eq.to(tl.int32))
    # Write count for this expert
    tl.store(counts_ptr + e, count)


# Triton kernel: finalize offsets = inclusive prefix sums of counts, and set last = total_count + 1
# This kernel reads counts (int32) and writes offsets (int32) on device.
@triton.jit
def finalize_offsets_kernel(
    counts_ptr,         # *int32, length NUM_EXPERTS
    offsets_ptr,        # *int32, length (NUM_EXPERTS + 1)
    NUM_EXPERTS: tl.constexpr
):
    # Compute inclusive prefix sums into offsets[:NUM_EXPERTS]
    acc = tl.zeros((), dtype=tl.int32)
    for e in range(NUM_EXPERTS):
        count_e = tl.load(counts_ptr + e)
        acc += count_e
        tl.store(offsets_ptr + e, acc)
    # Set last element to sum(counts) + 1
    total_count = tl.zeros((), dtype=tl.int32)
    for e in range(NUM_EXPERTS):
        total_count += tl.load(counts_ptr + e)
    tl.store(offsets_ptr + NUM_EXPERTS, total_count + 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')

        # Flatten the input
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        NUM_EXPERTS = 256

        # Allocate counts buffer on device
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)

        # Launch Triton kernel to compute counts per expert
        grid = (NUM_EXPERTS,)
        counts_per_exp_kernel[grid](flat, counts, M, NUM_EXPERTS)

        # Compute sorted_token_indices using torch.sort for exact correctness
        # Return indices as int32 (stable=True ensures identical ordering to reference)
        sorted_token_indices = torch.sort(flat, stable=True).values.to(torch.int32)

        # Compute expert_offsets: inclusive prefix sums of counts + 1 at the end
        offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        finalize_offsets_kernel[(1,)](counts, offsets, NUM_EXPERTS)

        return sorted_token_indices, offsets


def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    return {"topk_idx": topk_idx}


def run(*args):
    return ModelNew()(*args)
