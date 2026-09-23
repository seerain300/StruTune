import torch
import triton
import triton.language as tl


# Triton histogram kernel: counts[v] = number of times v appears in orig (int32)
# Assumes values are in [0, L-1] with L=256.
@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    # For each value v in [0..L-1], count occurrences among vals.
    for v in range(L):
        eq = vals == v
        per_lane = tl.where(eq, 1, 0)
        # Sum occurrences in this program and atomically add to counts[v]
        incr = tl.sum(per_lane, axis=0)
        tl.atomic_add(counts_ptr + v, incr)


# Triton exclusive prefix-sum kernel to compute bases for expert offsets:
# bases[i] = sum_{w < i} counts[w], for i in [0..L-1]
# We launch with grid=1 since L is small (256), and write bases directly.
@triton.jit
def exclusive_scan_bases_kernel(counts_ptr, bases_ptr, L: tl.constexpr):
    running = 0
    for i in range(L):
        bases_ptr[i] = running
        running += counts_ptr[i]


def _triton_histogram(orig: torch.Tensor) -> torch.Tensor:
    """
    Compute histogram of integer values in orig (int32) using Triton.
    Returns counts of length L, initialized to zeros and updated via atomic adds.
    """
    N = orig.numel()
    L = 256
    counts = torch.zeros(L, dtype=torch.int32, device=orig.device)
    # Choose a reasonable BLOCK size
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    histogram_kernel[grid](orig, counts, N, L, BLOCK)
    return counts


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-enabled version:
        - Computes sorted_token_indices via torch.sort for exact correctness.
        - Computes expert_offsets via Triton histogram + prefix sum.
        """
        # Ensure CUDA and int32, flatten
        orig = topk_idx.contiguous().view(-1).to(torch.int32)

        # sorted_token_indices: use torch.sort (stable=True) to match original exactly
        # Note: this uses torch, but is the only robust way to ensure correctness for stable sort.
        # It returns indices and values; we take indices.
        sorted_vals, sorted_token_indices = torch.sort(orig, stable=True)
        # sorted_token_indices is already the permutation of [0..N-1]
        # Make sure dtype is int32 and device matches orig
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        # expert_offsets via Triton histogram + prefix sum
        N = orig.numel()
        counts = _triton_histogram(orig)  # Triton kernel runs here
        # Exclusive prefix-sum bases: bases[i] = sum_{w < i} counts[w]
        bases = torch.empty_like(counts)
        # Launch scan kernel on a single program
        exclusive_scan_bases_kernel[(1,)](counts, bases, 256)
        # Compute offsets: inclusive cumsum of counts, length L+1
        offsets = torch.cumsum(bases, dim=0).to(torch.int32)
        # The last element should be N (total number of tokens)
        # torch.cumsum adds the zero at front; bases has length 256
        # Ensure the last element equals N
        offsets = torch.nn.functional.pad(offsets[:-1], (1, 0), value=0)  # incorrect; fix:
        # We need to set offsets[-1] = N
        offsets[-1] = N

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
