import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_by_bucket(a_ptr, out_ptr, N: tl.constexpr, num_buckets: tl.constexpr):
    """
    Compute stable argsort permutation 'out' of length N for flattened array 'a_ptr'
    (assumed to have integer values in [0, num_buckets-1]).

    For each bucket b in 0..num_buckets-1:
      - Find all indices i where a[i] == b.
      - Among those, compute pos_i = number of earlier indices j in the same bucket with j < i.
        This preserves original order for ties.
      - Place i at out[start + pos_i], then increment start.
    """
    start = tl.zeros((), dtype=tl.int64)
    # Iterate over each bucket b; num_buckets is constexpr, so Triton can unroll.
    for b in range(num_buckets):
        # Step 1: mark indices in this bucket
        # We need a vector of booleans indicating a[i] == b.
        # Note: Triton doesn't have direct vector equality check on int32 vs int64 well; we keep int64 indexing.
        # We build a mask per bucket using a loop over i. However, Triton can't index vectors by dynamic i here easily,
        # so we rely on a two-phase: compute count of this bucket, then assign positions.
        # Phase A: count how many elements equal b
        count = tl.zeros((), dtype=tl.int32)
        for i in range(N):
            vi = tl.load(a_ptr + i)
            # vi and b are int32; equality works
            if vi == b:
                count += 1

        # Phase B: assign positions to each i in bucket
        # We iterate i over N again; for each i in bucket, compute its position pos among bucket members
        # (number of earlier indices in the bucket).
        for i in range(N):
            vi = tl.load(a_ptr + i)
            if vi == b:
                # Compute pos = number of earlier indices j in bucket with j < i
                pos = tl.zeros((), dtype=tl.int64)
                for j in range(N):
                    vj = tl.load(a_ptr + j)
                    if vj == b and j < i:
                        pos += 1
                # Place i at out[start + pos] and increment start
                dest = start + pos
                tl.store(out_ptr + dest, tl.full((), i, dtype=tl.int64))
                start += 1


@triton.jit
def _histogram_kernel(a_ptr, out_ptr, N: tl.constexpr, num_buckets: tl.constexpr):
    """
    Compute histogram of values in 'a_ptr' (int32), counts in 'out_ptr' (int64).
    """
    for i in range(N):
        vi = tl.load(a_ptr + i)
        # atomic add into bucket vi
        tl.atomic_add(out_ptr + vi, 1)


@triton.jit
def _inclusive_scan_prefix_sum(in_ptr, out_ptr, n: tl.constexpr):
    """
    Perform inclusive prefix sum: out[i] = sum_{j=0..i} in[j], with in[0..n-1], out[0..n].
    We initialize out[0] = in[0], and then perform iterative doubling scan.
    """
    # in_ptr: [0..n-1], out_ptr: [0..n]
    # Start with base cases
    # Note: Triton kernels cannot read/write beyond n; we'll implement scan in-place in out_ptr[1..]
    # out[0] = in[0]
    # For i in 0..n-1: out[i+1] = out[i] + in[i]
    # However, Triton does not provide direct vectorized in-place scan; we implement iterative doubling manually.
    # This kernel is invoked with grid (1,), using n as constexpr.

    # First pass: out[0] = in[0]
    # We can't directly read in_ptr[0] here; instead, the host code ensures offsets[0] = 0.

    # Iterative doubling scan
    step = 1
    while step < n:
        # For each i, out[i + step] = out[i] + in[i]
        # We iterate i from 0 to n-step-1
        for i in range(0, n - step):
            out_val = tl.load(out_ptr + i)
            in_val = tl.load(in_ptr + i)
            tl.store(out_ptr + i + step, out_val + in_val)
        step *= 2


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Extract flattened topk_idx from args. ModelNew is invoked like Model, and get_inputs returns a dict with 'topk_idx'.
        # We assume inputs are passed similarly. Here, we mirror the original behavior: flatten topk_idx.
        # However, since the evaluation harness calls forward with the same signature as Model, and Model.forward returns run(*args),
        # we reimplement run here with Triton kernels.
        # We expect args[0] to be topk_idx tensor.
        topk_idx = args[0]
        # Ensure device is CUDA for Triton
        device = topk_idx.device
        # Flatten to 1D
        flat = topk_idx.reshape(-1)

        # Use int32 for values in kernels; ensure contiguous
        a = flat.to(torch.int32).contiguous()
        N = a.numel()

        # 1) Stable argsort permutation (indices) using Triton
        out = torch.empty(N, dtype=torch.int64, device=device)  # indices 0..N-1
        # Call kernel; pass N as constexpr so Triton can unroll loops
        _stable_argsort_by_bucket[(1,)](a, out, N=N, num_buckets=256)

        # 2) Histogram of values in flattened topk_idx (int32 -> int64 counts)
        histogram = torch.zeros(256, dtype=torch.int64, device=device)
        _histogram_kernel[(1,)](a, histogram, N=N, num_buckets=256)

        # 3) Prefix sum for expert offsets (length 257)
        offsets = torch.empty(257, dtype=torch.int64, device=device)
        # Initialize offsets[0] = 0 (already zero by default); inclusive scan will fill the rest
        # Note: _inclusive_scan_prefix_sum expects base at index 0 already set; but histogram starts from bucket 0.
        # We'll compute scan from histogram to offsets[1..], with offsets[0]=0.
        # We can do a simple loop to set inclusive sums manually for clarity:
        # offsets[0] = 0
        offsets[0] = 0
        # Fill offsets[1..] via iterative doubling inside Triton; here we call a small in-place kernel:
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, n=256)

        # Convert to int32 for consistency with original (sorted_token_indices and offsets)
        sorted_token_indices = out.to(torch.int32)
        offsets = offsets.to(torch.int32)

        # Return sorted_token_indices (1D of length N) and expert_offsets (1D of length 257)
        return sorted_token_indices, offsets