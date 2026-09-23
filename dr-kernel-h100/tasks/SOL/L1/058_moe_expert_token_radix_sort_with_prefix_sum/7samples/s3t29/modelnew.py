import torch

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: stable argsort of flat values into out (permutation indices).
# Each program handles one bucket 'bucket_id'. It scans through all indices i and places
# each element equal to 'bucket_id' at stable position based on number of earlier elements
# equal to 'bucket_id' seen so far. Since we iterate i in increasing order, earlier i come first.
if TRITON_AVAILABLE:
    @triton.jit
    def _stable_argsort_by_values_kernel(flat_ptr, out_ptr, N, bucket_id: tl.constexpr):
        # We assume flat_ptr points to int32 values and out_ptr to int32 indices.
        # Each thread/program handles one bucket b = bucket_id.
        # We'll iterate over i = 0..N-1, but Triton requires a static loop. Since N is runtime here,
        # we structure the kernel so that the inner loop iterates i over a vector, but we pass
        # runtime N via a while loop to ensure compatibility. Triton supports runtime while loops.
        i = 0
        while i < N:
            val = tl.load(flat_ptr + i)  # val is the bucket id
            # Only process elements equal to this bucket
            if val == bucket_id:
                # Compute stable position: count how many earlier indices j had flat[j] == bucket_id
                count_less = tl.zeros((), dtype=tl.int32)
                j = 0
                while j < i:
                    vj = tl.load(flat_ptr + j)
                    if vj == bucket_id:
                        count_less += 1
                    j += 1
                # Place i at out[count_less]
                tl.store(out_ptr + count_less, i)
            i += 1

    # Kernel: compute histogram of flat values (int32), one atomic add per element.
    @triton.jit
    def _histogram_kernel(flat_ptr, histogram_ptr, N, num_buckets: tl.constexpr):
        i = 0
        while i < N:
            val = tl.load(flat_ptr + i)
            # atomic add to histogram[val]
            tl.atomic_add(histogram_ptr + val, 1)
            i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is fixed to 256 as in the original run
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure we have the flattened 1D vector
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        device = flat.device
        N = flat.numel()

        # 1) Stable argsort using Triton kernel
        out = torch.empty(N, dtype=torch.int32, device=device)  # holds permutation indices 0..N-1
        # Launch one program per bucket b in 0..255
        if TRITON_AVAILABLE:
            for b in range(self.num_experts):
                _stable_argsort_by_values_kernel[(1,)](flat, out, N, bucket_id=b)
        else:
            # Fallback: if Triton not available, we cannot produce correct argsort without torch.
            # But evaluation environment should provide Triton; we keep the Triton path.
            # For robustness, we can attempt to mimic stable sort with torch, but it's not allowed here.
            # Therefore, we raise an error to indicate Triton is required.
            raise RuntimeError("Triton is required but not available.")

        # 2) Histogram of expert IDs (counts per bucket) using Triton
        histogram = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        if TRITON_AVAILABLE:
            _histogram_kernel[(1,)](flat, histogram, N, num_buckets=self.num_experts)

        # 3) Prefix sum to get expert_offsets (length num_experts + 1), starting at 0
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        # Inclusive prefix sums
        total = 0
        for p in range(self.num_experts):
            total += histogram[p]
            offsets[p + 1] = total

        # Return sorted_token_indices (1D of length N) and expert_offsets (1D of length 257)
        return out, offsets