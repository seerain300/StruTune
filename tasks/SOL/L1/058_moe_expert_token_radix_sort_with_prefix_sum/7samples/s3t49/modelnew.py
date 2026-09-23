import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rand_indices_kernel(
    out_ptr: tl.pointer_type(dtype=tl.int32, ndim=1),
    counter_ptr: tl.pointer_type(dtype=tl.int32, ndim=1),
    B: tl.int32,
    S: tl.int32,
    K: tl.int32,
    num_experts: tl.int32,
    seed: tl.int32,
):
    # Each program handles a chunk of elements
    pid = tl.program_id(0)
    start = pid * B
    i = start + tl.arange(0, B)
    mask = i < (S * K)

    # Derive a seed per element: simple deterministic mapping via atomic counter
    # Initialize local counter to start
    local_counter = start
    # Compute number of valid lanes
    num_valid = tl.sum(mask.to(tl.int32), axis=0)
    # We need per-lane seeds; use local counter and lane index
    # Note: Triton does not have tl.num_programs; we assume grid covers all elements
    # Compute per-lane seed = (local_counter + lane) * seed
    # But we need a single scalar seed per program for tl.rand; derive from local_counter
    per_lane = tl.arange(0, B)
    per_lane_masked = per_lane * mask
    # Atomic increment per valid lane to advance the global counter
    # Initialize counter if not set
    # However, we only need a scalar counter per program to derive the seed
    # We'll set the seed from the program id and local_counter
    # seed_program = (local_counter ^ seed) ensures uniqueness per program
    seed_program = (local_counter ^ seed) & 0xFFFFFFFF
    # Generate random float in [0, 1) per lane, but only for masked lanes
    rand_val = tl.rand(seed_program)
    # Map to [0, num_experts)
    val = rand_val * num_experts
    val = tl.floor(val).to(tl.int32)
    # Ensure in bounds
    val = tl.maximum(val, 0)
    val = tl.minimum(val, num_experts - 1)

    # Store to output where mask is true
    tl.store(out_ptr + i, val, mask=mask)


@triton.jit
def _stable_argsort_permutation_kernel(
    flat_ptr: tl.pointer_type(dtype=tl.int32, ndim=1),
    out_ptr: tl.pointer_type(dtype=tl.int32, ndim=1),
    N: tl.int32,
    BLOCK: tl.constexpr,
):
    # Each program handles BLOCK original indices
    pid = tl.program_id(0)
    start = pid * BLOCK
    i_vec = start + tl.arange(0, BLOCK)
    mask_i = i_vec < N

    # Load the value for each original index i
    # Note: we use out_ptr[i] as output permutation; we write i at out[rank]
    # We need to compute rank for each i
    # We'll scan all j and count less and tie. This is O(N^2) but robust.
    # Initialize rank vector
    rank_vec = tl.zeros([BLOCK], dtype=tl.int32)

    # Loop over all positions j; Triton supports while loops
    j = 0
    while j < N:
        # Load flat[j]
        a_j = tl.load(flat_ptr + j)
        # Compute comparisons against each i in this program
        # less: flat[j] < flat[i_vec]
        less = (a_j < tl.load(flat_ptr + i_vec))
        # tie: flat[j] == flat[i_vec] AND j < i
        tie = (a_j == tl.load(flat_ptr + i_vec)) & (j < i_vec)
        # Sum across vector lanes (convert bool to int32)
        count = (less | tie).to(tl.int32)
        # Add to rank for each i
        rank_vec += count
        j += 1

    # Now write each i to out[rank_vec]
    # out_ptr is 1D int32; we write i at position rank_vec
    # Ensure mask for valid i
    tl.store(out_ptr + rank_vec, i_vec, mask=mask_i)


@triton.jit
def _histogram_atomic_kernel(
    flat_ptr: tl.pointer_type(dtype=tl.int32, ndim=1),
    N: tl.int32,
    histogram_ptr: tl.pointer_type(dtype=tl.int32, ndim=1),
    num_buckets: tl.int32,
):
    # One program per element; atomic add to the bucket
    pid = tl.program_id(0)
    if pid >= N:
        return
    val = tl.load(flat_ptr + pid)
    # Ensure val in [0, num_buckets - 1]
    val = tl.maximum(val, 0)
    val = tl.minimum(val, num_buckets - 1)
    # Atomic add 1 to histogram bucket val
    tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum_kernel(
    input_ptr: tl.pointer_type(dtype=tl.int32, ndim=1),
    output_ptr: tl.pointer_type(dtype=tl.int32, ndim=1),
    num_buckets: tl.int32,
):
    # Single program performing inclusive scan across num_buckets
    # input_ptr length = num_buckets, output_ptr length = num_buckets + 1
    # We write inclusive prefix sums into output_ptr[1:], and set output_ptr[0] = 0 in host
    # Simple sequential scan
    acc = tl.zeros([1], dtype=tl.int32)
    for i in range(0, num_buckets):
        v = tl.load(input_ptr + i)
        acc += v
        tl.store(output_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is fixed at 256 per original setup
        self.num_experts = 256

    def forward(self, axes_and_scalars: dict, device: torch.device):
        # Generate random topk_idx using Triton
        batch_size = axes_and_scalars["batch_size"]
        seq_len = axes_and_scalars["seq_len"]
        num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
        N = batch_size * seq_len * num_experts_per_tok

        # Triton random indices: generate int32 in [0, num_experts)
        topk_idx = torch.empty((batch_size, seq_len, num_experts_per_tok), dtype=torch.int32, device=device)
        # Flatten for simplicity
        flat_out = torch.empty(N, dtype=torch.int32, device=device)
        # We need a counter to synchronize random calls; initialize to 0
        counter = torch.zeros(1, dtype=torch.int32, device=device)
        grid = (triton.cdiv(N, 1024),)  # Each program handles 1024 elements
        _triton_rand_indices_kernel[grid](
            flat_out,
            counter,
            1024,  # BLOCK size for program workload
            seq_len,
            num_experts_per_tok,
            self.num_experts,
            12345,  # seed for RNG
        )
        # Write back into the 3D tensor
        # Reshape flat_out to (batch_size, seq_len, num_experts_per_tok)
        topk_idx = flat_out.view(batch_size, seq_len, num_experts_per_tok)

        # Flatten to 1D int32 for argsort
        flat = topk_idx.contiguous().view(-1).to(torch.int32)

        # 1) Triton stable argsort permutation (int32 indices)
        out_i32 = torch.empty(N, dtype=torch.int32, device=device)
        _stable_argsort_permutation_kernel[(triton.cdiv(N, 1024),)](
            flat, out_i32, N, BLOCK=1024
        )
        sorted_token_indices = out_i32.to(torch.int64)  # match original dtype

        # 2) Triton histogram of expert IDs
        histogram = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        _histogram_atomic_kernel[(N,)](flat, N, histogram, self.num_experts)

        # 3) Triton inclusive prefix sum to produce expert_offsets (length self.num_experts + 1)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum_kernel[(1,)](histogram, offsets, self.num_experts)

        # Return results: sorted_token_indices (int64), expert_offsets (int32)
        return sorted_token_indices, offsets