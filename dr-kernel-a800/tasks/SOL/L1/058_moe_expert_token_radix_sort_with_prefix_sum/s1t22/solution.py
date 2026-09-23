# Triton kernels
import triton
import triton.language as tl


# 1) Histogram via atomic add: counts_ptr[in[k]] += 1 for each element
@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load flat values; other=0 ensures invalid lanes use 0 (ignored by mask)
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Atomically increment counts at the loaded indices
    # counts_ptr is int32
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# 2) Scan to compute inclusive le_counts and exclusive lt_counts across NUM_EXPERTS (256)
@triton.jit
def scan_counts_kernel(counts_ptr, le_ptr, lt_ptr, NUM_EXPERTS: tl.constexpr):
    # Single-program scan across 256 elements
    # le_counts[0] = 0
    le = 0
    for k in range(NUM_EXPERTS):
        c = tl.load(counts_ptr + k)  # int32
        le += c
        tl.store(le_ptr + k, le)      # inclusive count at k
        tl.store(lt_ptr + k, le - c)  # exclusive count at k


# 3) Compute stable argsort permutation in Triton: write sorted_token_indices
@triton.jit
def compute_out_pos(flat_ptr, out_ptr, counts_ptr, le_ptr, lt_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values and base positions
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    base = tl.load(le_ptr + vals)  # inclusive count at val; int32

    # Determine if there are any duplicates of val before this position
    # We need lt_counts[val]; if lt_counts[val] > 0, then duplicates exist and we shift later ones back by 1.
    lt_val = tl.load(lt_ptr + vals)  # exclusive count at val; int32

    # Build mask of valid lanes (offsets < N) and compute stable tie-break: subtract 1 if duplicates exist and lt_val > 0
    # duplicate = 1 if lt_val > 0 else 0
    duplicate = tl.where(lt_val > 0, 1, 0)

    pos = base - duplicate  # int32
    # Store positions for valid lanes
    tl.store(out_ptr + offsets, pos, mask=mask)


# 4) Compute expert_offsets = inclusive counts per expert: offsets[0]=0, offsets[1:]=le_counts
@triton.jit
def inclusive_counts_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    # offsets_ptr is int32, length NUM_EXPERTS+1
    tl.store(offsets_ptr, 0)  # offsets[0] = 0
    le = 0
    for k in range(NUM_EXPERTS):
        c = tl.load(counts_ptr + k)
        le += c
        tl.store(offsets_ptr + 1 + k, le)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, block_hist: int = 1024, block_out: int = 1024):
        super().__init__()
        self.num_experts = num_experts
        self.block_hist = block_hist
        self.block_out = block_out

    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D and ensure int32
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = flat.numel()

        # Prepare output buffers
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        le_counts = torch.empty(self.num_experts, dtype=torch.int32, device=flat.device)
        lt_counts = torch.empty(self.num_experts, dtype=torch.int32, device=flat.device)
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)

        # 1) Histogram of flat values via atomic add
        grid_hist = (triton.cdiv(N, self.block_hist),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, self.block_hist)

        # 2) Scan to get inclusive le_counts and exclusive lt_counts
        # Single program
        scan_counts_kernel[(1,)](counts, le_counts, lt_counts, self.num_experts)

        # 3) Compute stable permutation using counts, le_counts, lt_counts
        grid_out = (triton.cdiv(N, self.block_out),)
        compute_out_pos[grid_out](flat, sorted_token_indices, counts, le_counts, lt_counts, N, self.block_out)

        # 4) Build expert_offsets = inclusive counts per expert (offsets[0]=0, offsets[1:]=le_counts)
        inclusive_counts_kernel[(1,)](counts, expert_offsets, self.num_experts)

        # Return sorted_token_indices (permutation) and expert_offsets
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
