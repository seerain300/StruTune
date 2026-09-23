import torch
import triton
import triton.language as tl


@triton.jit
def detect_duplicates_atomic_kernel(
    indices_ptr,       # *int64, shape [M]
    N: tl.constexpr,   # number of rows in out and number of indices
    has_dup_ptr,       # *int32, scalar flag at [0]
    BLOCK_SIZE: tl.constexpr,
):
    # We'll do a simple O(N^2) duplicate detection:
    # For each i from 0..N-1, compare indices[i] to all previous j in [0..i-1].
    # If any match, set has_dup = 1.
    # We only need one program to do this scan; grid = (1,)
    # Note: This kernel is intentionally simple and does not modify out/expert.
    # We read indices and write a scalar flag.
    # Note: Triton allows scalar loops; we avoid tl.arange over pointers here.
    for i in range(N):
        # Load current index
        idx_i = tl.load(indices_ptr + i).to(tl.int32)
        # Compare with all previous indices
        for j in range(i):
            idx_j = tl.load(indices_ptr + j).to(tl.int32)
            if idx_i == idx_j:
                tl.store(has_dup_ptr, tl.full((), 1, tl.int32))
                return
    # No duplicates found
    tl.store(has_dup_ptr, tl.full((), 0, tl.int32))


@triton.jit
def scatter_add_rows_atomic_kernel(
    out_ptr,            # *bf16, shape [N, H], contiguous
    expert_ptr,         # *bf16, shape [M, H], contiguous
    indices_ptr,        # *int64 (PyTorch long), shape [M]
    N: tl.constexpr,    # rows in out/expert
    M: tl.constexpr,    # number of expert outputs
    H: tl.constexpr,    # hidden_size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        row = row_start + i
        if row >= N:
            break
        idx = tl.load(indices_ptr + row).to(tl.int32)
        # Atomic add per hidden dimension
        for j in range(H):
            src_offset = row * H + j
            dst_offset = idx * H + j
            src_val = tl.load(expert_ptr + src_offset)
            tl.atomic_add(out_ptr + dst_offset, src_val)


@triton.jit
def scatter_rows_noatomic_kernel(
    out_ptr,            # *bf16, shape [N, H], contiguous
    expert_ptr,         # *bf16, shape [M, H], contiguous
    indices_ptr,        # *int64 (PyTorch long), shape [M]
    N: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        row = row_start + i
        if row >= N:
            break
        idx = tl.load(indices_ptr + row).to(tl.int32)
        # Copy expert row into destination row (no atomics)
        for j in range(H):
            src_offset = row * H + j
            dst_offset = idx * H + j
            val = tl.load(expert_ptr + src_offset)
            tl.store(out_ptr + dst_offset, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized forward that performs:
            out = final_hidden_states.clone()
            out.index_add_(dim=0, index=token_indices, source=expert_outputs)
        Entry point: ModelNew.forward
        """
        # Ensure CUDA tensors and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton"
        out = final_hidden_states.clone()
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        N = final_hidden_states.shape[0]
        M = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]
        assert expert_outputs.shape[1] == H, "expert_outputs second dim must match hidden_size"
        assert token_indices.shape[0] == M, "token_indices length must equal number of expert outputs"

        # Choose a moderate BLOCK_SIZE to balance occupancy and simplicity
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        # Detect duplicates inside Triton to avoid host-side PyTorch ops
        has_dup_ptr = torch.empty((), dtype=torch.int32, device=token_indices.device)
        detect_duplicates_atomic_kernel[(1,)](
            token_indices,
            N=N,
            has_dup_ptr=has_dup_ptr,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,
            num_stages=1,
        )
        has_duplicates = bool(has_dup_ptr.item())

        if has_duplicates:
            scatter_add_rows_atomic_kernel[grid](
                out, expert_outputs, token_indices,
                N=N, M=M, H=H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        else:
            scatter_rows_noatomic_kernel[grid](
                out, expert_outputs, token_indices,
                N=N, M=M, H=H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=2,
            )

        return out


def run(*args):
    return ModelNew()(*args)
