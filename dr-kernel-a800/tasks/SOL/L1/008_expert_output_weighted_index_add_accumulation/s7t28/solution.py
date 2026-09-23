import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_kernel(
    out_ptr,        # *bf16, shape (N, H), contiguous
    src_ptr,        # *bf16, shape (M, H), contiguous
    indices_ptr,    # *int32, shape (M,), contiguous
    M,              # int32, number of source rows to scatter
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
):
    pid = tl.program_id(axis=0)
    # One program per row i in [0, M)
    if pid >= M:
        return

    # Load destination index (row in out) for this source row
    dst = tl.load(indices_ptr + pid)

    # Guard: ensure dst is within [0, N). The reference guarantees indices < N,
    # but keeping this guard avoids undefined behavior in edge cases.
    if dst < 0 or dst >= N:
        return

    # Accumulate across hidden dimension
    # For each j in [0, H): out[dst, j] += src[pid, j]
    # We do one atomic add per element to correctly handle duplicates.
    # Note: Triton loop with runtime bounds is acceptable here.
    for j in range(0, H):
        out_offset = dst * H + j
        src_offset = pid * H + j
        val = tl.load(src_ptr + src_offset)
        tl.atomic_add(out_ptr + out_offset, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match the original behavior (index_add_ semantics on a fresh buffer)
        out = final_hidden_states.clone()

        # Ensure tensors are on CUDA and contiguous
        # (get_inputs should place them on device already; we still enforce contiguity)
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Triton prefers int32 for indices
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch one program per source row
        M = expert_outputs.shape[0]
        N = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]

        grid = (M,)
        scatter_add_per_row_kernel[grid](
            out, expert_outputs, token_indices, M, N, H,
            num_warps=1,  # keep simple and robust
            num_stages=1,
        )

        return out


def run(*args):
    return ModelNew()(*args)
