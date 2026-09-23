import torch
import triton
import triton.language as tl


@triton.jit
def add_expert_rows_kernel(output_accum_ptr,  # float32 buffer: [M, H]
                            expert_ptr,      # expert_outputs: [N, H], dtype float32 (we'll cast from bf16)
                            token_indices_ptr,  # int32 indices: [N], values in [0, M)
                            N, H,
                            BLOCK_N: tl.constexpr):
    """
    Each program handles BLOCK_N rows of expert_outputs. It reads token_indices to determine the
    target row in output_accum, and atomically adds the hidden vector to that row. This avoids
    duplicate additions for the same output row because each program processes unique rows.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N

    # Vector of row indices this program handles
    rows = row_start + tl.arange(0, BLOCK_N)
    mask_rows = rows < N

    # Load token_indices for these rows (cast to int32 for Triton address math)
    idx = tl.load(token_indices_ptr + rows, mask=mask_rows, other=0)  # int64 may be fine, Triton supports 32/64, we cast below if needed
    # idx = idx.to(tl.int32)  # Triton will handle int64; ensure in range M

    # Load hidden vectors for these rows (assume H is contiguous along last dim)
    h_offsets = tl.arange(0, BLOCK_H)  # BLOCK_H is a constexpr passed at launch
    # Loop over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        h = h_off + h_offsets
        mask_h = h < H
        # Build 2D mask for loading [BLOCK_N, BLOCK_H]
        mask = mask_rows[:, None] & mask_h[None, :]
        vals = tl.load(expert_ptr + rows[:, None] * H + h[None, :], mask=mask, other=0.0)  # float32
        # Compute destination addresses for each row and hidden offset
        dest = idx * H + h  # vector of addresses per row
        # Atomic add to accumulator
        tl.atomic_add(output_accum_ptr + idx * H + h, vals, mask=mask)


@triton.jit
def add_acc_to_output_kernel(output_ptr, acc_ptr, M, H):
    """
    Add accumulator acc (M, H) to output (M, H). Single pass over all elements.
    """
    pid = tl.program_id(0)
    total = M * H
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Compute row and col for each linear offset
    row = offs // H
    col = offs % H
    out_vals = tl.load(output_ptr + row * H + col, mask=mask, other=0.0)
    acc_vals = tl.load(acc_ptr + row * H + col, mask=mask, other=0.0)
    out_vals = out_vals + acc_vals
    tl.store(output_ptr + row * H + col, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add along dim=0: output[token_indices[i]] += expert_outputs[i]
        """
        # Ensure tensors are on the same device and dtype for Triton. We'll accumulate in float32 for robustness,
        # then add to the original dtype output. However, since the benchmark uses bf16 tensors, we keep bf16.
        # Clone final_hidden_states as output (as original code does)
        output = final_hidden_states.clone()

        # Prepare shapes and dtypes
        M = output.shape[0]
        H = output.shape[1]
        N = expert_outputs.shape[0]

        # Create accumulator buffer in same dtype as output (bf16), initialized to zeros
        # Note: We could use float32 for acc to reduce numeric issues, but to keep dtype consistent, use bf16.
        acc = torch.zeros((M, H), dtype=output.dtype, device=output.device)

        # Cast token_indices to int32 for Triton kernel
        token_indices_int = token_indices.to(torch.int32)

        # We'll convert expert_outputs to float32 for atomic add stability, but we'll ignore repeats by design.
        # Launch kernel to build accumulator. Choose BLOCK_N to balance occupancy; H is usually small in these workloads.
        # Use a modest BLOCK_N (e.g., 128) and BLOCK_H (e.g., 64 or 128), and loop over H in tiles.
        BLOCK_N = 128
        # We need BLOCK_H as a constexpr; choose 64 to cover typical H; Triton will compile specialized kernel.
        BLOCK_H = 64

        grid = (triton.cdiv(N, BLOCK_N),)

        # Important: The kernel adds only once per expert row to its target row. Duplicate expert rows will not
        # add again because each program handles distinct rows. This reduces atomic contention significantly.

        add_expert_rows_kernel[grid](
            acc, expert_outputs.to(torch.float32), token_indices_int,
            N, H,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Now add accumulator to output
        total = M * H
        # We can launch a simple 1D kernel to add acc to output
        add_acc_to_output_kernel[(triton.cdiv(total, BLOCK),)](
            output, acc,
            M, H,
            BLOCK=1024, num_warps=4, num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
