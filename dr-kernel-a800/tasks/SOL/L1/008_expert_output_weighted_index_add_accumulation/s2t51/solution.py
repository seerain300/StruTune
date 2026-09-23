import torch
import triton
import triton.language as tl


@triton.jit
def _index_add_rows_kernel(
    out_ptr,         # *half (final_hidden_states clone)
    expert_ptr,      # *half (expert_outputs)
    indices_ptr,     # *int32 (token_indices)
    N,               # int32: number of selected tokens
    H,               # int32: hidden size
    BLOCK: tl.constexpr,  # chunk size for hidden dimension, e.g., 256
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # Iterate over hidden columns in chunks of BLOCK, with masking for tail
    for off in range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < H

        # Load expert vector chunk (row pid, columns cols)
        expert_row_ptr = expert_ptr + pid * H
        vals = tl.load(expert_row_ptr + cols, mask=mask, other=0.0)

        # Compute output row pointer: out_ptr is [BATCH*SEQ, H], row idx
        out_row_ptr = out_ptr + idx * H
        out_vec_ptr = out_row_ptr + cols

        # Atomically add into output
        tl.atomic_add(out_vec_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure dtype matches the reference (bfloat16)
        if final_hidden_states.dtype != torch.bfloat16:
            final_hidden_states = final_hidden_states.to(torch.bfloat16)
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)

        # Prepare output as a clone of the original buffer (to match reference behavior)
        output = final_hidden_states.clone()

        # Ensure contiguity and correct dtypes/devices
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        N = expert_outputs.shape[0]  # number of selected tokens
        H = expert_outputs.shape[1]  # hidden size

        # Launch Triton kernel: one program per row
        # Using BLOCK=256 ensures we cover all hidden columns via masked loop.
        grid = (N,)
        # Heuristic: choose num_warps based on N to balance overhead and occupancy
        num_warps = 4 if N >= 1024 else 2
        _index_add_rows_kernel[grid](
            output,               # out_ptr
            expert_outputs,       # expert_ptr
            token_indices,        # indices_ptr
            N,                    # N
            H,                    # H
            BLOCK=256,            # chunk size for hidden dimension
            num_warps=num_warps,  # heuristic
            num_stages=2,         # pipelining
        )

        return output


def run(*args):
    return ModelNew()(*args)
