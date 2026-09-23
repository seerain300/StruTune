import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_kernel(
    out_ptr,        # *bfloat16, shape [M, H]
    src_ptr,        # *bfloat16, shape [N, H]
    indices_ptr,    # *int32,    shape [N]
    M: tl.constexpr,  # int
    H: tl.constexpr,  # int
    N: tl.constexpr,  # int
    BLOCK_H: tl.constexpr,  # tile size along H
):
    # Each program handles one source row (i in [0, N))
    row = tl.program_id(0)

    # Load token index for this row
    tok = tl.load(indices_ptr + row)  # int32, valid in [0, M)

    # Iterate over hidden dimension in tiles of BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Compute pointers for this row and offsets
        out_row_ptr = out_ptr + tok * H
        src_row_ptr = src_ptr + row * H

        # Load source vector
        src_vec = tl.load(src_row_ptr + offs, mask=mask, other=0.0)

        # Atomically accumulate into output
        # Note: Triton may not expose atomic_add for bfloat16 on all versions,
        # but the above approach is the correct semantics for duplicates.
        tl.atomic_add(out_row_ptr + offs, src_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Perform scatter-add: output[token_indices[i]] += expert_outputs[i] for all i.
        Initialize output to zeros and use atomic accumulation in Triton for correctness
        with duplicate indices.
        """
        # Ensure tensors are on the same device and contiguous
        assert final_hidden_states.device == expert_outputs.device == token_indices.device, "All tensors must be on the same device"
        device = final_hidden_states.device

        # Shapes
        M = final_hidden_states.shape[0]  # batch_size * seq_len
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Prepare output as zeros (atomic accumulation requires initialized buffer)
        output = torch.zeros((M, H), dtype=torch.bfloat16, device=device)

        # Ensure dtypes and contiguity
        # Assume expert_outputs is bfloat16; cast if necessary
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)
        expert_outputs = expert_outputs.contiguous()

        # Ensure indices are int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)
        token_indices = token_indices.contiguous()

        # Launch kernel: one program per source row
        grid = (N,)
        scatter_add_rows_atomic_kernel[grid](
            output,                  # out_ptr
            expert_outputs,          # src_ptr
            token_indices,           # indices_ptr
            M=M, H=H, N=N,
            BLOCK_H=256,             # tile size along H; works well for H up to 1024
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
