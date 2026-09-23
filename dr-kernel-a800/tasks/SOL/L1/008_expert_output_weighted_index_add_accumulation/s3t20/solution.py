import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_program_fp32_kernel(
    out_ptr,         # *fp32, shape [M, H]
    expert_ptr,      # *fp32, shape [N, H]
    indices_ptr,     # *int32, shape [N]
    M: tl.constexpr, # number of rows in output
    N: tl.constexpr, # number of updates
    H: tl.constexpr, # hidden size
    BLOCK_H: tl.constexpr,
):
    # One program per update (i)
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Destination row index for this update
    idx = tl.load(indices_ptr + pid)  # int32

    # Accumulator for the full hidden row (vector of length BLOCK_H)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Process the hidden dimension in chunks of BLOCK_H
    for offs in range(0, H, BLOCK_H):
        cols = offs + tl.arange(0, BLOCK_H)
        mask = cols < H
        # Load the expert row segment as a vector
        v = tl.load(expert_ptr + pid * H + cols, mask=mask, other=0.0)
        # Accumulate into the vector
        acc += v

    # Store the accumulated vector into the destination row
    out_row_base = idx * H
    for offs in range(0, H, BLOCK_H):
        cols = offs + tl.arange(0, BLOCK_H)
        mask = cols < H
        tl.store(out_ptr + out_row_base + cols, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton requires CUDA tensors; ensure inputs are on CUDA
        if not (final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            raise RuntimeError("ModelNew.forward expects CUDA tensors for Triton execution.")

        # Ensure contiguity
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        M = final_hidden_states.shape[0]  # batch_seq_len
        N = expert_outputs.shape[0]       # num_selected_tokens
        H = final_hidden_states.shape[1]  # hidden_size

        # Accumulate in float32 for correctness (bfloat16 atomic_add not supported)
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Convert expert_outputs to float32
        expert_fp32 = expert_outputs.to(torch.float32)

        # Convert token_indices to int32
        indices_i32 = token_indices.to(torch.int32)

        # Choose BLOCK_H: use 1024 if possible; mask handles tails
        BLOCK_H = 1024 if H >= 1024 else (512 if H >= 512 else (256 if H >= 256 else 128))

        # Launch Triton kernel: one program per update
        grid = (N,)
        scatter_add_row_program_fp32_kernel[grid](
            out_fp32, expert_fp32, indices_i32,
            M=M, N=N, H=H, BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2
        )

        # Cast back to bfloat16 to match original output dtype
        result = out_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
