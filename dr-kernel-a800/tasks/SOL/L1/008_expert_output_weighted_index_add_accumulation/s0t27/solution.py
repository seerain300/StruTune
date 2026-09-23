import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_row_atomic_kernel(
    output_ptr,            # *fp16/bf16 pointer to output tensor [M, H]
    expert_outputs_ptr,    # *fp16/bf16 pointer to expert_outputs tensor [N, H]
    token_indices_ptr,     # *int32 pointer to token_indices tensor [N]
    N: tl.int32,           # number of source rows (N)
    H: tl.int32,           # number of hidden features (H)
    BLOCK_H: tl.constexpr, # tile size for H dimension
):
    pid_n = tl.program_id(0)  # program processes one source row n = pid_n
    if pid_n >= N:
        return

    # Load the destination row index for this source row
    # token_indices are int64 in PyTorch; cast to int32 for Triton math
    idx = tl.load(token_indices_ptr + pid_n)
    idx = idx.to(tl.int32)

    # Base pointer for this row in output: output[idx, :]
    # We'll compute addresses as idx * H + h_offsets
    start = 0
    # Process H in chunks of BLOCK_H
    while start < H:
        h_offsets = start + tl.arange(0, BLOCK_H)  # vector of H offsets
        mask = h_offsets < H

        # Load vals from expert_outputs[n, h_offsets]
        vals = tl.load(expert_outputs_ptr + pid_n * H + h_offsets, mask=mask, other=0.0)

        # Compute output addresses and perform atomic add
        dest = idx * H + h_offsets
        tl.atomic_add(output_ptr + dest, vals, mask=mask)

        start += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-only forward that performs scatter-add:
          output[token_indices[i]] += expert_outputs[i]
        Input tensors:
          - final_hidden_states: (M, H) fp16/bf16
          - expert_outputs: (N, H) fp16/bf16
          - token_indices: (N,) int64
        """
        # Ensure device and dtype compatibility
        if final_hidden_states.device != expert_outputs.device or final_hidden_states.device != token_indices.device:
            raise RuntimeError("All tensors must be on the same device.")
        if token_indices.dtype != torch.long:
            token_indices = token_indices.to(torch.long)

        # Clone to preserve original buffer (like the original run)
        output = final_hidden_states.clone()

        M, H = output.shape
        N = expert_outputs.shape[0]

        # Triton grid: one program per source row
        grid = (N,)

        # Choose BLOCK_H and kernel config based on H
        if H >= 2048:
            BLOCK_H = 256
            num_warps = 8
            num_stages = 3
        elif H >= 1024:
            BLOCK_H = 256
            num_warps = 4
            num_stages = 3
        elif H >= 512:
            BLOCK_H = 128
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_H = 64
            num_warps = 2
            num_stages = 2

        # Triton kernel expects int32 indices for better performance; token_indices are long, we can pass as-is (Triton loads int64 and we cast)
        scatter_add_row_atomic_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
