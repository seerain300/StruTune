import torch
import triton
import triton.language as tl


# Optimized Triton kernel: one program per source row, atomically add the entire hidden vector.
@triton.jit
def scatter_add_full_row_kernel(output_ptr, expert_ptr, index_ptr,
                                N, H,
                                BLOCK_H: tl.constexpr):
    n = tl.program_id(0)
    if n >= N:
        return

    # Load target index (which row in output to add to)
    idx = tl.load(index_ptr + n)

    # Load the entire hidden vector for this source row
    h = tl.arange(0, BLOCK_H)  # BLOCK_H is constexpr = H
    vals = tl.load(expert_ptr + n * H + h)

    # Compute destination pointers and atomically add
    dest = idx * H + h
    tl.atomic_add(output_ptr + dest, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Triton-only forward that performs scatter-add: output[token_indices[i]] += expert_outputs[i]
        """
        # Clone input buffer to initialize output; we'll modify this clone.
        output = final_hidden_states.clone()

        # Shapes
        M = final_hidden_states.shape[0]
        N = expert_outputs.shape[0]
        H = final_hidden_states.shape[1]

        # Launch Triton kernel: one program per source row. BLOCK_H is set to H (compile-time).
        grid = (N,)

        # Choose num_warps based on H to improve occupancy.
        if H >= 1024:
            num_warps = 8
            num_stages = 3
        elif H >= 512:
            num_warps = 4
            num_stages = 2
        else:
            num_warps = 2
            num_stages = 2

        scatter_add_full_row_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK_H=H,  # process the entire hidden dimension in one vector
            num_warps=num_warps, num_stages=num_stages,
        )

        return output


def run(*args):
    return ModelNew()(*args)
