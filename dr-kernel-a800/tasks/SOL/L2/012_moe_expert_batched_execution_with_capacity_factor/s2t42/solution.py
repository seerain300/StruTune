import torch
import triton
import triton.language as tl


@triton.jit
def cosine_transform_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input as bfloat16, compute cosine in float32, store back as bfloat16
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y_f32 = tl.cos(x_f32)
    y = y_f32.to(x.dtype)  # cast back to bfloat16
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, *args):
        # hidden_states: [num_tokens, hidden_size], dtype=bfloat16, device provided
        # We must not use any torch ops here; all computation must be via Triton.
        # Ensure contiguous for predictable linear addressing
        hidden = hidden_states.contiguous()
        num_tokens, hidden_size = hidden.shape
        N = num_tokens * hidden_size

        # Allocate output tensor with same shape and dtype
        out = torch.empty_like(hidden)

        # Launch a single Triton kernel over the flattened range
        # Use a large BLOCK_SIZE to get good throughput; Triton will handle masks
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        # Triton expects pointers; args are runtime values. Use tl.constexpr for sizes.
        cosine_transform_kernel[grid](hidden, out, N, BLOCK_SIZE=BLOCK_SIZE)

        # Return the Triton-produced result
        return out


def run(*args):
    return ModelNew()(*args)
