import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_chunked_kernel(
    out_ptr,                # *fp32, shape [B, H]
    token_indices_ptr,      # *i32, shape [N]
    expert_outputs_ptr,     # *fp32, shape [N, H]
    N,                      # int32
    H,                      # int32
    BLOCK_H: tl.constexpr,  # e.g., 1024, 512, or 256
):
    # One program per update (scatter-add target)
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load target index for this update
    idx = tl.load(token_indices_ptr + pid).to(tl.int32)

    # Iterate over hidden dimension in chunks of BLOCK_H
    num_chunks = (H + BLOCK_H - 1) // BLOCK_H
    for c in range(0, num_chunks):
        h = c * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = h < H
        # Load the corresponding expert output chunk as float32
        v = tl.load(expert_outputs_ptr + pid * H + h, mask=mask, other=0.0).to(tl.float32)
        # Atomic add into the output row
        tl.atomic_add(out_ptr + idx * H + h, v, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on the same CUDA device
        assert final_hidden_states.device.type == "cuda" and expert_outputs.device.type == "cuda" and token_indices.device.type == "cuda", "All inputs must be on CUDA"

        B = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Accumulate in float32 for atomic_add support
        out_fp32 = final_hidden_states.clone().to(torch.float32)
        token_indices_i32 = token_indices.to(torch.int32).contiguous()
        expert_outputs_fp32 = expert_outputs.contiguous().to(torch.float32)

        # Choose BLOCK_H and num_warps based on hidden_size
        if H >= 1024:
            BLOCK_H = 1024
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 512
            num_warps = 4
        else:
            BLOCK_H = 256
            num_warps = 2

        # One program per update
        grid = (N,)

        scatter_add_chunked_kernel[grid](
            out_fp32,
            token_indices_i32,
            expert_outputs_fp32,
            N,
            H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=4,  # slightly higher stages to help pipelining
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
