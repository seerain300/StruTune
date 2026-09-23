import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_hidden_chunks_kernel(
    out_ptr,           # float32*  [batch_seq_len, hidden_size]
    token_indices_ptr, # int32*    [num_selected_tokens]
    expert_ptr,        # float32*  [num_selected_tokens, hidden_size]
    N,                 # int32: number of updates (num_selected_tokens)
    H,                 # int32: hidden_size
    BLOCK_H: tl.constexpr,  # chunk size along hidden dimension
):
    pid = tl.program_id(0)
    # Each program handles one update i = pid
    if pid >= N:
        return

    # Load the token index (row position to accumulate into)
    idx = tl.load(token_indices_ptr + pid)  # int32

    # Iterate over hidden dimension in chunks of BLOCK_H
    h = 0
    while h < H:
        offs_h = h + tl.arange(0, BLOCK_H)               # vector of hidden offsets
        mask_h = offs_h < H                              # mask for valid hidden positions

        # Compute linear offsets
        out_offsets = idx * H + offs_h                   # row-major addressing
        # Load v_chunk (expert_outputs[pid, h:h+BLOCK_H]) as float32
        v = tl.load(expert_ptr + pid * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

        # Atomic add into output
        tl.atomic_add(out_ptr + out_offsets, v, mask=mask_h)

        h += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA device."
        device = final_hidden_states.device

        # Prepare input tensors
        # Cast token_indices to int32 for Triton
        token_indices_i32 = token_indices.to(torch.int32)

        # Make tensors contiguous
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Clone and cast to float32 for accumulation (bf16 atomics not supported)
        out_fp32 = final_hidden_states.clone().to(torch.float32)
        expert_outputs_fp32 = expert_outputs.to(torch.float32)

        # Shapes
        N = expert_outputs_fp32.shape[0]  # num_selected_tokens
        H = final_hidden_states.shape[1]  # hidden_size

        # Heuristics for BLOCK_H and kernel launch
        if H >= 1024:
            BLOCK_H = 1024
            num_warps = 8
        elif H >= 512:
            BLOCK_H = 512
            num_warps = 4
        else:
            BLOCK_H = 256
            num_warps = 2

        # Grid: one program per update
        grid = (N,)

        scatter_add_hidden_chunks_kernel[grid](
            out_fp32, token_indices_i32, expert_outputs_fp32, N, H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
