import torch
import triton
import triton.language as tl


@triton.jit
def atomic_add_tokens_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H], contiguous
    src_ptr,        # *bf16, pointer to src tensor [M, H], contiguous
    indices_ptr,    # *int32, pointer to indices tensor [M], contiguous
    M,              # int32, number of source rows (tokens)
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    VEC: tl.constexpr,  # vector width along H (compile-time constant)
):
    # One program per source row/token
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return  # safety if indices are out of bounds (shouldn't happen with provided inputs)

    # Base pointers for this token row in src and destination row in out
    row_src_ptr = src_ptr + pid * H
    row_out_ptr = out_ptr + dst * H

    # Process hidden dimension in vector chunks of size VEC
    start = 0
    while start < H:
        offs = start + tl.arange(0, VEC)
        mask = offs < H
        # Load a chunk from src[i, offs]
        val = tl.load(row_src_ptr + offs, mask=mask, other=0.0)  # bf16
        # Atomic add into out[dst, offs]
        tl.atomic_add(row_out_ptr + offs, val, mask=mask)
        start += VEC


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized atomic accumulation:
        out[token_indices[i]] += expert_outputs[i] for all i
        with out shape [batch_seq_len, hidden_size].
        """
        # Ensure inputs are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors"
        out = final_hidden_states.clone()  # preserve original semantics: return updated buffer

        M = expert_outputs.shape[0]
        N = out.shape[0]
        H = out.shape[1]

        # Make sure tensors are contiguous and dtypes are bf16, indices int32
        out = out.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous().to(torch.int32)

        # Choose vector width and warps based on H for performance
        if H >= 2048:
            VEC = 512
            num_warps = 8
        elif H >= 512:
            VEC = 256
            num_warps = 4
        else:
            VEC = 128
            num_warps = 2

        # Launch Triton kernel: one program per token (row)
        grid = (M,)
        atomic_add_tokens_kernel[grid](
            out, expert_outputs, token_indices, M, N, H,
            VEC=VEC,
            num_warps=num_warps,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
