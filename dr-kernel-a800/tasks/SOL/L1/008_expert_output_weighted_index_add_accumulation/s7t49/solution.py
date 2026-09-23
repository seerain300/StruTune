import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_per_row_chunked_kernel(
    out_ptr,        # *bf16, pointer to out tensor [N, H], contiguous
    src_ptr,        # *bf16, pointer to src tensor [M, H], contiguous
    indices_ptr,    # *int32, pointer to indices tensor [M], contiguous
    M,              # int32, number of source rows
    N,              # int32, number of rows in out (batch_seq_len)
    H,              # int32, hidden size
    BLOCK_SIZE: tl.constexpr,  # chunk size along H (compile-time constant)
):
    # One program per source row
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Destination row index in out
    dst = tl.load(indices_ptr + pid)  # int32
    if dst < 0 or dst >= N:
        return  # safety (shouldn't happen with provided inputs)

    # Base pointers for this destination row
    out_row_ptr = out_ptr + dst * H
    src_row_ptr = src_ptr + pid * H

    # Process hidden dimension in chunks of BLOCK_SIZE
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load source chunk (masked for tail)
        vals = tl.load(src_row_ptr + offs, mask=mask, other=0.0)  # bfloat16
        # Atomic add into output
        tl.atomic_add(out_row_ptr + offs, vals, mask=mask)
        start += BLOCK_SIZE


def _pick_block_size(H: int) -> int:
    # Simple heuristic: use 256 for larger H, else 128
    return 256 if H >= 256 else 128


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Enforce dtypes and contiguity
        if final_hidden_states.dtype != torch.bfloat16:
            final_hidden_states = final_hidden_states.to(torch.bfloat16)
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)
        # Ensure contiguity
        out = final_hidden_states.contiguous()
        src = expert_outputs.contiguous()
        idx = token_indices.to(torch.int32).contiguous()

        # Shapes
        M = src.shape[0]  # number of selected tokens
        N = out.shape[0]  # batch_seq_len
        H = out.shape[1]

        # Sanity checks
        assert idx.numel() == M, "token_indices length must match num_selected_tokens"
        # Optional: ensure indices are within valid range
        # This can be costly; inputs from get_inputs are valid, but we keep a guard
        # if torch.any((idx < 0) | (idx >= N)):
        #     raise RuntimeError("Invalid token_indices out of range")

        # Launch Triton kernel: one program per source row
        BLOCK_SIZE = _pick_block_size(H)
        grid = (M,)
        scatter_add_per_row_chunked_kernel[grid](
            out, src, idx, M, N, H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,
            num_stages=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
