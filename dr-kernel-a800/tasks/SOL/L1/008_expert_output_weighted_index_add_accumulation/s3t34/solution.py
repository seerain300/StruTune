import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_atomic_fp32(
    out_ptr,            # *fp32, shape [batch_seq_len, H]
    indices_ptr,        # *i32,  shape [N]
    src_ptr,            # *fp32, shape [N, H]
    N: tl.constexpr,    # number of updates (batch_seq_len * num_experts_per_tok)
    H: tl.constexpr,    # hidden_size (columns)
    BLOCK_H: tl.constexpr,  # vectorization over hidden dim
):
    pid = tl.program_id(axis=0)  # one program per update
    if pid >= N:
        return

    # Load token index and compute base offset for output row
    idx = tl.load(indices_ptr + pid)
    base = idx * H

    # Iterate over hidden dimension in chunks of BLOCK_H
    for h in range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        # Load source vector for this update
        v = tl.load(src_ptr + pid * H + offs, mask=mask, other=0.0)  # fp32
        # Atomic add into output row
        dest_ptr = out_ptr + base + offs
        tl.atomic_add(dest_ptr, v, mask=mask)


def _pick_block_and_warps(hidden_size: int):
    # Heuristics tuned for typical sizes; keep robustness
    if hidden_size >= 1024:
        return 1024, 8
    elif hidden_size >= 512:
        return 512, 4
    else:
        return 256, 2


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA device"
        # Ensure contiguity
        final_hidden = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        batch_seq_len = final_hidden.shape[0]
        H = final_hidden.shape[1]
        N = expert_outputs.shape[0]

        # Output buffer in float32 for atomic accumulation
        out_fp32 = torch.zeros((batch_seq_len, H), dtype=torch.float32, device=final_hidden_states.device)

        # Cast indices to int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        # Cast expert_outputs to float32
        src_fp32 = expert_outputs.to(torch.float32)

        # Pick BLOCK_H and num_warps
        BLOCK_H, num_warps = _pick_block_and_warps(H)

        # Launch one program per update
        grid = (N,)
        scatter_add_rows_atomic_fp32[grid](
            out_fp32,
            indices_i32,
            src_fp32,
            N,
            H,
            BLOCK_H,
            num_warps=num_warps,
            num_stages=2,
        )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
