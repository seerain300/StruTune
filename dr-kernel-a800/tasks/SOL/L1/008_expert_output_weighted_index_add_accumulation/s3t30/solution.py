import torch
import triton
import triton.language as tl


@triton.jit
def full_row_kernel(
    out_ptr,               # *fp32, shape [N_rows, H]
    token_indices_ptr,     # *i32, shape [N]
    expert_outputs_ptr,    # *fp32, shape [N, H]
    N,                     # int: number of updates
    N_rows,                # int: number of rows in output
    H,                     # int: hidden size
    T: tl.constexpr,       # updates per program
    BLOCK_H: tl.constexpr  # set to H for full-row (compile-time constant)
):
    pid = tl.program_id(0)  # one program handles T updates
    start = pid * T
    for k in range(T):
        pos = start + k
        mask_pos = pos < N
        # Load index for this update
        idx = tl.load(token_indices_ptr + pos, mask=mask_pos, other=0)  # i32
        row_base = idx * H
        offs = tl.arange(0, BLOCK_H)
        mask_h = offs < H  # safety; mask_pos ensures valid pos
        vals = tl.load(
            expert_outputs_ptr + pos * H + offs,
            mask=mask_pos & mask_h,
            other=0.0
        )
        tl.atomic_add(out_ptr + row_base + offs, vals, mask=mask_pos)


@triton.jit
def chunked_multi_update_kernel(
    out_ptr,               # *fp32, shape [N_rows, H]
    token_indices_ptr,     # *i32, shape [N]
    expert_outputs_ptr,    # *fp32, shape [N, H]
    N,                     # int
    N_rows,                # int
    H,                     # int
    T: tl.constexpr,       # updates per program
    BLOCK_H: tl.constexpr  # chunk size along hidden dim
):
    pid = tl.program_id(0)  # each program handles T updates
    start = pid * T
    for k in range(T):
        pos = start + k
        mask_pos = pos < N
        idx = tl.load(token_indices_ptr + pos, mask=mask_pos, other=0)  # i32
        row_base = idx * H
        # Iterate across hidden dim in chunks of BLOCK_H
        for h0 in range(0, H, BLOCK_H):
            offs = tl.arange(0, BLOCK_H)
            h_idx = h0 + offs
            mask_h = h_idx < H
            vals = tl.load(
                expert_outputs_ptr + pos * H + h_idx,
                mask=mask_pos & mask_h,
                other=0.0
            )
            tl.atomic_add(out_ptr + row_base + h_idx, vals, mask=mask_pos)


def _choose_kernel_params(H: int):
    # Heuristics for performance:
    # - For H<=1024 and H multiple of 1024, use full-row kernel to minimize atomics.
    # - For H multiple of 512, use chunked with BLOCK_H=512.
    # - Else, use chunked with BLOCK_H=256.
    if H <= 1024 and (H % 1024 == 0):
        return dict(BLOCK_H=1024, num_warps=8, num_stages=2, kernel='full_row', T=4)
    elif H % 512 == 0:
        return dict(BLOCK_H=512, num_warps=4, num_stages=2, kernel='chunked', T=4)
    else:
        return dict(BLOCK_H=256, num_warps=2, num_stages=2, kernel='chunked', T=4)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguity
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        N_rows = final_hidden_states.shape[0]
        H = final_hidden_states.shape[1]
        N = expert_outputs.shape[0]

        # Accumulate in float32 for atomic_add support
        out_fp32 = final_hidden_states.to(torch.float32).clone()

        # Triton params
        params = _choose_kernel_params(H)
        BLOCK_H = params['BLOCK_H']
        num_warps = params['num_warps']
        num_stages = params['num_stages']
        kernel = params['kernel']
        T = params.get('T', 4)

        # Grid: each program handles T updates
        grid = (triton.cdiv(N, T),)

        if kernel == 'full_row':
            full_row_kernel[grid](
                out_fp32, token_indices.to(torch.int32), expert_outputs.to(torch.float32),
                N, N_rows, H, T=T, BLOCK_H=BLOCK_H,
                num_warps=num_warps, num_stages=num_stages
            )
        else:
            chunked_multi_update_kernel[grid](
                out_fp32, token_indices.to(torch.int32), expert_outputs.to(torch.float32),
                N, N_rows, H, T=T, BLOCK_H=BLOCK_H,
                num_warps=num_warps, num_stages=num_stages
            )

        # Cast back to bfloat16 to match original API
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
