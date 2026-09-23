import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _compute_row_sumsq(hidden_ptr,          # *ptr to hidden [B, H] flattened as [B*H]
                        out_sumsq_ptr,      # *ptr to float32 output [B]
                        H: tl.constexpr,    # hidden size (compile-time)
                        BLOCK_SIZE: tl.constexpr  # chunk size (compile-time)
                        ):
    row = tl.program_id(0)
    base = row * H
    sumsq = 0.0  # fp32 accumulator
    # Iterate over columns in chunks of BLOCK_SIZE
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(hidden_ptr + base + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        # Ensure masked lanes don't contribute
        x32 = tl.where(mask, x32, 0.0)
        sumsq += tl.sum(x32 * x32, axis=0)
    tl.store(out_sumsq_ptr + row, sumsq)


@triton.jit
def _normalize_and_scale_elementwise(hidden_ptr,       # *ptr to hidden [B, H] flattened
                                     weight_ptr,      # *ptr to weight [H] flattened
                                     out_ptr,         # *ptr to output [B, H] flattened
                                     inv_rms_ptr,     # *ptr to inv_rms [B] (fp32)
                                     B,               # batch size (runtime int)
                                     H: tl.constexpr, # hidden size (compile-time)
                                     BLOCK_SIZE: tl.constexpr  # chunk size along columns
                                     ):
    # 2D launch: (row, col_block)
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    col_start = col_block * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    base_hidden = row * H
    base_out = row * H

    # Load x slice and cast to fp32
    x = tl.load(hidden_ptr + base_hidden + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Load weight slice and cast to fp32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    w32 = w.to(tl.float32)

    # Load inv_rms for this row (scalar fp32)
    inv_rms = tl.load(inv_rms_ptr + row)

    # Compute y = x * inv_rms * w
    y32 = x32 * inv_rms * w32

    # Store FP32 output
    tl.store(out_ptr + base_out + offs, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure CUDA tensors and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()
        B, H = hidden.shape
        assert H == HIDDEN_SIZE, f"Expected hidden size 4096, got {H}"

        # Flatten hidden for row-wise addressing
        hidden_flat = hidden.view(B * H)

        # 1) Compute per-row sum of squares in FP32
        sumsq = torch.empty(B, dtype=torch.float32, device=hidden.device)
        _compute_row_sumsq[(B,)](
            hidden_flat, sumsq,
            H, BLOCK_SIZE=1024,  # compile-time chunk size
            num_warps=4, num_stages=2
        )

        # 2) Compute inv_rms per row
        inv_rms = torch.rsqrt(sumsq / H + EPS)  # [B], fp32

        # 3) Normalize and scale elementwise: y = x * inv_rms[row] * weight
        out_flat = torch.empty(B * H, dtype=torch.float32, device=hidden.device)
        # 2D grid: (rows, column blocks)
        BLOCK_SIZE = 128  # chunk along columns; H=4096 -> 32 blocks
        grid = (B, triton.cdiv(H, BLOCK_SIZE))
        _normalize_and_scale_elementwise[grid](
            hidden_flat, weight.view(H), out_flat, inv_rms,
            B, H, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, num_stages=2
        )

        # 4) Reshape and cast to original dtype to match original behavior
        out = out_flat.view(B, H)
        out = out.to(hidden_states.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
