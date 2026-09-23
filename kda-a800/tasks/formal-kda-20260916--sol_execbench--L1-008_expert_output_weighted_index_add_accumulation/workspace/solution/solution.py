import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# L1/008 expert_output_weighted_index_add_accumulation
#
# Semantics (reference):
#   output = final_hidden_states.clone()
#   output.index_add_(0, token_indices, expert_outputs)   # output[idx[i]] += src[i]
#
# Candidate c002 = Option B0 form (fp32 scratch, 3 Triton kernels) with a
# *full-row scatter* (docs/plan.md Phase B, H4):
#   1. init:     fp32 scratch buf[M,H] <- base (cast bf16 -> fp32)         [tiled]
#   2. scatter:  one program per source row (grid (N,)); the program loops
#                over the NUM_H hidden tiles of that row, so it loads the
#                int64 idx exactly once and issues the fp32 atomic RMWs for
#                all of buf[idx]'s cache lines back-to-back (better L2 locality
#                for the accumulator, fewer redundant idx loads, fewer programs).
#   3. finalize: output[M,H] (bf16) <- buf   (cast fp32 -> bf16, round once)  [tiled]
#
# Only the scatter kernel's tiling/grid changes vs c001 (single knob). init and
# finalize kernels and all launch configs are identical to c001.
#
# All compute is in Triton. PyTorch is used only for tensor allocation/metadata.
# fp32 accumulation + single final rounding is >= as accurate as the reference's
# repeatedly-rounded bf16 CAS accumulation, so it stays within tolerance.
# ---------------------------------------------------------------------------


@triton.jit
def _init_copy_kernel(
    src_ptr,          # bf16 [M, H]  (final_hidden_states)
    dst_ptr,          # fp32 [M, H]  (buf)
    M, H,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    row = pid_m.to(tl.int64) * H
    x = tl.load(src_ptr + row + offs_h, mask=mask, other=0.0)
    tl.store(dst_ptr + row + offs_h, x.to(tl.float32), mask=mask)


@triton.jit
def _scatter_add_kernel(
    src_ptr,          # bf16 [N, H]  (expert_outputs)
    idx_ptr,          # int64 [N]    (token_indices)
    buf_ptr,          # fp32 [M, H]  (accumulation scratch)
    N, H,
    BLOCK_H: tl.constexpr,
    NUM_H: tl.constexpr,
):
    pid_n = tl.program_id(0)

    # One int64 index load per source row (amortized over all hidden tiles).
    idx = tl.load(idx_ptr + pid_n).to(tl.int64)
    src_row = pid_n.to(tl.int64) * H
    dst_row = idx * H

    for i in tl.static_range(NUM_H):
        offs_h = i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs_h < H
        s = tl.load(src_ptr + src_row + offs_h, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(buf_ptr + dst_row + offs_h, s, mask=mask)


@triton.jit
def _finalize_kernel(
    buf_ptr,          # fp32 [M, H]
    out_ptr,          # bf16 [M, H]
    M, H,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    row = pid_m.to(tl.int64) * H
    x = tl.load(buf_ptr + row + offs_h, mask=mask)
    tl.store(out_ptr + row + offs_h, x.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    assert final_hidden_states.ndim == 2
    assert expert_outputs.ndim == 2
    assert token_indices.ndim == 1
    M, H = final_hidden_states.shape
    N, Hs = expert_outputs.shape
    assert Hs == H, "hidden dim mismatch"
    assert token_indices.shape[0] == N, "index/source count mismatch"

    # Metadata / launch plumbing only (no computational fallback).
    base = final_hidden_states if final_hidden_states.is_contiguous() else final_hidden_states.contiguous()
    src = expert_outputs if expert_outputs.is_contiguous() else expert_outputs.contiguous()
    idx = token_indices if token_indices.is_contiguous() else token_indices.contiguous()

    buf = torch.empty((M, H), dtype=torch.float32, device=base.device)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=base.device)

    BLOCK_H = 1024
    num_h = triton.cdiv(H, BLOCK_H)

    # 1. init buf <- base (fp32)  [tiled, identical to c001]
    _init_copy_kernel[(M, num_h)](base, buf, M, H, BLOCK_H=BLOCK_H, num_warps=4)

    # 2. scatter-add expert outputs into buf (native fp32 atomics), full-row:
    #    one program per source row, looping over the num_h hidden tiles.
    _scatter_add_kernel[(N,)](src, idx, buf, N, H, BLOCK_H=BLOCK_H, NUM_H=num_h, num_warps=4)

    # 3. finalize: round buf -> bf16 output  [tiled, identical to c001]
    _finalize_kernel[(M, num_h)](buf, out, M, H, BLOCK_H=BLOCK_H, num_warps=4)

    return out
