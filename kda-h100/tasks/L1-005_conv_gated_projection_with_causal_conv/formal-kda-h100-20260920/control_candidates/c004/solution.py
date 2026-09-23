"""
Solution for L1/005 conv_gated_projection_with_causal_conv (H100, sm_90).

Candidate c004 — Option B: fuse the depthwise causal conv + output gating into the
out_proj GEMM prologue.  The `y` intermediate is produced in registers inside the
output-projection kernel and never written to / read from DRAM, and the separate
conv launch is removed (3 kernels -> 2).

Pipeline:
  K1 `_triple_gemm_gate_kernel`   : x -> bx=(Bg*V), cg=Cg           (unchanged from c003)
  K2 `_conv_gate_out_gemm_kernel` : (bx,cg,conv_w,conv_b) -> y (in regs) -> out=y@W_out^T+b

Rationale (docs/plan.md H5 + c003 evidence): GEMM scheduling is near its ceiling
(+1.2% from c003 grouping); the remaining low spots are tiny/medium N where the
extra conv launch + full y DRAM round-trip dominate.  Fusing removes the y write
(N*H*2 bytes) + y read + one launch.  The conv is recomputed once per output-channel
block (num_pid_n = H/BLOCK_N times); with large BLOCK_N this adds only ~2/BLOCK_N of
the out_proj arithmetic (negligible) while the bx tap loads stay L2-resident.

Per-batch causal correctness: token flat index n = b*S + local_s.  Tap j reads
bx[n-(K-1)+j, h], valid iff local_s-(K-1)+j >= 0 (else it is a left-pad zero of THIS
batch).  local_s is recovered per row via n // S, so a BLOCK_M tile that straddles a
batch boundary still masks each row against its own batch start -> no cross-batch halo.

Reference math (per docs/plan.md §1, verified in docs/draft.md §1.2):
  BCx    = x @ in_proj_weight^T + in_proj_bias            # (B,S,3H) bf16
  Bg,Cg,V = BCx[:,:,0:H], BCx[:,:,H:2H], BCx[:,:,2H:3H]
  Bx     = Bg * V                                          # bf16 input gate
  conv[b,s,h] = conv_bias[h] + sum_{k=0..3} conv_weight[h,0,k]*Bx[b,s-3+k,h]
  y      = Cg * conv                                       # bf16 output gate
  output = y @ out_proj_weight^T + out_proj_bias           # (B,S,H) bf16

All compute is in Triton. Torch is used only for shape/stride metadata, output
allocation, and kernel launch. No Torch/CPU/NumPy computational fallback.
"""

import torch
import triton
import triton.language as tl


# K1 has THREE fp32 accumulators per tile (Bg, Cg, V) -> ~3x register pressure of
# a normal GEMM, so BLOCK_M*BLOCK_N is kept moderate to avoid spills.
def _k1_configs():
    return [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    ]


# Fused conv+gate+out_proj: prefer LARGER BLOCK_N because the conv/y tile is
# recomputed once per output-channel block (num_pid_n = H/BLOCK_N), so bigger
# BLOCK_N -> fewer recomputes and less redundant bx tap traffic.
def _k2_configs():
    return [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ]


@triton.jit
def _grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    """Canonical Triton matmul super-grouping: remap a linear pid to (pid_m,pid_n)
    so that GROUP_M consecutive M-tiles are visited before advancing N, improving
    L2 reuse of the weight column tiles.  Bijective over the tile grid."""
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ---------------------------------------------------------------------------
# K1: triple projection + input gating (unchanged from c003).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_k1_configs(), key=["N", "H"])
@triton.jit
def _triple_gemm_gate_kernel(
    x_ptr, w_ptr, b_ptr,          # x:(N,H), in_proj_weight:(3H,H), in_proj_bias:(3H,)
    bx_ptr, cg_ptr,               # outputs: (N,H) bf16
    N, H,
    stride_xm, stride_xk,
    stride_wo, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(H, BLOCK_N)
    pid_m, pid_n = _grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_base = x_ptr + offs_m[:, None] * stride_xm            # + k*stride_xk
    # Transposed weight operand: b[k, n] = W[o_base + n, k]
    wB = w_ptr + (0 * H + offs_n[None, :]) * stride_wo + offs_k[:, None] * stride_wk
    wC = w_ptr + (1 * H + offs_n[None, :]) * stride_wo + offs_k[:, None] * stride_wk
    wV = w_ptr + (2 * H + offs_n[None, :]) * stride_wo + offs_k[:, None] * stride_wk

    accB = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accC = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accV = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_mask = offs_m[:, None] < N
    n_mask = offs_n[None, :] < H

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        k_mask_row = (k0 + offs_k)[None, :] < H     # for x tile [M,K]
        k_mask_col = (k0 + offs_k)[:, None] < H     # for w tile [K,N]

        xt = tl.load(x_base + k[None, :] * stride_xk,
                     mask=m_mask & k_mask_row, other=0.0)
        bt = tl.load(wB + k0 * stride_wk, mask=k_mask_col & n_mask, other=0.0)
        ct = tl.load(wC + k0 * stride_wk, mask=k_mask_col & n_mask, other=0.0)
        vt = tl.load(wV + k0 * stride_wk, mask=k_mask_col & n_mask, other=0.0)

        accB += tl.dot(xt, bt)
        accC += tl.dot(xt, ct)
        accV += tl.dot(xt, vt)

    # Bias (bf16 inputs) added in fp32 to mirror F.linear.
    bB = tl.load(b_ptr + 0 * H + offs_n, mask=offs_n < H, other=0.0).to(tl.float32)
    bC = tl.load(b_ptr + 1 * H + offs_n, mask=offs_n < H, other=0.0).to(tl.float32)
    bV = tl.load(b_ptr + 2 * H + offs_n, mask=offs_n < H, other=0.0).to(tl.float32)

    accB += bB[None, :]
    accC += bC[None, :]
    accV += bV[None, :]

    # Match reference bf16 rounding: BCx is bf16, then Bx = Bg_bf16 * V_bf16.
    bg = accB.to(tl.bfloat16).to(tl.float32)
    vv = accV.to(tl.bfloat16).to(tl.float32)
    bx = (bg * vv).to(tl.bfloat16)
    cg = accC.to(tl.bfloat16)

    out_mask = m_mask & n_mask
    out_off = offs_m[:, None] * H + offs_n[None, :]
    tl.store(bx_ptr + out_off, bx, mask=out_mask)
    tl.store(cg_ptr + out_off, cg, mask=out_mask)


# ---------------------------------------------------------------------------
# K2 (fused): causal conv + output gating (produces y in registers) + out_proj.
#   output[n,o] = out_proj_bias[o] + sum_h y[n,h] * W_out[o,h]
#   y[n,h]      = cg[n,h] * round_bf16( conv_bias[h] + sum_j cw[h,j]*bx[n-(K-1)+j,h] )
#   Contraction dim = h (hidden) which is also the conv channel dim.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_k2_configs(), key=["N", "H"])
@triton.jit
def _conv_gate_out_gemm_kernel(
    bx_ptr, cg_ptr, cw_ptr, cb_ptr,   # (N,H),(N,H),(H,1,K),(H,)
    w_ptr, ob_ptr, o_ptr,             # out_proj_weight:(H,H), out_proj_bias:(H,), out:(N,H)
    N, H, S,
    stride_wo, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, K_CONV: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(H, BLOCK_N)
    pid_m, pid_n = _grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # token rows n
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # output channels o
    offs_k = tl.arange(0, BLOCK_K)                        # hidden channels h

    m_mask = offs_m < N
    n_mask = offs_n[None, :] < H

    # Per-row local sequence position for per-batch causal masking.
    b_idx = offs_m // S
    local_s = offs_m - b_idx * S                          # [BLOCK_M]

    w_base = w_ptr + offs_n[None, :] * stride_wo + offs_k[:, None] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        kh = k0 + offs_k                                  # hidden channel indices [BLOCK_K]
        k_mask_row = kh[None, :] < H                      # [1,BLOCK_K] over columns of y tile
        k_mask_col = kh[:, None] < H                      # [BLOCK_K,1] over rows of W_out tile

        # conv accumulate in fp32 over K_CONV taps, per (row, hidden) of this k-block
        cb = tl.load(cb_ptr + kh, mask=kh < H, other=0.0).to(tl.float32)     # [BLOCK_K]
        conv = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32) + cb[None, :]
        for j in tl.static_range(K_CONV):
            src_row = offs_m - (K_CONV - 1) + j
            tap_ok = (local_s - (K_CONV - 1) + j) >= 0    # per-row, same-batch causal
            load_mask = tap_ok[:, None] & m_mask[:, None] & k_mask_row
            bx_off = src_row[:, None] * H + kh[None, :]
            bxv = tl.load(bx_ptr + bx_off, mask=load_mask, other=0.0).to(tl.float32)
            wk = tl.load(cw_ptr + kh * K_CONV + j, mask=kh < H, other=0.0).to(tl.float32)
            conv += bxv * wk[None, :]

        conv_bf16 = conv.to(tl.bfloat16).to(tl.float32)   # F.conv1d bf16 output rounding

        cg_off = offs_m[:, None] * H + kh[None, :]
        cg = tl.load(cg_ptr + cg_off, mask=m_mask[:, None] & k_mask_row, other=0.0).to(tl.float32)
        y = (cg * conv_bf16).to(tl.bfloat16)              # [BLOCK_M, BLOCK_K] bf16

        wt = tl.load(w_base + k0 * stride_wk, mask=k_mask_col & n_mask, other=0.0)
        acc += tl.dot(y, wt)

    ob = tl.load(ob_ptr + offs_n, mask=offs_n < H, other=0.0).to(tl.float32)
    acc += ob[None, :]
    out = acc.to(tl.bfloat16)
    out_off = offs_m[:, None] * H + offs_n[None, :]
    tl.store(o_ptr + out_off, out, mask=m_mask[:, None] & n_mask)


_GROUP_M = 8


@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    B, S, H = x.shape
    K_CONV = conv_weight.shape[2]
    N = B * S

    x2d = x.reshape(N, H)
    dev = x.device

    bx = torch.empty((N, H), dtype=torch.bfloat16, device=dev)
    cg = torch.empty((N, H), dtype=torch.bfloat16, device=dev)
    out = torch.empty((N, H), dtype=torch.bfloat16, device=dev)

    # K1: triple projection + input gating (autotuned, grouped-M ordering, 1D grid)
    grid1 = lambda META: (triton.cdiv(N, META["BLOCK_M"]) * triton.cdiv(H, META["BLOCK_N"]),)
    _triple_gemm_gate_kernel[grid1](
        x2d, in_proj_weight, in_proj_bias, bx, cg,
        N, H,
        x2d.stride(0), x2d.stride(1),
        in_proj_weight.stride(0), in_proj_weight.stride(1),
        GROUP_M=_GROUP_M,
    )

    # K2: fused causal conv + output gating + out_proj (autotuned, grouped-M, 1D grid)
    grid2 = lambda META: (triton.cdiv(N, META["BLOCK_M"]) * triton.cdiv(H, META["BLOCK_N"]),)
    _conv_gate_out_gemm_kernel[grid2](
        bx, cg, conv_weight, conv_bias,
        out_proj_weight, out_proj_bias, out,
        N, H, S,
        out_proj_weight.stride(0), out_proj_weight.stride(1),
        GROUP_M=_GROUP_M, K_CONV=K_CONV,
    )

    return out.reshape(B, S, H)
