"""
Solution for L1/005 conv_gated_projection_with_causal_conv (H100, sm_90).

Candidate c005 — branch from working-best c003 (autotuned Option A + GROUP_M
super-grouping on both GEMMs).  c004's Option B fusion was rejected (conv recompute
per N-tile dominated).  c005 keeps the standalone 3-kernel structure -- so y is
computed exactly ONCE, no redundant recompute -- but AUTOTUNES the conv+gate kernel
K2 (previously fixed BLOCK_S=64/BLOCK_H=64/num_warps=4 since c001) over tile shape
and warp count.

Rationale (docs/plan.md H4 + c003 evidence): the conv K2 is a memory-bound pass
(reads bx + cg, writes y).  At large N it is dwarfed by the two GEMMs, but at tiny N
(N=256) the GEMMs are trivial and K2 + launch overhead cap speedup at ~1.26-1.30x.
Autotuning lets tiny-N pick small tiles (more programs -> better SM occupancy on the
132-SM H100) while larger shapes pick wide-BLOCK_H tiles (H is contiguous ->
coalesced bf16 loads).  Same math as c003, so all-16 correctness is preserved.

Pipeline:
  K1 `_triple_gemm_gate_kernel`  : x -> bx=(Bg*V), cg=Cg
  K2 `_causal_conv_gate_kernel`  : (bx,cg,conv_w,conv_b) -> y   (AUTOTUNED)
  K3 `_out_gemm_kernel`          : y -> out = y@W_out^T + b

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


def _k3_configs():
    return [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8, num_stages=4),
    ]


# Conv+gate (memory-bound): small tiles -> more programs -> occupancy at tiny N;
# wide BLOCK_H -> coalesced bf16 loads (H is the contiguous axis of (N,H)).
def _conv_configs():
    return [
        triton.Config({"BLOCK_S": 32,  "BLOCK_H": 64},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 64,  "BLOCK_H": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 32,  "BLOCK_H": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64,  "BLOCK_H": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 64,  "BLOCK_H": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 256, "BLOCK_H": 128}, num_warps=8, num_stages=2),
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
# K2: depthwise causal conv (kernel_size=K_CONV) + output gating.  Now AUTOTUNED
#   over (BLOCK_S, BLOCK_H, num_warps).  Same math as c001-c003; per-batch causal
#   masking (bx[s'<0]=0, halo never crosses batch boundary; grid dim 0 = batch b).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_conv_configs(), key=["S", "H"])
@triton.jit
def _causal_conv_gate_kernel(
    bx_ptr, cg_ptr, cw_ptr, cb_ptr, y_ptr,   # (N,H),(N,H),(H,1,K),(H,),(N,H)
    S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr, K_CONV: tl.constexpr,
):
    b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    s_valid = offs_s < S
    h_valid = offs_h < H

    cb = tl.load(cb_ptr + offs_h, mask=h_valid, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32) + cb[None, :]

    row_base = b * S  # start of this batch's tokens in the (N,H) buffer
    for k in tl.static_range(K_CONV):
        # tap k reads Bx at s - (K_CONV-1) + k
        src_s = offs_s - (K_CONV - 1) + k
        load_mask = (src_s[:, None] >= 0) & s_valid[:, None] & h_valid[None, :]
        bx_off = (row_base + src_s[:, None]) * H + offs_h[None, :]
        bxv = tl.load(bx_ptr + bx_off, mask=load_mask, other=0.0).to(tl.float32)
        wk = tl.load(cw_ptr + offs_h * K_CONV + k, mask=h_valid, other=0.0).to(tl.float32)
        acc += bxv * wk[None, :]

    conv_bf16 = acc.to(tl.bfloat16).to(tl.float32)   # F.conv1d bf16 output rounding

    y_off = (row_base + offs_s[:, None]) * H + offs_h[None, :]
    store_mask = s_valid[:, None] & h_valid[None, :]
    cg = tl.load(cg_ptr + y_off, mask=store_mask, other=0.0).to(tl.float32)
    y = (cg * conv_bf16).to(tl.bfloat16)
    tl.store(y_ptr + y_off, y, mask=store_mask)


# ---------------------------------------------------------------------------
# K3: output projection (unchanged from c003).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_k3_configs(), key=["N", "H"])
@triton.jit
def _out_gemm_kernel(
    y_ptr, w_ptr, b_ptr, o_ptr,    # y:(N,H), out_proj_weight:(H,H), bias:(H,), out:(N,H)
    N, H,
    stride_ym, stride_yk,
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

    y_base = y_ptr + offs_m[:, None] * stride_ym
    w_base = w_ptr + offs_n[None, :] * stride_wo + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < N
    n_mask = offs_n[None, :] < H

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k
        k_mask_row = (k0 + offs_k)[None, :] < H
        k_mask_col = (k0 + offs_k)[:, None] < H
        yt = tl.load(y_base + k[None, :] * stride_yk,
                     mask=m_mask & k_mask_row, other=0.0)
        wt = tl.load(w_base + k0 * stride_wk, mask=k_mask_col & n_mask, other=0.0)
        acc += tl.dot(yt, wt)

    bb = tl.load(b_ptr + offs_n, mask=offs_n < H, other=0.0).to(tl.float32)
    acc += bb[None, :]
    out = acc.to(tl.bfloat16)
    out_off = offs_m[:, None] * H + offs_n[None, :]
    tl.store(o_ptr + out_off, out, mask=m_mask & n_mask)


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
    y = torch.empty((N, H), dtype=torch.bfloat16, device=dev)
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

    # K2: causal conv + output gating (autotuned)
    grid2 = lambda META: (B, triton.cdiv(S, META["BLOCK_S"]), triton.cdiv(H, META["BLOCK_H"]))
    _causal_conv_gate_kernel[grid2](
        bx, cg, conv_weight, conv_bias, y,
        S, H,
        K_CONV=K_CONV,
    )

    # K3: output projection (autotuned, grouped-M ordering, 1D grid)
    grid3 = lambda META: (triton.cdiv(N, META["BLOCK_M"]) * triton.cdiv(H, META["BLOCK_N"]),)
    _out_gemm_kernel[grid3](
        y, out_proj_weight, out_proj_bias, out,
        N, H,
        y.stride(0), y.stride(1),
        out_proj_weight.stride(0), out_proj_weight.stride(1),
        GROUP_M=_GROUP_M,
    )

    return out.reshape(B, S, H)
