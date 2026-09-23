import math

import torch
import triton
import triton.language as tl


def _autotune_configs():
    cfgs = []
    for bv in (1, 2, 4, 8, 16):
        for w in (1, 2, 4):
            cfgs.append(triton.Config({"BLOCK_V": bv}, num_warps=w))
    return cfgs


@triton.autotune(configs=_autotune_configs(), key=["n_bh"])
@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    alog_ptr, a_ptr, dt_ptr, b_ptr,
    out_ptr, ns_ptr,
    scale,
    n_bh,
    Hq: tl.constexpr, Hv: tl.constexpr, G: tl.constexpr,
    V: tl.constexpr, K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # 1-D grid: pid -> (batch*Hv, v-tile). Each program handles one (b, v-head)
    # and a tile of BLOCK_V value rows.
    pid = tl.program_id(0)
    num_v_tiles: tl.constexpr = V // BLOCK_V
    pid_bh = pid // num_v_tiles
    pid_v = pid % num_v_tiles

    b = pid_bh // Hv
    h = pid_bh % Hv
    hqk = h // G  # GVA: v-head h reads q/k-head h // G

    offs_k = tl.arange(0, K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    # --- gates (scalars for this (b,h)) ---
    alog = tl.load(alog_ptr + h).to(tl.float32)
    dt = tl.load(dt_ptr + h).to(tl.float32)
    aval = tl.load(a_ptr + (b * Hv + h)).to(tl.float32)
    bval = tl.load(b_ptr + (b * Hv + h)).to(tl.float32)

    x = aval + dt
    # numerically-stable softplus: relu(x) + log1p(exp(-|x|))  (matches F.softplus)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    g = tl.exp(-tl.exp(alog) * sp)
    # numerically-stable sigmoid
    beta = tl.where(bval >= 0,
                    1.0 / (1.0 + tl.exp(-bval)),
                    tl.exp(bval) / (1.0 + tl.exp(bval)))

    # --- per-head vectors (length K), loaded once and reused over the v-tile ---
    q_h = tl.load(q_ptr + (b * Hq + hqk) * K + offs_k).to(tl.float32)
    k_h = tl.load(k_ptr + (b * Hq + hqk) * K + offs_k).to(tl.float32)
    v_h = tl.load(v_ptr + (b * Hv + h) * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)

    # --- k-last state rows: state[b, h, offs_v, :] (K contiguous) ---
    s_off = ((b * Hv + h) * V + offs_v[:, None]) * K + offs_k[None, :]
    S = tl.load(state_ptr + s_off, mask=mask_v[:, None], other=0.0).to(tl.float32)

    # --- faithful delta-rule update (materialise new_row, then dot with q) ---
    sk = tl.sum(S * k_h[None, :], axis=1)          # [BLOCK_V]
    old_v = g * sk
    new_v = beta * v_h + (1.0 - beta) * old_v
    delta = new_v - old_v                          # = beta * (v_h - old_v)
    new_tile = g * S + k_h[None, :] * delta[:, None]
    out = scale * tl.sum(q_h[None, :] * new_tile, axis=1)  # [BLOCK_V]

    # --- stores ---
    tl.store(ns_ptr + s_off, new_tile, mask=mask_v[:, None])
    tl.store(out_ptr + (b * Hv + h) * V + offs_v, out.to(out_ptr.dtype.element_ty), mask=mask_v)


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    # c003: 1-D grid + @triton.autotune over (BLOCK_V, num_warps) keyed on batch
    # (n_bh = B*Hv). c002 showed we are latency/occupancy-bound (flat ~0.045ms
    # even at B=64), so let the autotuner pick the occupancy/tiling that best
    # fills the SMs per batch regime. Wrapper overhead kept minimal as in c002;
    # kernel math is identical to c001/c002.
    B, T, Hq, K = q.shape
    Hv = v.shape[2]
    V = v.shape[3]
    G = Hv // Hq

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()
    if state is None:
        state = torch.zeros(B, Hv, V, K, dtype=torch.float32, device=q.device)
    else:
        state = state.contiguous()

    output = torch.empty((B, T, Hv, V), dtype=torch.bfloat16, device=q.device)
    new_state = torch.empty((B, Hv, V, K), dtype=torch.float32, device=q.device)

    n_bh = B * Hv
    grid = lambda META: (n_bh * (V // META["BLOCK_V"]),)
    _gdn_decode_kernel[grid](
        q, k, v, state,
        A_log, a, dt_bias, b,
        output, new_state,
        float(scale),
        n_bh,
        Hq, Hv, G, V, K,
    )

    return output, new_state
