import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    alog_ptr, a_ptr, dt_ptr, b_ptr,
    out_ptr, ns_ptr,
    scale,
    Hq: tl.constexpr, Hv: tl.constexpr, G: tl.constexpr,
    V: tl.constexpr, K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # program handles one (batch, v-head) and a tile of BLOCK_V value rows.
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

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
    B, T, Hq, K = q.shape
    _, _, Hk, _ = k.shape
    _, _, Hv, V = v.shape
    G = Hv // Hq

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    device = q.device

    # contiguous, squeezed views for stride-free flat addressing in the kernel
    q_c = q.reshape(B, Hq, K).contiguous()
    k_c = k.reshape(B, Hk, K).contiguous()
    v_c = v.reshape(B, Hv, V).contiguous()
    a_c = a.reshape(B, Hv).contiguous()
    b_c = b.reshape(B, Hv).contiguous()
    alog_c = A_log.reshape(Hv).contiguous().float()
    dt_c = dt_bias.reshape(Hv).contiguous().float()

    if state is None:
        state_c = torch.zeros(B, Hv, V, K, dtype=torch.float32, device=device)
    else:
        state_c = state.reshape(B, Hv, V, K).contiguous().float()

    output = torch.empty(B, Hv, V, dtype=torch.bfloat16, device=device)
    new_state = torch.empty(B, Hv, V, K, dtype=torch.float32, device=device)

    BLOCK_V = 8
    grid = (B * Hv, triton.cdiv(V, BLOCK_V))
    _gdn_decode_kernel[grid](
        q_c, k_c, v_c, state_c,
        alog_c, a_c, dt_c, b_c,
        output, new_state,
        float(scale),
        Hq, Hv, G, V, K,
        BLOCK_V=BLOCK_V,
        num_warps=4,
    )

    return output.reshape(B, T, Hv, V), new_state
