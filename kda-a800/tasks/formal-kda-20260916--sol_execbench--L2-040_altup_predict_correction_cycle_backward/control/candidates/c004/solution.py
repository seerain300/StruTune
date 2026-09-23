import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # overflow-safe tanh built only from tl.exp/tl.abs/tl.where (no tl.sigmoid)
    ax = tl.abs(x)
    e = tl.exp(-2.0 * ax)
    t = (1.0 - e) / (1.0 + e)
    return tl.where(x >= 0, t, -t)


@triton.jit
def _altup_bwd_kernel(
    grad_corrected_ptr, hidden_ptr, activated_ptr,
    pcw_ptr, ccw_ptr, rw_ptr, nw_ptr,
    ghs_ptr, ga_ptr, gpcw_ptr, gccw_ptr, grw_ptr, gnw_ptr,
    rows, eps,
    H: tl.constexpr, IDX: tl.constexpr, BLOCK: tl.constexpr, NUM_TILES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    inv_H = 1.0 / H
    rscale = 1.0 / H
    base_idx = (IDX * rows + row) * H            # active variant (hidden[IDX]) base
    base_act = row * H                           # activated base
    base0 = (0 * rows + row) * H
    base1 = (1 * rows + row) * H
    base2 = (2 * rows + row) * H

    # small coef weights (scalars), fp32
    pcw = [[tl.load(pcw_ptr + p * 3 + q).to(tl.float32) for q in range(3)] for p in range(9)]
    ccw = [[tl.load(ccw_ptr + i * 3 + q).to(tl.float32) for q in range(3)] for i in range(3)]

    # ---------------- Pass A: variance + raw router dots ----------------
    sq_p = 0.0
    sq_c = 0.0
    raw_p0 = 0.0; raw_p1 = 0.0; raw_p2 = 0.0
    raw_c0 = 0.0; raw_c1 = 0.0; raw_c2 = 0.0
    for t in tl.static_range(NUM_TILES):
        h = t * BLOCK + tl.arange(0, BLOCK)
        m = h < H
        a = tl.load(hidden_ptr + base_idx + h, mask=m, other=0.0).to(tl.float32)
        act = tl.load(activated_ptr + base_act + h, mask=m, other=0.0).to(tl.float32)
        nw = tl.load(nw_ptr + h, mask=m, other=0.0).to(tl.float32)
        rw0 = tl.load(rw_ptr + 0 * H + h, mask=m, other=0.0).to(tl.float32)
        rw1 = tl.load(rw_ptr + 1 * H + h, mask=m, other=0.0).to(tl.float32)
        rw2 = tl.load(rw_ptr + 2 * H + h, mask=m, other=0.0).to(tl.float32)
        sq_p += tl.sum(a * a, axis=0)
        sq_c += tl.sum(act * act, axis=0)
        anw = a * nw
        cnw = act * nw
        raw_p0 += tl.sum(anw * rw0, axis=0)
        raw_p1 += tl.sum(anw * rw1, axis=0)
        raw_p2 += tl.sum(anw * rw2, axis=0)
        raw_c0 += tl.sum(cnw * rw0, axis=0)
        raw_c1 += tl.sum(cnw * rw1, axis=0)
        raw_c2 += tl.sum(cnw * rw2, axis=0)

    rstd_p = 1.0 / tl.sqrt(sq_p * inv_H + eps)
    rstd_c = 1.0 / tl.sqrt(sq_c * inv_H + eps)
    routed_p = [rstd_p * rscale * raw_p0, rstd_p * rscale * raw_p1, rstd_p * rscale * raw_p2]
    routed_c = [rstd_c * rscale * raw_c0, rstd_c * rscale * raw_c1, rstd_c * rscale * raw_c2]
    mod_p = [_tanh(routed_p[k]) for k in range(3)]
    mod_c = [_tanh(routed_c[k]) for k in range(3)]

    flat = [mod_p[0] * pcw[p][0] + mod_p[1] * pcw[p][1] + mod_p[2] * pcw[p][2] for p in range(9)]
    # C[m][n] = flat[n*3 + m]
    C = [[flat[n * 3 + m] for n in range(3)] for m in range(3)]
    acoef = [mod_c[0] * ccw[n][0] + mod_c[1] * ccw[n][1] + mod_c[2] * ccw[n][2] + 1.0 for n in range(3)]

    # ---------------- Pass B: g_acoef and g_flat reductions ----------------
    g_acoef0 = 0.0; g_acoef1 = 0.0; g_acoef2 = 0.0
    g_flat = [0.0 for _ in range(9)]
    for t in tl.static_range(NUM_TILES):
        h = t * BLOCK + tl.arange(0, BLOCK)
        m = h < H
        h0 = tl.load(hidden_ptr + base0 + h, mask=m, other=0.0).to(tl.float32)
        h1 = tl.load(hidden_ptr + base1 + h, mask=m, other=0.0).to(tl.float32)
        h2 = tl.load(hidden_ptr + base2 + h, mask=m, other=0.0).to(tl.float32)
        gc0 = tl.load(grad_corrected_ptr + base0 + h, mask=m, other=0.0).to(tl.float32)
        gc1 = tl.load(grad_corrected_ptr + base1 + h, mask=m, other=0.0).to(tl.float32)
        gc2 = tl.load(grad_corrected_ptr + base2 + h, mask=m, other=0.0).to(tl.float32)
        act = tl.load(activated_ptr + base_act + h, mask=m, other=0.0).to(tl.float32)
        hlist = [h0, h1, h2]
        gclist = [gc0, gc1, gc2]
        active = hlist[IDX]
        pred_idx = active + h0 * C[0][IDX] + h1 * C[1][IDX] + h2 * C[2][IDX]
        innov = act - pred_idx
        g_acoef0 += tl.sum(gc0 * innov, axis=0)
        g_acoef1 += tl.sum(gc1 * innov, axis=0)
        g_acoef2 += tl.sum(gc2 * innov, axis=0)
        g_innov = gc0 * acoef[0] + gc1 * acoef[1] + gc2 * acoef[2]
        gpred = [gclist[a] - (g_innov if a == IDX else 0.0) for a in range(3)]
        for a in range(3):
            for c in range(3):
                g_flat[a * 3 + c] += tl.sum(hlist[c] * gpred[a], axis=0)

    g_acoef = [g_acoef0, g_acoef1, g_acoef2]
    # tiny weight-grad atomics (once per row)
    for p in range(3):
        for q in range(3):
            tl.atomic_add(gccw_ptr + p * 3 + q, g_acoef[p] * mod_c[q])
    for p in range(9):
        for q in range(3):
            tl.atomic_add(gpcw_ptr + p * 3 + q, g_flat[p] * mod_p[q])

    g_mod_c = [g_acoef[0] * ccw[0][q] + g_acoef[1] * ccw[1][q] + g_acoef[2] * ccw[2][q] for q in range(3)]
    g_routed_c = [g_mod_c[k] * (1.0 - mod_c[k] * mod_c[k]) for k in range(3)]
    g_mod_p = [sum(g_flat[p] * pcw[p][q] for p in range(9)) for q in range(3)]
    g_routed_p = [g_mod_p[k] * (1.0 - mod_p[k] * mod_p[k]) for k in range(3)]

    # ---------------- Pass C: mbar reductions + grad_rw/grad_nw atomics ----------------
    mbar_c = 0.0
    mbar_p = 0.0
    for t in tl.static_range(NUM_TILES):
        h = t * BLOCK + tl.arange(0, BLOCK)
        m = h < H
        a = tl.load(hidden_ptr + base_idx + h, mask=m, other=0.0).to(tl.float32)
        act = tl.load(activated_ptr + base_act + h, mask=m, other=0.0).to(tl.float32)
        nw = tl.load(nw_ptr + h, mask=m, other=0.0).to(tl.float32)
        rw0 = tl.load(rw_ptr + 0 * H + h, mask=m, other=0.0).to(tl.float32)
        rw1 = tl.load(rw_ptr + 1 * H + h, mask=m, other=0.0).to(tl.float32)
        rw2 = tl.load(rw_ptr + 2 * H + h, mask=m, other=0.0).to(tl.float32)
        gsc_c = g_routed_c[0] * rw0 + g_routed_c[1] * rw1 + g_routed_c[2] * rw2
        gsc_p = g_routed_p[0] * rw0 + g_routed_p[1] * rw1 + g_routed_p[2] * rw2
        g_normed_c = gsc_c * rscale
        g_normed_p = gsc_p * rscale
        g_norm_c = g_normed_c * nw
        g_norm_p = g_normed_p * nw
        mbar_c += tl.sum(g_norm_c * act, axis=0)
        mbar_p += tl.sum(g_norm_p * a, axis=0)
        nrm_c = act * rstd_c
        nrm_p = a * rstd_p
        scl_c = nrm_c * nw * rscale
        scl_p = nrm_p * nw * rscale
        tl.atomic_add(grw_ptr + 0 * H + h, g_routed_c[0] * scl_c + g_routed_p[0] * scl_p, mask=m)
        tl.atomic_add(grw_ptr + 1 * H + h, g_routed_c[1] * scl_c + g_routed_p[1] * scl_p, mask=m)
        tl.atomic_add(grw_ptr + 2 * H + h, g_routed_c[2] * scl_c + g_routed_p[2] * scl_p, mask=m)
        tl.atomic_add(gnw_ptr + h, g_normed_c * nrm_c + g_normed_p * nrm_p, mask=m)

    mbar_c = mbar_c * inv_H
    mbar_p = mbar_p * inv_H
    rstd_c3 = rstd_c * rstd_c * rstd_c
    rstd_p3 = rstd_p * rstd_p * rstd_p

    # ---------------- Pass D: stores of grad_activated and grad_hidden_states ----------------
    for t in tl.static_range(NUM_TILES):
        h = t * BLOCK + tl.arange(0, BLOCK)
        m = h < H
        a = tl.load(hidden_ptr + base_idx + h, mask=m, other=0.0).to(tl.float32)
        act = tl.load(activated_ptr + base_act + h, mask=m, other=0.0).to(tl.float32)
        nw = tl.load(nw_ptr + h, mask=m, other=0.0).to(tl.float32)
        rw0 = tl.load(rw_ptr + 0 * H + h, mask=m, other=0.0).to(tl.float32)
        rw1 = tl.load(rw_ptr + 1 * H + h, mask=m, other=0.0).to(tl.float32)
        rw2 = tl.load(rw_ptr + 2 * H + h, mask=m, other=0.0).to(tl.float32)
        gc0 = tl.load(grad_corrected_ptr + base0 + h, mask=m, other=0.0).to(tl.float32)
        gc1 = tl.load(grad_corrected_ptr + base1 + h, mask=m, other=0.0).to(tl.float32)
        gc2 = tl.load(grad_corrected_ptr + base2 + h, mask=m, other=0.0).to(tl.float32)

        g_innov = gc0 * acoef[0] + gc1 * acoef[1] + gc2 * acoef[2]
        gsc_c = g_routed_c[0] * rw0 + g_routed_c[1] * rw1 + g_routed_c[2] * rw2
        g_norm_c = gsc_c * rscale * nw
        ga = g_innov + g_norm_c * rstd_c - act * rstd_c3 * mbar_c
        tl.store(ga_ptr + base_act + h, ga.to(tl.bfloat16), mask=m)

        gclist = [gc0, gc1, gc2]
        gpred = [gclist[a] - (g_innov if a == IDX else 0.0) for a in range(3)]
        gsc_p = g_routed_p[0] * rw0 + g_routed_p[1] * rw1 + g_routed_p[2] * rw2
        g_norm_p = gsc_p * rscale * nw
        g_active = g_norm_p * rstd_p - a * rstd_p3 * mbar_p

        ghs = [gpred[mm] + gpred[0] * C[mm][0] + gpred[1] * C[mm][1] + gpred[2] * C[mm][2] for mm in range(3)]
        ghs = [ghs[mm] + (g_active if mm == IDX else 0.0) for mm in range(3)]
        tl.store(ghs_ptr + base0 + h, ghs[0].to(tl.bfloat16), mask=m)
        tl.store(ghs_ptr + base1 + h, ghs[1].to(tl.bfloat16), mask=m)
        tl.store(ghs_ptr + base2 + h, ghs[2].to(tl.bfloat16), mask=m)


def run(
    grad_corrected: torch.Tensor,
    hidden_states: torch.Tensor,
    activated: torch.Tensor,
    prediction_coef_weight: torch.Tensor,
    correction_coef_weight: torch.Tensor,
    router_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    altup_active_idx,
    rms_norm_eps: float,
):
    N, B, S, H = hidden_states.shape
    rows = B * S

    grad_corrected = grad_corrected.contiguous()
    hidden_states = hidden_states.contiguous()
    activated = activated.contiguous()
    prediction_coef_weight = prediction_coef_weight.contiguous()
    correction_coef_weight = correction_coef_weight.contiguous()
    router_weight = router_weight.contiguous()
    norm_weight = norm_weight.contiguous()

    grad_hidden_states = torch.empty_like(hidden_states)
    grad_activated = torch.empty_like(activated)
    grad_pcw = torch.zeros_like(prediction_coef_weight, dtype=torch.float32)
    grad_ccw = torch.zeros_like(correction_coef_weight, dtype=torch.float32)
    grad_rw = torch.zeros_like(router_weight, dtype=torch.float32)
    grad_nw = torch.zeros_like(norm_weight, dtype=torch.float32)

    idx = int(altup_active_idx)
    eps = float(rms_norm_eps)
    BLOCK = 256                     # power-of-two; 2304 = 9*256 -> exact tiling, tl.arange-safe
    num_tiles = (H + BLOCK - 1) // BLOCK

    grid = (rows,)
    _altup_bwd_kernel[grid](
        grad_corrected, hidden_states, activated,
        prediction_coef_weight, correction_coef_weight, router_weight, norm_weight,
        grad_hidden_states, grad_activated, grad_pcw, grad_ccw, grad_rw, grad_nw,
        rows, eps,
        H=H, IDX=idx, BLOCK=BLOCK, NUM_TILES=num_tiles, num_warps=4,
    )

    return (
        grad_hidden_states,
        grad_activated,
        grad_pcw,
        grad_ccw,
        grad_rw,
        grad_nw,
    )
