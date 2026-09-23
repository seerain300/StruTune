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
    H: tl.constexpr, IDX: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Single-pass, fully H-resident: each heavy tensor is read exactly once and
    # every H-reduction (variance, router dots, g_acoef, g_flat, mbar) is done on
    # the resident vectors. No list comprehensions / list literals of triton
    # values (unsupported by this build): everything unrolled into named scalars.
    row = tl.program_id(0).to(tl.int64)
    inv_H = 1.0 / H
    rscale = 1.0 / H
    base_idx = (IDX * rows + row) * H
    base_act = row * H
    base0 = (0 * rows + row) * H
    base1 = (1 * rows + row) * H
    base2 = (2 * rows + row) * H

    h = tl.arange(0, BLOCK_H)
    m = h < H

    # ---- single load of every per-row vector ----
    h0 = tl.load(hidden_ptr + base0 + h, mask=m, other=0.0).to(tl.float32)
    h1 = tl.load(hidden_ptr + base1 + h, mask=m, other=0.0).to(tl.float32)
    h2 = tl.load(hidden_ptr + base2 + h, mask=m, other=0.0).to(tl.float32)
    gc0 = tl.load(grad_corrected_ptr + base0 + h, mask=m, other=0.0).to(tl.float32)
    gc1 = tl.load(grad_corrected_ptr + base1 + h, mask=m, other=0.0).to(tl.float32)
    gc2 = tl.load(grad_corrected_ptr + base2 + h, mask=m, other=0.0).to(tl.float32)
    act = tl.load(activated_ptr + base_act + h, mask=m, other=0.0).to(tl.float32)
    nw = tl.load(nw_ptr + h, mask=m, other=0.0).to(tl.float32)
    rw0 = tl.load(rw_ptr + 0 * H + h, mask=m, other=0.0).to(tl.float32)
    rw1 = tl.load(rw_ptr + 1 * H + h, mask=m, other=0.0).to(tl.float32)
    rw2 = tl.load(rw_ptr + 2 * H + h, mask=m, other=0.0).to(tl.float32)

    if IDX == 0:
        active = h0
    elif IDX == 1:
        active = h1
    else:
        active = h2

    # coef weights [9,3] and [3,3] as scalars
    pcw00 = tl.load(pcw_ptr + 0).to(tl.float32); pcw01 = tl.load(pcw_ptr + 1).to(tl.float32); pcw02 = tl.load(pcw_ptr + 2).to(tl.float32)
    pcw10 = tl.load(pcw_ptr + 3).to(tl.float32); pcw11 = tl.load(pcw_ptr + 4).to(tl.float32); pcw12 = tl.load(pcw_ptr + 5).to(tl.float32)
    pcw20 = tl.load(pcw_ptr + 6).to(tl.float32); pcw21 = tl.load(pcw_ptr + 7).to(tl.float32); pcw22 = tl.load(pcw_ptr + 8).to(tl.float32)
    pcw30 = tl.load(pcw_ptr + 9).to(tl.float32); pcw31 = tl.load(pcw_ptr + 10).to(tl.float32); pcw32 = tl.load(pcw_ptr + 11).to(tl.float32)
    pcw40 = tl.load(pcw_ptr + 12).to(tl.float32); pcw41 = tl.load(pcw_ptr + 13).to(tl.float32); pcw42 = tl.load(pcw_ptr + 14).to(tl.float32)
    pcw50 = tl.load(pcw_ptr + 15).to(tl.float32); pcw51 = tl.load(pcw_ptr + 16).to(tl.float32); pcw52 = tl.load(pcw_ptr + 17).to(tl.float32)
    pcw60 = tl.load(pcw_ptr + 18).to(tl.float32); pcw61 = tl.load(pcw_ptr + 19).to(tl.float32); pcw62 = tl.load(pcw_ptr + 20).to(tl.float32)
    pcw70 = tl.load(pcw_ptr + 21).to(tl.float32); pcw71 = tl.load(pcw_ptr + 22).to(tl.float32); pcw72 = tl.load(pcw_ptr + 23).to(tl.float32)
    pcw80 = tl.load(pcw_ptr + 24).to(tl.float32); pcw81 = tl.load(pcw_ptr + 25).to(tl.float32); pcw82 = tl.load(pcw_ptr + 26).to(tl.float32)
    ccw00 = tl.load(ccw_ptr + 0).to(tl.float32); ccw01 = tl.load(ccw_ptr + 1).to(tl.float32); ccw02 = tl.load(ccw_ptr + 2).to(tl.float32)
    ccw10 = tl.load(ccw_ptr + 3).to(tl.float32); ccw11 = tl.load(ccw_ptr + 4).to(tl.float32); ccw12 = tl.load(ccw_ptr + 5).to(tl.float32)
    ccw20 = tl.load(ccw_ptr + 6).to(tl.float32); ccw21 = tl.load(ccw_ptr + 7).to(tl.float32); ccw22 = tl.load(ccw_ptr + 8).to(tl.float32)

    # ---- RMSNorm + router (predict & correct) ----
    var_p = tl.sum(active * active, axis=0) * inv_H
    var_c = tl.sum(act * act, axis=0) * inv_H
    rstd_p = 1.0 / tl.sqrt(var_p + eps)
    rstd_c = 1.0 / tl.sqrt(var_c + eps)
    anw = active * nw
    cnw = act * nw
    raw_p0 = tl.sum(anw * rw0, axis=0); raw_p1 = tl.sum(anw * rw1, axis=0); raw_p2 = tl.sum(anw * rw2, axis=0)
    raw_c0 = tl.sum(cnw * rw0, axis=0); raw_c1 = tl.sum(cnw * rw1, axis=0); raw_c2 = tl.sum(cnw * rw2, axis=0)
    routed_p0 = rstd_p * rscale * raw_p0; routed_p1 = rstd_p * rscale * raw_p1; routed_p2 = rstd_p * rscale * raw_p2
    routed_c0 = rstd_c * rscale * raw_c0; routed_c1 = rstd_c * rscale * raw_c1; routed_c2 = rstd_c * rscale * raw_c2
    mod_p0 = _tanh(routed_p0); mod_p1 = _tanh(routed_p1); mod_p2 = _tanh(routed_p2)
    mod_c0 = _tanh(routed_c0); mod_c1 = _tanh(routed_c1); mod_c2 = _tanh(routed_c2)

    flat0 = mod_p0 * pcw00 + mod_p1 * pcw01 + mod_p2 * pcw02
    flat1 = mod_p0 * pcw10 + mod_p1 * pcw11 + mod_p2 * pcw12
    flat2 = mod_p0 * pcw20 + mod_p1 * pcw21 + mod_p2 * pcw22
    flat3 = mod_p0 * pcw30 + mod_p1 * pcw31 + mod_p2 * pcw32
    flat4 = mod_p0 * pcw40 + mod_p1 * pcw41 + mod_p2 * pcw42
    flat5 = mod_p0 * pcw50 + mod_p1 * pcw51 + mod_p2 * pcw52
    flat6 = mod_p0 * pcw60 + mod_p1 * pcw61 + mod_p2 * pcw62
    flat7 = mod_p0 * pcw70 + mod_p1 * pcw71 + mod_p2 * pcw72
    flat8 = mod_p0 * pcw80 + mod_p1 * pcw81 + mod_p2 * pcw82

    C00 = flat0; C01 = flat3; C02 = flat6
    C10 = flat1; C11 = flat4; C12 = flat7
    C20 = flat2; C21 = flat5; C22 = flat8
    if IDX == 0:
        Cidx0 = C00; Cidx1 = C10; Cidx2 = C20
    elif IDX == 1:
        Cidx0 = C01; Cidx1 = C11; Cidx2 = C21
    else:
        Cidx0 = C02; Cidx1 = C12; Cidx2 = C22

    acoef0 = mod_c0 * ccw00 + mod_c1 * ccw01 + mod_c2 * ccw02 + 1.0
    acoef1 = mod_c0 * ccw10 + mod_c1 * ccw11 + mod_c2 * ccw12 + 1.0
    acoef2 = mod_c0 * ccw20 + mod_c1 * ccw21 + mod_c2 * ccw22 + 1.0

    # ---- correct-step backward reductions ----
    pred_idx = active + h0 * Cidx0 + h1 * Cidx1 + h2 * Cidx2
    innov = act - pred_idx
    g_acoef0 = tl.sum(gc0 * innov, axis=0)
    g_acoef1 = tl.sum(gc1 * innov, axis=0)
    g_acoef2 = tl.sum(gc2 * innov, axis=0)
    g_innov = gc0 * acoef0 + gc1 * acoef1 + gc2 * acoef2

    gpred0 = gc0
    gpred1 = gc1
    gpred2 = gc2
    if IDX == 0:
        gpred0 = gc0 - g_innov
    elif IDX == 1:
        gpred1 = gc1 - g_innov
    else:
        gpred2 = gc2 - g_innov

    gf0 = tl.sum(h0 * gpred0, axis=0); gf1 = tl.sum(h1 * gpred0, axis=0); gf2 = tl.sum(h2 * gpred0, axis=0)
    gf3 = tl.sum(h0 * gpred1, axis=0); gf4 = tl.sum(h1 * gpred1, axis=0); gf5 = tl.sum(h2 * gpred1, axis=0)
    gf6 = tl.sum(h0 * gpred2, axis=0); gf7 = tl.sum(h1 * gpred2, axis=0); gf8 = tl.sum(h2 * gpred2, axis=0)

    # ---- tiny weight-grad atomics (once per row) ----
    tl.atomic_add(gccw_ptr + 0, g_acoef0 * mod_c0); tl.atomic_add(gccw_ptr + 1, g_acoef0 * mod_c1); tl.atomic_add(gccw_ptr + 2, g_acoef0 * mod_c2)
    tl.atomic_add(gccw_ptr + 3, g_acoef1 * mod_c0); tl.atomic_add(gccw_ptr + 4, g_acoef1 * mod_c1); tl.atomic_add(gccw_ptr + 5, g_acoef1 * mod_c2)
    tl.atomic_add(gccw_ptr + 6, g_acoef2 * mod_c0); tl.atomic_add(gccw_ptr + 7, g_acoef2 * mod_c1); tl.atomic_add(gccw_ptr + 8, g_acoef2 * mod_c2)

    tl.atomic_add(gpcw_ptr + 0, gf0 * mod_p0);  tl.atomic_add(gpcw_ptr + 1, gf0 * mod_p1);  tl.atomic_add(gpcw_ptr + 2, gf0 * mod_p2)
    tl.atomic_add(gpcw_ptr + 3, gf1 * mod_p0);  tl.atomic_add(gpcw_ptr + 4, gf1 * mod_p1);  tl.atomic_add(gpcw_ptr + 5, gf1 * mod_p2)
    tl.atomic_add(gpcw_ptr + 6, gf2 * mod_p0);  tl.atomic_add(gpcw_ptr + 7, gf2 * mod_p1);  tl.atomic_add(gpcw_ptr + 8, gf2 * mod_p2)
    tl.atomic_add(gpcw_ptr + 9, gf3 * mod_p0);  tl.atomic_add(gpcw_ptr + 10, gf3 * mod_p1); tl.atomic_add(gpcw_ptr + 11, gf3 * mod_p2)
    tl.atomic_add(gpcw_ptr + 12, gf4 * mod_p0); tl.atomic_add(gpcw_ptr + 13, gf4 * mod_p1); tl.atomic_add(gpcw_ptr + 14, gf4 * mod_p2)
    tl.atomic_add(gpcw_ptr + 15, gf5 * mod_p0); tl.atomic_add(gpcw_ptr + 16, gf5 * mod_p1); tl.atomic_add(gpcw_ptr + 17, gf5 * mod_p2)
    tl.atomic_add(gpcw_ptr + 18, gf6 * mod_p0); tl.atomic_add(gpcw_ptr + 19, gf6 * mod_p1); tl.atomic_add(gpcw_ptr + 20, gf6 * mod_p2)
    tl.atomic_add(gpcw_ptr + 21, gf7 * mod_p0); tl.atomic_add(gpcw_ptr + 22, gf7 * mod_p1); tl.atomic_add(gpcw_ptr + 23, gf7 * mod_p2)
    tl.atomic_add(gpcw_ptr + 24, gf8 * mod_p0); tl.atomic_add(gpcw_ptr + 25, gf8 * mod_p1); tl.atomic_add(gpcw_ptr + 26, gf8 * mod_p2)

    g_mod_c0 = g_acoef0 * ccw00 + g_acoef1 * ccw10 + g_acoef2 * ccw20
    g_mod_c1 = g_acoef0 * ccw01 + g_acoef1 * ccw11 + g_acoef2 * ccw21
    g_mod_c2 = g_acoef0 * ccw02 + g_acoef1 * ccw12 + g_acoef2 * ccw22
    g_routed_c0 = g_mod_c0 * (1.0 - mod_c0 * mod_c0)
    g_routed_c1 = g_mod_c1 * (1.0 - mod_c1 * mod_c1)
    g_routed_c2 = g_mod_c2 * (1.0 - mod_c2 * mod_c2)

    g_mod_p0 = (gf0 * pcw00 + gf1 * pcw10 + gf2 * pcw20 + gf3 * pcw30 + gf4 * pcw40
                + gf5 * pcw50 + gf6 * pcw60 + gf7 * pcw70 + gf8 * pcw80)
    g_mod_p1 = (gf0 * pcw01 + gf1 * pcw11 + gf2 * pcw21 + gf3 * pcw31 + gf4 * pcw41
                + gf5 * pcw51 + gf6 * pcw61 + gf7 * pcw71 + gf8 * pcw81)
    g_mod_p2 = (gf0 * pcw02 + gf1 * pcw12 + gf2 * pcw22 + gf3 * pcw32 + gf4 * pcw42
                + gf5 * pcw52 + gf6 * pcw62 + gf7 * pcw72 + gf8 * pcw82)
    g_routed_p0 = g_mod_p0 * (1.0 - mod_p0 * mod_p0)
    g_routed_p1 = g_mod_p1 * (1.0 - mod_p1 * mod_p1)
    g_routed_p2 = g_mod_p2 * (1.0 - mod_p2 * mod_p2)

    # ---- router/RMSNorm input-grads: mbar reductions + grad_rw/grad_nw atomics ----
    nrm_c = act * rstd_c
    nrm_p = active * rstd_p
    scl_c = nrm_c * nw * rscale
    scl_p = nrm_p * nw * rscale
    gsc_c = g_routed_c0 * rw0 + g_routed_c1 * rw1 + g_routed_c2 * rw2
    gsc_p = g_routed_p0 * rw0 + g_routed_p1 * rw1 + g_routed_p2 * rw2
    g_normed_c = gsc_c * rscale
    g_normed_p = gsc_p * rscale
    g_norm_c = g_normed_c * nw
    g_norm_p = g_normed_p * nw
    mbar_c = tl.sum(g_norm_c * act, axis=0) * inv_H
    mbar_p = tl.sum(g_norm_p * active, axis=0) * inv_H

    tl.atomic_add(grw_ptr + 0 * H + h, g_routed_c0 * scl_c + g_routed_p0 * scl_p, mask=m)
    tl.atomic_add(grw_ptr + 1 * H + h, g_routed_c1 * scl_c + g_routed_p1 * scl_p, mask=m)
    tl.atomic_add(grw_ptr + 2 * H + h, g_routed_c2 * scl_c + g_routed_p2 * scl_p, mask=m)
    tl.atomic_add(gnw_ptr + h, g_normed_c * nrm_c + g_normed_p * nrm_p, mask=m)

    rstd_c3 = rstd_c * rstd_c * rstd_c
    rstd_p3 = rstd_p * rstd_p * rstd_p

    # ---- stores ----
    ga = g_innov + g_norm_c * rstd_c - act * rstd_c3 * mbar_c
    tl.store(ga_ptr + base_act + h, ga.to(tl.bfloat16), mask=m)

    g_active = g_norm_p * rstd_p - active * rstd_p3 * mbar_p
    ghs0 = gpred0 + gpred0 * C00 + gpred1 * C01 + gpred2 * C02
    ghs1 = gpred1 + gpred0 * C10 + gpred1 * C11 + gpred2 * C12
    ghs2 = gpred2 + gpred0 * C20 + gpred1 * C21 + gpred2 * C22
    if IDX == 0:
        ghs0 = ghs0 + g_active
    elif IDX == 1:
        ghs1 = ghs1 + g_active
    else:
        ghs2 = ghs2 + g_active
    tl.store(ghs_ptr + base0 + h, ghs0.to(tl.bfloat16), mask=m)
    tl.store(ghs_ptr + base1 + h, ghs1.to(tl.bfloat16), mask=m)
    tl.store(ghs_ptr + base2 + h, ghs2.to(tl.bfloat16), mask=m)


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
    BLOCK_H = triton.next_power_of_2(H)   # 4096; single resident block over H

    grid = (rows,)
    _altup_bwd_kernel[grid](
        grad_corrected, hidden_states, activated,
        prediction_coef_weight, correction_coef_weight, router_weight, norm_weight,
        grad_hidden_states, grad_activated, grad_pcw, grad_ccw, grad_rw, grad_nw,
        rows, eps,
        H=H, IDX=idx, BLOCK_H=BLOCK_H, num_warps=16,
    )

    return (
        grad_hidden_states,
        grad_activated,
        grad_pcw,
        grad_ccw,
        grad_rw,
        grad_nw,
    )
