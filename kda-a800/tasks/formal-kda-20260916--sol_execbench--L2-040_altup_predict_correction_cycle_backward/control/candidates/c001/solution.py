import torch
import triton
import triton.language as tl


@triton.jit
def _altup_bwd_kernel(
    grad_corrected_ptr, hidden_ptr, activated_ptr,
    pcw_ptr, ccw_ptr, rw_ptr, nw_ptr,
    ghs_ptr, ga_ptr, gpcw_ptr, gccw_ptr, grw_ptr, gnw_ptr,
    rows, eps,
    H: tl.constexpr, IDX: tl.constexpr, BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK_H)
    mask = h < H
    inv_H = 1.0 / H
    router_scale = 1.0 / H

    hbase = row * H
    # variant k base offset = (k*rows + row)*H  (row is int64 -> offsets promote to int64)
    off0 = (0 * rows + row) * H + h
    off1 = (1 * rows + row) * H + h
    off2 = (2 * rows + row) * H + h

    h0 = tl.load(hidden_ptr + off0, mask=mask, other=0.0).to(tl.float32)
    h1 = tl.load(hidden_ptr + off1, mask=mask, other=0.0).to(tl.float32)
    h2 = tl.load(hidden_ptr + off2, mask=mask, other=0.0).to(tl.float32)
    gc0 = tl.load(grad_corrected_ptr + off0, mask=mask, other=0.0).to(tl.float32)
    gc1 = tl.load(grad_corrected_ptr + off1, mask=mask, other=0.0).to(tl.float32)
    gc2 = tl.load(grad_corrected_ptr + off2, mask=mask, other=0.0).to(tl.float32)
    act = tl.load(activated_ptr + hbase + h, mask=mask, other=0.0).to(tl.float32)
    nw = tl.load(nw_ptr + h, mask=mask, other=0.0).to(tl.float32)
    rw0 = tl.load(rw_ptr + 0 * H + h, mask=mask, other=0.0).to(tl.float32)
    rw1 = tl.load(rw_ptr + 1 * H + h, mask=mask, other=0.0).to(tl.float32)
    rw2 = tl.load(rw_ptr + 2 * H + h, mask=mask, other=0.0).to(tl.float32)

    hlist = [h0, h1, h2]
    gclist = [gc0, gc1, gc2]
    rwlist = [rw0, rw1, rw2]

    # small weight tiles (scalars), upcast to fp32
    pcw = [[tl.load(pcw_ptr + p * 3 + q).to(tl.float32) for q in range(3)] for p in range(9)]
    ccw = [[tl.load(ccw_ptr + i * 3 + q).to(tl.float32) for q in range(3)] for i in range(3)]

    # ---------------- forward recompute: predict step ----------------
    active = hlist[IDX]
    var_p = tl.sum(active * active, axis=0) * inv_H
    rstd_p = 1.0 / tl.sqrt(var_p + eps)
    nrm_p = active * rstd_p                       # normalized (before norm_weight)
    scl_p = nrm_p * nw * router_scale             # scaled input to router
    routed_p = [tl.sum(scl_p * rwlist[k], axis=0) for k in range(3)]
    mod_p = [2.0 * tl.sigmoid(2.0 * routed_p[k]) - 1.0 for k in range(3)]

    flat = [mod_p[0] * pcw[p][0] + mod_p[1] * pcw[p][1] + mod_p[2] * pcw[p][2] for p in range(9)]
    # C[m][n] = flat[n*3 + m]
    C = [[flat[n * 3 + m] for n in range(3)] for m in range(3)]

    # predictions[IDX] = sum_k hidden[k]*C[k][IDX] + hidden[IDX]
    pred_idx = hlist[IDX] + h0 * C[0][IDX] + h1 * C[1][IDX] + h2 * C[2][IDX]
    innov = act - pred_idx

    # ---------------- forward recompute: correct step ----------------
    var_c = tl.sum(act * act, axis=0) * inv_H
    rstd_c = 1.0 / tl.sqrt(var_c + eps)
    nrm_c = act * rstd_c
    scl_c = nrm_c * nw * router_scale
    routed_c = [tl.sum(scl_c * rwlist[k], axis=0) for k in range(3)]
    mod_c = [2.0 * tl.sigmoid(2.0 * routed_c[k]) - 1.0 for k in range(3)]
    acoef = [mod_c[0] * ccw[n][0] + mod_c[1] * ccw[n][1] + mod_c[2] * ccw[n][2] + 1.0 for n in range(3)]

    # ---------------- backward: correct step ----------------
    g_innov = gc0 * acoef[0] + gc1 * acoef[1] + gc2 * acoef[2]
    g_acoef = [tl.sum(gclist[n] * innov, axis=0) for n in range(3)]

    # grad_correction_coef_weight[p,q] += g_acoef[p]*mod_c[q]
    for p in range(3):
        for q in range(3):
            tl.atomic_add(gccw_ptr + p * 3 + q, g_acoef[p] * mod_c[q])

    g_mod_c = [g_acoef[0] * ccw[0][q] + g_acoef[1] * ccw[1][q] + g_acoef[2] * ccw[2][q] for q in range(3)]

    ga = g_innov
    # gpred[k] = gc[k]; gpred[IDX] -= g_innov  (do BEFORE predict-backward use)
    gpred = [gclist[k] for k in range(3)]
    gpred[IDX] = gpred[IDX] - g_innov

    # router + rmsnorm backward (correct)
    g_routed_c = [g_mod_c[k] * (1.0 - mod_c[k] * mod_c[k]) for k in range(3)]
    for k in range(3):
        tl.atomic_add(grw_ptr + k * H + h, g_routed_c[k] * scl_c, mask=mask)
    g_scaled_c = g_routed_c[0] * rw0 + g_routed_c[1] * rw1 + g_routed_c[2] * rw2
    g_normed_c = g_scaled_c * router_scale
    tl.atomic_add(gnw_ptr + h, g_normed_c * nrm_c, mask=mask)
    g_norm_c = g_normed_c * nw
    mbar_c = tl.sum(g_norm_c * act, axis=0) * inv_H
    ga = ga + g_norm_c * rstd_c - act * (rstd_c * rstd_c * rstd_c) * mbar_c

    tl.store(ga_ptr + hbase + h, ga.to(tl.bfloat16), mask=mask)

    # ---------------- backward: predict step ----------------
    # ghs[m] = gpred[m] + sum_n gpred[n]*C[m][n]
    ghs = [gpred[m] + gpred[0] * C[m][0] + gpred[1] * C[m][1] + gpred[2] * C[m][2] for m in range(3)]

    # g_flat[3*a + c] = sum_h hidden[c]*gpred[a]
    g_flat = [tl.sum(hlist[c] * gpred[a], axis=0) for a in range(3) for c in range(3)]
    # index: g_flat[a*3 + c]

    for p in range(9):
        for q in range(3):
            tl.atomic_add(gpcw_ptr + p * 3 + q, g_flat[p] * mod_p[q])

    g_mod_p = [sum(g_flat[p] * pcw[p][q] for p in range(9)) for q in range(3)]

    g_routed_p = [g_mod_p[k] * (1.0 - mod_p[k] * mod_p[k]) for k in range(3)]
    for k in range(3):
        tl.atomic_add(grw_ptr + k * H + h, g_routed_p[k] * scl_p, mask=mask)
    g_scaled_p = g_routed_p[0] * rw0 + g_routed_p[1] * rw1 + g_routed_p[2] * rw2
    g_normed_p = g_scaled_p * router_scale
    tl.atomic_add(gnw_ptr + h, g_normed_p * nrm_p, mask=mask)
    g_norm_p = g_normed_p * nw
    mbar_p = tl.sum(g_norm_p * active, axis=0) * inv_H
    g_active_input = g_norm_p * rstd_p - active * (rstd_p * rstd_p * rstd_p) * mbar_p

    ghs[IDX] = ghs[IDX] + g_active_input

    tl.store(ghs_ptr + off0, ghs[0].to(tl.bfloat16), mask=mask)
    tl.store(ghs_ptr + off1, ghs[1].to(tl.bfloat16), mask=mask)
    tl.store(ghs_ptr + off2, ghs[2].to(tl.bfloat16), mask=mask)


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
    BLOCK_H = triton.next_power_of_2(H)

    grid = (rows,)
    _altup_bwd_kernel[grid](
        grad_corrected, hidden_states, activated,
        prediction_coef_weight, correction_coef_weight, router_weight, norm_weight,
        grad_hidden_states, grad_activated, grad_pcw, grad_ccw, grad_rw, grad_nw,
        rows, eps,
        H=H, IDX=idx, BLOCK_H=BLOCK_H, num_warps=8,
    )

    return (
        grad_hidden_states,
        grad_activated,
        grad_pcw,
        grad_ccw,
        grad_rw,
        grad_nw,
    )
