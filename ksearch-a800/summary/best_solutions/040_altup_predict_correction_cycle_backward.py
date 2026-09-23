# task: 040_altup_predict_correction_cycle_backward
# bench: SOL-L2 | batch: formal2_solL2_20260916
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=17.403x
# feedback best (5-workload sample during search): 23.223x
# torch fallback audit: 干净 (-)
# tokens: 3,350,041

import torch
import triton
import triton.language as tl


_H = tl.constexpr(2304)
_INV_H = tl.constexpr(1.0 / 2304.0)


@triton.jit
def _router_modality(
    xpn,
    xcn,
    router_ptr,
    router_offset: tl.constexpr,
    predict_rstd,
    correct_rstd,
    h,
    mask,
):
    rw = tl.load(
        router_ptr + router_offset + h, mask=mask, other=0.0
    ).to(tl.float32)
    predict_raw = tl.sum(xpn * rw, axis=0)
    correct_raw = tl.sum(xcn * rw, axis=0)
    predict_routed = predict_raw * predict_rstd * _INV_H
    correct_routed = correct_raw * correct_rstd * _INV_H
    predict_modality = 2.0 * tl.sigmoid(2.0 * predict_routed) - 1.0
    correct_modality = 2.0 * tl.sigmoid(2.0 * correct_routed) - 1.0
    return predict_modality, correct_modality


@triton.jit
def _rms_modality_prepass_kernel(
    hidden_ptr,
    activated_ptr,
    router_ptr,
    norm_ptr,
    predict_rstd_ptr,
    correct_rstd_ptr,
    predict_modalities_ptr,
    correct_modalities_ptr,
    num_tokens: tl.constexpr,
    active_idx: tl.constexpr,
    rms_eps,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < _H

    dense_base = token * _H
    active_base = (active_idx * num_tokens + token) * _H

    xp = tl.load(
        hidden_ptr + active_base + h, mask=mask, other=0.0
    ).to(tl.float32)
    act = tl.load(
        activated_ptr + dense_base + h, mask=mask, other=0.0
    ).to(tl.float32)
    nw = tl.load(norm_ptr + h, mask=mask, other=0.0).to(tl.float32)

    predict_var = tl.sum(xp * xp, axis=0) * _INV_H
    correct_var = tl.sum(act * act, axis=0) * _INV_H
    predict_rstd = tl.rsqrt(predict_var + rms_eps)
    correct_rstd = tl.rsqrt(correct_var + rms_eps)

    xpn = xp * nw
    xcn = act * nw

    mp0, mc0 = _router_modality(
        xpn, xcn, router_ptr, 0, predict_rstd, correct_rstd, h, mask
    )
    mp1, mc1 = _router_modality(
        xpn, xcn, router_ptr, _H, predict_rstd, correct_rstd, h, mask
    )
    mp2, mc2 = _router_modality(
        xpn, xcn, router_ptr, 2 * _H, predict_rstd, correct_rstd, h, mask
    )

    tl.store(predict_rstd_ptr + token, predict_rstd)
    tl.store(correct_rstd_ptr + token, correct_rstd)

    base = token * 3
    tl.store(predict_modalities_ptr + base, mp0)
    tl.store(predict_modalities_ptr + base + 1, mp1)
    tl.store(predict_modalities_ptr + base + 2, mp2)
    tl.store(correct_modalities_ptr + base, mc0)
    tl.store(correct_modalities_ptr + base + 1, mc1)
    tl.store(correct_modalities_ptr + base + 2, mc2)


@triton.jit
def _dense_backward_kernel(
    grad_corrected_ptr,
    hidden_ptr,
    activated_ptr,
    prediction_weight_ptr,
    correction_weight_ptr,
    router_ptr,
    norm_ptr,
    predict_rstd_ptr,
    correct_rstd_ptr,
    predict_modalities_ptr,
    correct_modalities_ptr,
    grad_prediction_coefs_ptr,
    grad_correction_coefs_ptr,
    grad_predict_routed_ptr,
    grad_correct_routed_ptr,
    grad_hidden_ptr,
    grad_activated_ptr,
    num_tokens: tl.constexpr,
    active_idx: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < _H
    dense_base = token * _H
    active_base = (active_idx * num_tokens + token) * _H

    xp = tl.load(
        hidden_ptr + active_base + h, mask=mask, other=0.0
    ).to(tl.float32)
    act = tl.load(
        activated_ptr + dense_base + h, mask=mask, other=0.0
    ).to(tl.float32)
    nw = tl.load(norm_ptr + h, mask=mask, other=0.0).to(tl.float32)

    predict_rstd = tl.load(predict_rstd_ptr + token).to(tl.float32)
    correct_rstd = tl.load(correct_rstd_ptr + token).to(tl.float32)

    modality_base = token * 3
    mp0 = tl.load(predict_modalities_ptr + modality_base).to(tl.float32)
    mp1 = tl.load(predict_modalities_ptr + modality_base + 1).to(tl.float32)
    mp2 = tl.load(predict_modalities_ptr + modality_base + 2).to(tl.float32)
    mc0 = tl.load(correct_modalities_ptr + modality_base).to(tl.float32)
    mc1 = tl.load(correct_modalities_ptr + modality_base + 1).to(tl.float32)
    mc2 = tl.load(correct_modalities_ptr + modality_base + 2).to(tl.float32)

    if active_idx == 0:
        h0 = xp
    else:
        h0 = tl.load(
            hidden_ptr + token * _H + h, mask=mask, other=0.0
        ).to(tl.float32)

    if active_idx == 1:
        h1 = xp
    else:
        h1 = tl.load(
            hidden_ptr + (num_tokens + token) * _H + h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    if active_idx == 2:
        h2 = xp
    else:
        h2 = tl.load(
            hidden_ptr + (2 * num_tokens + token) * _H + h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    gc0 = tl.load(
        grad_corrected_ptr + token * _H + h, mask=mask, other=0.0
    ).to(tl.float32)
    gc1 = tl.load(
        grad_corrected_ptr + (num_tokens + token) * _H + h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gc2 = tl.load(
        grad_corrected_ptr + (2 * num_tokens + token) * _H + h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    pw0 = tl.load(prediction_weight_ptr).to(tl.float32)
    pw1 = tl.load(prediction_weight_ptr + 1).to(tl.float32)
    pw2 = tl.load(prediction_weight_ptr + 2).to(tl.float32)
    pw3 = tl.load(prediction_weight_ptr + 3).to(tl.float32)
    pw4 = tl.load(prediction_weight_ptr + 4).to(tl.float32)
    pw5 = tl.load(prediction_weight_ptr + 5).to(tl.float32)
    pw6 = tl.load(prediction_weight_ptr + 6).to(tl.float32)
    pw7 = tl.load(prediction_weight_ptr + 7).to(tl.float32)
    pw8 = tl.load(prediction_weight_ptr + 8).to(tl.float32)
    pw9 = tl.load(prediction_weight_ptr + 9).to(tl.float32)
    pw10 = tl.load(prediction_weight_ptr + 10).to(tl.float32)
    pw11 = tl.load(prediction_weight_ptr + 11).to(tl.float32)
    pw12 = tl.load(prediction_weight_ptr + 12).to(tl.float32)
    pw13 = tl.load(prediction_weight_ptr + 13).to(tl.float32)
    pw14 = tl.load(prediction_weight_ptr + 14).to(tl.float32)
    pw15 = tl.load(prediction_weight_ptr + 15).to(tl.float32)
    pw16 = tl.load(prediction_weight_ptr + 16).to(tl.float32)
    pw17 = tl.load(prediction_weight_ptr + 17).to(tl.float32)
    pw18 = tl.load(prediction_weight_ptr + 18).to(tl.float32)
    pw19 = tl.load(prediction_weight_ptr + 19).to(tl.float32)
    pw20 = tl.load(prediction_weight_ptr + 20).to(tl.float32)
    pw21 = tl.load(prediction_weight_ptr + 21).to(tl.float32)
    pw22 = tl.load(prediction_weight_ptr + 22).to(tl.float32)
    pw23 = tl.load(prediction_weight_ptr + 23).to(tl.float32)
    pw24 = tl.load(prediction_weight_ptr + 24).to(tl.float32)
    pw25 = tl.load(prediction_weight_ptr + 25).to(tl.float32)
    pw26 = tl.load(prediction_weight_ptr + 26).to(tl.float32)

    p00 = mp0 * pw0 + mp1 * pw1 + mp2 * pw2
    p01 = mp0 * pw3 + mp1 * pw4 + mp2 * pw5
    p02 = mp0 * pw6 + mp1 * pw7 + mp2 * pw8
    p10 = mp0 * pw9 + mp1 * pw10 + mp2 * pw11
    p11 = mp0 * pw12 + mp1 * pw13 + mp2 * pw14
    p12 = mp0 * pw15 + mp1 * pw16 + mp2 * pw17
    p20 = mp0 * pw18 + mp1 * pw19 + mp2 * pw20
    p21 = mp0 * pw21 + mp1 * pw22 + mp2 * pw23
    p22 = mp0 * pw24 + mp1 * pw25 + mp2 * pw26

    if active_idx == 0:
        pred_active = h0 + p00 * h0 + p01 * h1 + p02 * h2
    elif active_idx == 1:
        pred_active = h1 + p10 * h0 + p11 * h1 + p12 * h2
    else:
        pred_active = h2 + p20 * h0 + p21 * h1 + p22 * h2

    innovation = act - pred_active

    cw0 = tl.load(correction_weight_ptr).to(tl.float32)
    cw1 = tl.load(correction_weight_ptr + 1).to(tl.float32)
    cw2 = tl.load(correction_weight_ptr + 2).to(tl.float32)
    cw3 = tl.load(correction_weight_ptr + 3).to(tl.float32)
    cw4 = tl.load(correction_weight_ptr + 4).to(tl.float32)
    cw5 = tl.load(correction_weight_ptr + 5).to(tl.float32)
    cw6 = tl.load(correction_weight_ptr + 6).to(tl.float32)
    cw7 = tl.load(correction_weight_ptr + 7).to(tl.float32)
    cw8 = tl.load(correction_weight_ptr + 8).to(tl.float32)

    c0 = 1.0 + mc0 * cw0 + mc1 * cw1 + mc2 * cw2
    c1 = 1.0 + mc0 * cw3 + mc1 * cw4 + mc2 * cw5
    c2 = 1.0 + mc0 * cw6 + mc1 * cw7 + mc2 * cw8

    grad_innovation = gc0 * c0 + gc1 * c1 + gc2 * c2

    if active_idx == 0:
        gp0 = gc0 - grad_innovation
        gp1 = gc1
        gp2 = gc2
    elif active_idx == 1:
        gp0 = gc0
        gp1 = gc1 - grad_innovation
        gp2 = gc2
    else:
        gp0 = gc0
        gp1 = gc1
        gp2 = gc2 - grad_innovation

    gpc00 = tl.sum(gp0 * h0, axis=0)
    gpc01 = tl.sum(gp0 * h1, axis=0)
    gpc02 = tl.sum(gp0 * h2, axis=0)
    gpc10 = tl.sum(gp1 * h0, axis=0)
    gpc11 = tl.sum(gp1 * h1, axis=0)
    gpc12 = tl.sum(gp1 * h2, axis=0)
    gpc20 = tl.sum(gp2 * h0, axis=0)
    gpc21 = tl.sum(gp2 * h1, axis=0)
    gpc22 = tl.sum(gp2 * h2, axis=0)

    gcc0 = tl.sum(gc0 * innovation, axis=0)
    gcc1 = tl.sum(gc1 * innovation, axis=0)
    gcc2 = tl.sum(gc2 * innovation, axis=0)

    gp_base = token * 9
    tl.store(grad_prediction_coefs_ptr + gp_base, gpc00)
    tl.store(grad_prediction_coefs_ptr + gp_base + 1, gpc01)
    tl.store(grad_prediction_coefs_ptr + gp_base + 2, gpc02)
    tl.store(grad_prediction_coefs_ptr + gp_base + 3, gpc10)
    tl.store(grad_prediction_coefs_ptr + gp_base + 4, gpc11)
    tl.store(grad_prediction_coefs_ptr + gp_base + 5, gpc12)
    tl.store(grad_prediction_coefs_ptr + gp_base + 6, gpc20)
    tl.store(grad_prediction_coefs_ptr + gp_base + 7, gpc21)
    tl.store(grad_prediction_coefs_ptr + gp_base + 8, gpc22)

    gc_base = token * 3
    tl.store(grad_correction_coefs_ptr + gc_base, gcc0)
    tl.store(grad_correction_coefs_ptr + gc_base + 1, gcc1)
    tl.store(grad_correction_coefs_ptr + gc_base + 2, gcc2)

    gmp0 = (
        gpc00 * pw0 + gpc01 * pw3 + gpc02 * pw6
        + gpc10 * pw9 + gpc11 * pw12 + gpc12 * pw15
        + gpc20 * pw18 + gpc21 * pw21 + gpc22 * pw24
    )
    gmp1 = (
        gpc00 * pw1 + gpc01 * pw4 + gpc02 * pw7
        + gpc10 * pw10 + gpc11 * pw13 + gpc12 * pw16
        + gpc20 * pw19 + gpc21 * pw22 + gpc22 * pw25
    )
    gmp2 = (
        gpc00 * pw2 + gpc01 * pw5 + gpc02 * pw8
        + gpc10 * pw11 + gpc11 * pw14 + gpc12 * pw17
        + gpc20 * pw20 + gpc21 * pw23 + gpc22 * pw26
    )

    gmc0 = gcc0 * cw0 + gcc1 * cw3 + gcc2 * cw6
    gmc1 = gcc0 * cw1 + gcc1 * cw4 + gcc2 * cw7
    gmc2 = gcc0 * cw2 + gcc1 * cw5 + gcc2 * cw8

    grp0 = gmp0 * (1.0 - mp0 * mp0)
    grp1 = gmp1 * (1.0 - mp1 * mp1)
    grp2 = gmp2 * (1.0 - mp2 * mp2)
    grc0 = gmc0 * (1.0 - mc0 * mc0)
    grc1 = gmc1 * (1.0 - mc1 * mc1)
    grc2 = gmc2 * (1.0 - mc2 * mc2)

    tl.store(grad_predict_routed_ptr + modality_base, grp0)
    tl.store(grad_predict_routed_ptr + modality_base + 1, grp1)
    tl.store(grad_predict_routed_ptr + modality_base + 2, grp2)
    tl.store(grad_correct_routed_ptr + modality_base, grc0)
    tl.store(grad_correct_routed_ptr + modality_base + 1, grc1)
    tl.store(grad_correct_routed_ptr + modality_base + 2, grc2)

    rw0 = tl.load(router_ptr + h, mask=mask, other=0.0).to(tl.float32)
    rw1 = tl.load(
        router_ptr + _H + h, mask=mask, other=0.0
    ).to(tl.float32)
    rw2 = tl.load(
        router_ptr + 2 * _H + h, mask=mask, other=0.0
    ).to(tl.float32)

    grad_predict_normalized = (
        grp0 * rw0 + grp1 * rw1 + grp2 * rw2
    ) * _INV_H * nw
    grad_correct_normalized = (
        grc0 * rw0 + grc1 * rw1 + grc2 * rw2
    ) * _INV_H * nw

    predict_dot = tl.sum(
        grad_predict_normalized * xp, axis=0
    ) * _INV_H
    correct_dot = tl.sum(
        grad_correct_normalized * act, axis=0
    ) * _INV_H

    predict_rstd3 = predict_rstd * predict_rstd * predict_rstd
    correct_rstd3 = correct_rstd * correct_rstd * correct_rstd

    grad_active = (
        grad_predict_normalized * predict_rstd
        - xp * predict_rstd3 * predict_dot
    )
    grad_act_router = (
        grad_correct_normalized * correct_rstd
        - act * correct_rstd3 * correct_dot
    )

    gh0 = gp0 + p00 * gp0 + p10 * gp1 + p20 * gp2
    gh1 = gp1 + p01 * gp0 + p11 * gp1 + p21 * gp2
    gh2 = gp2 + p02 * gp0 + p12 * gp1 + p22 * gp2

    if active_idx == 0:
        gh0 += grad_active
    elif active_idx == 1:
        gh1 += grad_active
    else:
        gh2 += grad_active

    tl.store(grad_hidden_ptr + token * _H + h, gh0, mask=mask)
    tl.store(
        grad_hidden_ptr + (num_tokens + token) * _H + h,
        gh1,
        mask=mask,
    )
    tl.store(
        grad_hidden_ptr + (2 * num_tokens + token) * _H + h,
        gh2,
        mask=mask,
    )
    tl.store(
        grad_activated_ptr + dense_base + h,
        grad_innovation + grad_act_router,
        mask=mask,
    )


@triton.jit
def _feature_reduce_kernel(
    hidden_ptr,
    activated_ptr,
    router_ptr,
    norm_ptr,
    predict_rstd_ptr,
    correct_rstd_ptr,
    grad_predict_routed_ptr,
    grad_correct_routed_ptr,
    grad_router_ptr,
    grad_norm_ptr,
    num_tokens,
    active_idx: tl.constexpr,
    SINGLE_CHUNK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_block = tl.program_id(0)
    h_block = tl.program_id(1)

    t = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    t2 = t[:, None]
    h2 = h[None, :]

    tmask = t < num_tokens
    mask = tmask[:, None]

    xp = tl.load(
        hidden_ptr + (active_idx * num_tokens + t2) * _H + h2,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    xc = tl.load(
        activated_ptr + t2 * _H + h2,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    rp = tl.load(
        predict_rstd_ptr + t, mask=tmask, other=0.0
    )[:, None].to(tl.float32)
    rc = tl.load(
        correct_rstd_ptr + t, mask=tmask, other=0.0
    )[:, None].to(tl.float32)
    nw = tl.load(norm_ptr + h)[None, :].to(tl.float32)

    grp0 = tl.load(
        grad_predict_routed_ptr + t * 3,
        mask=tmask,
        other=0.0,
    )[:, None].to(tl.float32)
    grp1 = tl.load(
        grad_predict_routed_ptr + t * 3 + 1,
        mask=tmask,
        other=0.0,
    )[:, None].to(tl.float32)
    grp2 = tl.load(
        grad_predict_routed_ptr + t * 3 + 2,
        mask=tmask,
        other=0.0,
    )[:, None].to(tl.float32)

    grc0 = tl.load(
        grad_correct_routed_ptr + t * 3,
        mask=tmask,
        other=0.0,
    )[:, None].to(tl.float32)
    grc1 = tl.load(
        grad_correct_routed_ptr + t * 3 + 1,
        mask=tmask,
        other=0.0,
    )[:, None].to(tl.float32)
    grc2 = tl.load(
        grad_correct_routed_ptr + t * 3 + 2,
        mask=tmask,
        other=0.0,
    )[:, None].to(tl.float32)

    normalized_p = xp * rp
    normalized_c = xc * rc
    scaled_p = normalized_p * nw * _INV_H
    scaled_c = normalized_c * nw * _INV_H

    router0 = tl.sum(
        grp0 * scaled_p + grc0 * scaled_c, axis=0
    )
    router1 = tl.sum(
        grp1 * scaled_p + grc1 * scaled_c, axis=0
    )
    router2 = tl.sum(
        grp2 * scaled_p + grc2 * scaled_c, axis=0
    )

    rw0 = tl.load(router_ptr + h)[None, :].to(tl.float32)
    rw1 = tl.load(router_ptr + _H + h)[None, :].to(tl.float32)
    rw2 = tl.load(router_ptr + 2 * _H + h)[None, :].to(tl.float32)

    grad_normed_p = (
        grp0 * rw0 + grp1 * rw1 + grp2 * rw2
    ) * _INV_H
    grad_normed_c = (
        grc0 * rw0 + grc1 * rw1 + grc2 * rw2
    ) * _INV_H

    norm_grad = tl.sum(
        grad_normed_p * normalized_p
        + grad_normed_c * normalized_c,
        axis=0,
    )

    if SINGLE_CHUNK:
        tl.store(grad_router_ptr + h, router0)
        tl.store(grad_router_ptr + _H + h, router1)
        tl.store(grad_router_ptr + 2 * _H + h, router2)
        tl.store(grad_norm_ptr + h, norm_grad)
    else:
        tl.atomic_add(grad_router_ptr + h, router0)
        tl.atomic_add(grad_router_ptr + _H + h, router1)
        tl.atomic_add(grad_router_ptr + 2 * _H + h, router2)
        tl.atomic_add(grad_norm_ptr + h, norm_grad)


@triton.jit
def _coefficient_reduce_kernel(
    grad_prediction_coefs_ptr,
    grad_correction_coefs_ptr,
    predict_modalities_ptr,
    correct_modalities_ptr,
    grad_prediction_weight_ptr,
    grad_correction_weight_ptr,
    num_tokens,
    SINGLE_CHUNK: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    token_block = tl.program_id(1)

    t = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_tokens

    if row < 9:
        grad_coef = tl.load(
            grad_prediction_coefs_ptr + t * 9 + row,
            mask=tmask,
            other=0.0,
        ).to(tl.float32)
        modality0 = tl.load(
            predict_modalities_ptr + t * 3,
            mask=tmask,
            other=0.0,
        ).to(tl.float32)
        modality1 = tl.load(
            predict_modalities_ptr + t * 3 + 1,
            mask=tmask,
            other=0.0,
        ).to(tl.float32)
        modality2 = tl.load(
            predict_modalities_ptr + t * 3 + 2,
            mask=tmask,
            other=0.0,
        ).to(tl.float32)

        value0 = tl.sum(grad_coef * modality0, axis=0)
        value1 = tl.sum(grad_coef * modality1, axis=0)
        value2 = tl.sum(grad_coef * modality2, axis=0)
        output_base = row * 3

        if SINGLE_CHUNK:
            tl.store(grad_prediction_weight_ptr + output_base, value0)
            tl.store(grad_prediction_weight_ptr + output_base + 1, value1)
            tl.store(grad_prediction_weight_ptr + output_base + 2, value2)
        else:
            tl.atomic_add(
                grad_prediction_weight_ptr + output_base, value0
            )
            tl.atomic_add(
                grad_prediction_weight_ptr + output_base + 1, value1
            )
            tl.atomic_add(
                grad_prediction_weight_ptr + output_base + 2, value2
            )
    else:
        correction_row = row - 9
        grad_coef = tl.load(
            grad_correction_coefs_ptr + t * 3 + correction_row,
            mask=tmask,
            other=0.0,
        ).to(tl.float32)
        modality0 = tl.load(
            correct_modalities_ptr + t * 3,
            mask=tmask,
            other=0.0,
        ).to(tl.float32)
        modality1 = tl.load(
            correct_modalities_ptr + t * 3 + 1,
            mask=tmask,
            other=0.0,
        ).to(tl.float32)
        modality2 = tl.load(
            correct_modalities_ptr + t * 3 + 2,
            mask=tmask,
            other=0.0,
        ).to(tl.float32)

        value0 = tl.sum(grad_coef * modality0, axis=0)
        value1 = tl.sum(grad_coef * modality1, axis=0)
        value2 = tl.sum(grad_coef * modality2, axis=0)
        output_base = correction_row * 3

        if SINGLE_CHUNK:
            tl.store(grad_correction_weight_ptr + output_base, value0)
            tl.store(grad_correction_weight_ptr + output_base + 1, value1)
            tl.store(grad_correction_weight_ptr + output_base + 2, value2)
        else:
            tl.atomic_add(
                grad_correction_weight_ptr + output_base, value0
            )
            tl.atomic_add(
                grad_correction_weight_ptr + output_base + 1, value1
            )
            tl.atomic_add(
                grad_correction_weight_ptr + output_base + 2, value2
            )


@torch.no_grad()
def run(
    grad_corrected: torch.Tensor,
    hidden_states: torch.Tensor,
    activated: torch.Tensor,
    prediction_coef_weight: torch.Tensor,
    correction_coef_weight: torch.Tensor,
    router_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    altup_active_idx: int,
    rms_norm_eps: float,
):
    if isinstance(altup_active_idx, torch.Tensor):
        altup_active_idx = int(altup_active_idx.item())
    else:
        altup_active_idx = int(altup_active_idx)

    if isinstance(rms_norm_eps, torch.Tensor):
        rms_norm_eps = float(rms_norm_eps.item())
    else:
        rms_norm_eps = float(rms_norm_eps)

    num_tokens = hidden_states.shape[1] * hidden_states.shape[2]
    device = hidden_states.device

    if num_tokens >= 16384:
        prepass_num_warps = 4
        dense_num_warps = 4
        block_t = 128
        block_h = 32
        feature_num_warps = 8
    elif num_tokens >= 128:
        prepass_num_warps = 8
        dense_num_warps = 8
        block_t = 128
        block_h = 32
        feature_num_warps = 8
    elif num_tokens >= 64:
        prepass_num_warps = 8
        dense_num_warps = 8
        block_t = 64
        block_h = 64
        feature_num_warps = 8
    else:
        prepass_num_warps = 4
        dense_num_warps = 4
        block_t = 32
        block_h = 64
        feature_num_warps = 4

    feature_single_chunk = num_tokens <= block_t
    coefficient_block = 1024
    coefficient_chunks = triton.cdiv(num_tokens, coefficient_block)
    coefficient_single_chunk = num_tokens <= coefficient_block

    intermediates = torch.empty(
        num_tokens * 26, device=device, dtype=torch.float32
    )

    offset = 0
    predict_rstd = intermediates[offset:offset + num_tokens]
    offset += num_tokens
    correct_rstd = intermediates[offset:offset + num_tokens]
    offset += num_tokens

    predict_modalities = intermediates[
        offset:offset + num_tokens * 3
    ].view(num_tokens, 3)
    offset += num_tokens * 3

    correct_modalities = intermediates[
        offset:offset + num_tokens * 3
    ].view(num_tokens, 3)
    offset += num_tokens * 3

    grad_prediction_coefs = intermediates[
        offset:offset + num_tokens * 9
    ].view(num_tokens, 9)
    offset += num_tokens * 9

    grad_correction_coefs = intermediates[
        offset:offset + num_tokens * 3
    ].view(num_tokens, 3)
    offset += num_tokens * 3

    grad_predict_routed = intermediates[
        offset:offset + num_tokens * 3
    ].view(num_tokens, 3)
    offset += num_tokens * 3

    grad_correct_routed = intermediates[
        offset:offset + num_tokens * 3
    ].view(num_tokens, 3)

    grad_hidden_states = torch.empty_like(hidden_states)
    grad_activated = torch.empty_like(activated)

    if feature_single_chunk and coefficient_single_chunk:
        parameter_grads = torch.empty(
            9252, device=device, dtype=torch.float32
        )
    else:
        parameter_grads = torch.zeros(
            9252, device=device, dtype=torch.float32
        )

    grad_prediction_coef_weight = parameter_grads[:27].view(9, 3)
    grad_correction_coef_weight = parameter_grads[27:36].view(3, 3)
    grad_router_weight = parameter_grads[36:6948].view(3, 2304)
    grad_norm_weight = parameter_grads[6948:9252]

    _rms_modality_prepass_kernel[(num_tokens,)](
        hidden_states,
        activated,
        router_weight,
        norm_weight,
        predict_rstd,
        correct_rstd,
        predict_modalities,
        correct_modalities,
        num_tokens=num_tokens,
        active_idx=altup_active_idx,
        rms_eps=rms_norm_eps,
        BLOCK_H=4096,
        num_warps=prepass_num_warps,
    )

    _dense_backward_kernel[(num_tokens,)](
        grad_corrected,
        hidden_states,
        activated,
        prediction_coef_weight,
        correction_coef_weight,
        router_weight,
        norm_weight,
        predict_rstd,
        correct_rstd,
        predict_modalities,
        correct_modalities,
        grad_prediction_coefs,
        grad_correction_coefs,
        grad_predict_routed,
        grad_correct_routed,
        grad_hidden_states,
        grad_activated,
        num_tokens=num_tokens,
        active_idx=altup_active_idx,
        BLOCK_H=4096,
        num_warps=dense_num_warps,
    )

    _feature_reduce_kernel[
        (triton.cdiv(num_tokens, block_t), 2304 // block_h)
    ](
        hidden_states,
        activated,
        router_weight,
        norm_weight,
        predict_rstd,
        correct_rstd,
        grad_predict_routed,
        grad_correct_routed,
        grad_router_weight,
        grad_norm_weight,
        num_tokens,
        active_idx=altup_active_idx,
        SINGLE_CHUNK=feature_single_chunk,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        num_warps=feature_num_warps,
    )

    _coefficient_reduce_kernel[(12, coefficient_chunks)](
        grad_prediction_coefs,
        grad_correction_coefs,
        predict_modalities,
        correct_modalities,
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        num_tokens,
        SINGLE_CHUNK=coefficient_single_chunk,
        BLOCK_T=coefficient_block,
        num_warps=8,
    )

    return (
        grad_hidden_states,
        grad_activated,
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        grad_router_weight,
        grad_norm_weight,
    )