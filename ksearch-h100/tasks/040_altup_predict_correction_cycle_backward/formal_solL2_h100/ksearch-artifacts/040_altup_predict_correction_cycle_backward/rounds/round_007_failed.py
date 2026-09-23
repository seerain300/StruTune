# solution=GPT-5.6-Sol_040_altup_predict_correction_cycle_backward_triton_optimized_r7 score=-1.0 passed=False
I’m checking the repository layout and the active kernel file so I can make the smallest performance change against the actual source rather than reconstituting it from the prompt.import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 2304
_ALTUP_NUM_INPUTS = 3


@triton.jit
def _backward_kernel(
    grad_corrected,
    hidden_states,
    activated,
    prediction_coef_weight,
    correction_coef_weight,
    router_weight,
    norm_weight,
    grad_hidden_states,
    grad_activated,
    grad_prediction_coef_weight,
    grad_correction_coef_weight,
    grad_router_weight,
    grad_norm_weight,
    num_tokens: tl.constexpr,
    active_idx: tl.constexpr,
    rms_norm_eps,
    hidden_size: tl.constexpr,
    block_k: tl.constexpr,
):
    token = tl.program_id(0)
    k = tl.arange(0, block_k)
    mask = k < hidden_size

    hs0 = hidden_states + token * hidden_size + k
    hs1 = hidden_states + (num_tokens + token) * hidden_size + k
    hs2 = hidden_states + (2 * num_tokens + token) * hidden_size + k

    gc0_ptr = grad_corrected + token * hidden_size + k
    gc1_ptr = grad_corrected + (num_tokens + token) * hidden_size + k
    gc2_ptr = grad_corrected + (2 * num_tokens + token) * hidden_size + k

    h0 = tl.load(hs0, mask=mask, other=0.0).to(tl.float32)
    h1 = tl.load(hs1, mask=mask, other=0.0).to(tl.float32)
    h2 = tl.load(hs2, mask=mask, other=0.0).to(tl.float32)

    g0 = tl.load(gc0_ptr, mask=mask, other=0.0).to(tl.float32)
    g1 = tl.load(gc1_ptr, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(gc2_ptr, mask=mask, other=0.0).to(tl.float32)

    av = tl.load(
        activated + token * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    norm = tl.load(norm_weight + k, mask=mask, other=0.0).to(tl.float32)

    router0 = tl.load(
        router_weight + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    router1 = tl.load(
        router_weight + hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    router2 = tl.load(
        router_weight + 2 * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    router_scale = 1.0 / 2304.0

    if active_idx == 0:
        active_hidden = h0
    elif active_idx == 1:
        active_hidden = h1
    else:
        active_hidden = h2

    predict_var = tl.sum(active_hidden * active_hidden, axis=0) / hidden_size
    correct_var = tl.sum(av * av, axis=0) / hidden_size

    rstd_predict = tl.rsqrt(predict_var + rms_norm_eps)
    rstd_correct = tl.rsqrt(correct_var + rms_norm_eps)

    normalized_predict = active_hidden * rstd_predict
    normalized_correct = av * rstd_correct

    scaled_predict = normalized_predict * norm * router_scale
    scaled_correct = normalized_correct * norm * router_scale

    routed_predict0 = tl.sum(scaled_predict * router0, axis=0)
    routed_predict1 = tl.sum(scaled_predict * router1, axis=0)
    routed_predict2 = tl.sum(scaled_predict * router2, axis=0)

    routed_correct0 = tl.sum(scaled_correct * router0, axis=0)
    routed_correct1 = tl.sum(scaled_correct * router1, axis=0)
    routed_correct2 = tl.sum(scaled_correct * router2, axis=0)

    mp0 = tl.tanh(routed_predict0)
    mp1 = tl.tanh(routed_predict1)
    mp2 = tl.tanh(routed_predict2)

    mc0 = tl.tanh(routed_correct0)
    mc1 = tl.tanh(routed_correct1)
    mc2 = tl.tanh(routed_correct2)

    pw0 = tl.load(prediction_coef_weight + 0).to(tl.float32)
    pw1 = tl.load(prediction_coef_weight + 1).to(tl.float32)
    pw2 = tl.load(prediction_coef_weight + 2).to(tl.float32)
    pw3 = tl.load(prediction_coef_weight + 3).to(tl.float32)
    pw4 = tl.load(prediction_coef_weight + 4).to(tl.float32)
    pw5 = tl.load(prediction_coef_weight + 5).to(tl.float32)
    pw6 = tl.load(prediction_coef_weight + 6).to(tl.float32)
    pw7 = tl.load(prediction_coef_weight + 7).to(tl.float32)
    pw8 = tl.load(prediction_coef_weight + 8).to(tl.float32)
    pw9 = tl.load(prediction_coef_weight + 9).to(tl.float32)
    pw10 = tl.load(prediction_coef_weight + 10).to(tl.float32)
    pw11 = tl.load(prediction_coef_weight + 11).to(tl.float32)
    pw12 = tl.load(prediction_coef_weight + 12).to(tl.float32)
    pw13 = tl.load(prediction_coef_weight + 13).to(tl.float32)
    pw14 = tl.load(prediction_coef_weight + 14).to(tl.float32)
    pw15 = tl.load(prediction_coef_weight + 15).to(tl.float32)
    pw16 = tl.load(prediction_coef_weight + 16).to(tl.float32)
    pw17 = tl.load(prediction_coef_weight + 17).to(tl.float32)
    pw18 = tl.load(prediction_coef_weight + 18).to(tl.float32)
    pw19 = tl.load(prediction_coef_weight + 19).to(tl.float32)
    pw20 = tl.load(prediction_coef_weight + 20).to(tl.float32)
    pw21 = tl.load(prediction_coef_weight + 21).to(tl.float32)
    pw22 = tl.load(prediction_coef_weight + 22).to(tl.float32)
    pw23 = tl.load(prediction_coef_weight + 23).to(tl.float32)
    pw24 = tl.load(prediction_coef_weight + 24).to(tl.float32)
    pw25 = tl.load(prediction_coef_weight + 25).to(tl.float32)
    pw26 = tl.load(prediction_coef_weight + 26).to(tl.float32)

    p0 = mp0 * pw0 + mp1 * pw1 + mp2 * pw2
    p1 = mp0 * pw3 + mp1 * pw4 + mp2 * pw5
    p2 = mp0 * pw6 + mp1 * pw7 + mp2 * pw8
    p3 = mp0 * pw9 + mp1 * pw10 + mp2 * pw11
    p4 = mp0 * pw12 + mp1 * pw13 + mp2 * pw14
    p5 = mp0 * pw15 + mp1 * pw16 + mp2 * pw17
    p6 = mp0 * pw18 + mp1 * pw19 + mp2 * pw20
    p7 = mp0 * pw21 + mp1 * pw22 + mp2 * pw23
    p8 = mp0 * pw24 + mp1 * pw25 + mp2 * pw26

    cw0 = tl.load(correction_coef_weight + 0).to(tl.float32)
    cw1 = tl.load(correction_coef_weight + 1).to(tl.float32)
    cw2 = tl.load(correction_coef_weight + 2).to(tl.float32)
    cw3 = tl.load(correction_coef_weight + 3).to(tl.float32)
    cw4 = tl.load(correction_coef_weight + 4).to(tl.float32)
    cw5 = tl.load(correction_coef_weight + 5).to(tl.float32)
    cw6 = tl.load(correction_coef_weight + 6).to(tl.float32)
    cw7 = tl.load(correction_coef_weight + 7).to(tl.float32)
    cw8 = tl.load(correction_coef_weight + 8).to(tl.float32)

    c0 = 1.0 + mc0 * cw0 + mc1 * cw1 + mc2 * cw2
    c1 = 1.0 + mc0 * cw3 + mc1 * cw4 + mc2 * cw5
    c2 = 1.0 + mc0 * cw6 + mc1 * cw7 + mc2 * cw8

    if active_idx == 0:
        prediction_active = h0 * p0 + h1 * p3 + h2 * p6 + h0
    elif active_idx == 1:
        prediction_active = h0 * p1 + h1 * p4 + h2 * p7 + h1
    else:
        prediction_active = h0 * p2 + h1 * p5 + h2 * p8 + h2

    innovation = av - prediction_active
    grad_innovation = g0 * c0 + g1 * c1 + g2 * c2

    if active_idx == 0:
        pg0 = g0 - grad_innovation
        pg1 = g1
        pg2 = g2
    elif active_idx == 1:
        pg0 = g0
        pg1 = g1 - grad_innovation
        pg2 = g2
    else:
        pg0 = g0
        pg1 = g1
        pg2 = g2 - grad_innovation

    dmp0 = (
        pg0 * p0
        + pg1 * p3
        + pg2 * p6
    ) * (1.0 - mp0 * mp0)
    dmp1 = (
        pg0 * p1
        + pg1 * p4
        + pg2 * p7
    ) * (1.0 - mp1 * mp1)
    dmp2 = (
        pg0 * p2
        + pg1 * p5
        + pg2 * p8
    ) * (1.0 - mp2 * mp2)

    dmc0 = (
        tl.sum(g0 * innovation, axis=0) * cw0
        + tl.sum(g1 * innovation, axis=0) * cw3
        + tl.sum(g2 * innovation, axis=0) * cw6
    ) * (1.0 - mc0 * mc0)
    dmc1 = (
        tl.sum(g0 * innovation, axis=0) * cw1
        + tl.sum(g1 * innovation, axis=0) * cw4
        + tl.sum(g2 * innovation, axis=0) * cw7
    ) * (1.0 - mc1 * mc1)
    dmc2 = (
        tl.sum(g0 * innovation, axis=0) * cw2
        + tl.sum(g1 * innovation, axis=0) * cw5
        + tl.sum(g2 * innovation, axis=0) * cw8
    ) * (1.0 - mc2 * mc2)

    pred_grad0 = tl.sum(pg0 * h0, axis=0)
    pred_grad1 = tl.sum(pg1 * h0, axis=0)
    pred_grad2 = tl.sum(pg2 * h0, axis=0)
    pred_grad3 = tl.sum(pg0 * h1, axis=0)
    pred_grad4 = tl.sum(pg1 * h1, axis=0)
    pred_grad5 = tl.sum(pg2 * h1, axis=0)
    pred_grad6 = tl.sum(pg0 * h2, axis=0)
    pred_grad7 = tl.sum(pg1 * h2, axis=0)
    pred_grad8 = tl.sum(pg2 * h2, axis=0)

    corr_grad0 = tl.sum(tl.sum(g0 * innovation, axis=0) * mc0, axis=0)
    corr_grad1 = tl.sum(tl.sum(g0 * innovation, axis=0) * mc1, axis=0)
    corr_grad2 = tl.sum(tl.sum(g0 * innovation, axis=0) * mc2, axis=0)
    corr_grad3 = tl.sum(tl.sum(g1 * innovation, axis=0) * mc0, axis=0)
    corr_grad4 = tl.sum(tl.sum(g1 * innovation, axis=0) * mc1, axis=0)
    corr_grad5 = tl.sum(tl.sum(g1 * innovation, axis=0) * mc2, axis=0)
    corr_grad6 = tl.sum(tl.sum(g2 * innovation, axis=0) * mc0, axis=0)
    corr_grad7 = tl.sum(tl.sum(g2 * innovation, axis=0) * mc1, axis=0)
    corr_grad8 = tl.sum(tl.sum(g2 * innovation, axis=0) * mc2, axis=0)

    tl.atomic_add(grad_prediction_coef_weight + 0, pred_grad0)
    tl.atomic_add(grad_prediction_coef_weight + 1, pred_grad1)
    tl.atomic_add(grad_prediction_coef_weight + 2, pred_grad2)
    tl.atomic_add(grad_prediction_coef_weight + 3, pred_grad3)
    tl.atomic_add(grad_prediction_coef_weight + 4, pred_grad4)
    tl.atomic_add(grad_prediction_coef_weight + 5, pred_grad5)
    tl.atomic_add(grad_prediction_coef_weight + 6, pred_grad6)
    tl.atomic_add(grad_prediction_coef_weight + 7, pred_grad7)
    tl.atomic_add(grad_prediction_coef_weight + 8, pred_grad8)

    tl.atomic_add(grad_correction_coef_weight + 0, corr_grad0)
    tl.atomic_add(grad_correction_coef_weight + 1, corr_grad1)
    tl.atomic_add(grad_correction_coef_weight + 2, corr_grad2)
    tl.atomic_add(grad_correction_coef_weight + 3, corr_grad3)
    tl.atomic_add(grad_correction_coef_weight + 4, corr_grad4)
    tl.atomic_add(grad_correction_coef_weight + 5, corr_grad5)
    tl.atomic_add(grad_correction_coef_weight + 6, corr_grad6)
    tl.atomic_add(grad_correction_coef_weight + 7, corr_grad7)
    tl.atomic_add(grad_correction_coef_weight + 8, corr_grad8)

    normalized_grad_predict = (
        dmp0 * router0 + dmp1 * router1 + dmp2 * router2
    ) * router_scale * norm
    normalized_grad_correct = (
        dmc0 * router0 + dmc1 * router1 + dmc2 * router2
    ) * router_scale * norm

    mean_predict = tl.sum(normalized_grad_predict * active_hidden, axis=0) / hidden_size
    mean_correct = tl.sum(normalized_grad_correct * av, axis=0) / hidden_size

    grad_active_router = (
        normalized_grad_predict * rstd_predict
        - active_hidden * rstd_predict * rstd_predict * rstd_predict * mean_predict
    )
    grad_activated_router = (
        normalized_grad_correct * rstd_correct
        - av * rstd_correct * rstd_correct * rstd_correct * mean_correct
    )

    hg0 = pg0 + pg0 * p0 + pg1 * p3 + pg2 * p6
    hg1 = pg1 + pg0 * p1 + pg1 * p4 + pg2 * p7
    hg2 = pg2 + pg0 * p2 + pg1 * p5 + pg2 * p8

    if active_idx == 0:
        hg0 += grad_active_router
    elif active_idx == 1:
        hg1 += grad_active_router
    else:
        hg2 += grad_active_router

    grad_av = grad_innovation + grad_activated_router

    tl.store(
        grad_hidden_states + token * hidden_size + k,
        hg0.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        grad_hidden_states + (num_tokens + token) * hidden_size + k,
        hg1.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        grad_hidden_states + (2 * num_tokens + token) * hidden_size + k,
        hg2.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        grad_activated + token * hidden_size + k,
        grad_av.to(tl.bfloat16),
        mask=mask,
    )

    router_grad0 = dmp0 * scaled_predict + dmc0 * scaled_correct
    router_grad1 = dmp1 * scaled_predict + dmc1 * scaled_correct
    router_grad2 = dmp2 * scaled_predict + dmc2 * scaled_correct

    norm_grad = (
        normalized_grad_predict * normalized_predict
        + normalized_grad_correct * normalized_correct
    )

    tl.atomic_add(grad_router_weight + k, router_grad0, mask=mask)
    tl.atomic_add(
        grad_router_weight + hidden_size + k,
        router_grad1,
        mask=mask,
    )
    tl.atomic_add(
        grad_router_weight + 2 * hidden_size + k,
        router_grad2,
        mask=mask,
    )
    tl.atomic_add(grad_norm_weight + k, norm_grad, mask=mask)


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
    num_tokens = hidden_states.shape[1] * hidden_states.shape[2]
    device = hidden_states.device

    grad_hidden_states = torch.empty_like(hidden_states)
    grad_activated = torch.empty_like(activated)

    grad_prediction_coef_weight = torch.zeros(
        prediction_coef_weight.shape,
        device=device,
        dtype=torch.float32,
    )
    grad_correction_coef_weight = torch.zeros(
        correction_coef_weight.shape,
        device=device,
        dtype=torch.float32,
    )
    grad_router_weight = torch.zeros(
        router_weight.shape,
        device=device,
        dtype=torch.float32,
    )
    grad_norm_weight = torch.zeros(
        norm_weight.shape,
        device=device,
        dtype=torch.float32,
    )

    if num_tokens >= 32768:
        num_warps = 8
    elif num_tokens >= 4096:
        num_warps = 4
    else:
        num_warps = 4

    _backward_kernel[(num_tokens,)](
        grad_corrected,
        hidden_states,
        activated,
        prediction_coef_weight,
        correction_coef_weight,
        router_weight,
        norm_weight,
        grad_hidden_states,
        grad_activated,
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        grad_router_weight,
        grad_norm_weight,
        num_tokens,
        altup_active_idx,
        rms_norm_eps,
        hidden_size=_HIDDEN_SIZE,
        block_k=4096,
        num_warps=num_warps,
    )

    return (
        grad_hidden_states,
        grad_activated,
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        grad_router_weight,
        grad_norm_weight,
    )