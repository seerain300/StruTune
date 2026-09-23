# solution=GPT-5.6-Sol_040_altup_predict_correction_cycle_backward_triton_optimized_r3 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 2304


@triton.jit
def _router_forward_kernel(
    hidden_states,
    activated,
    router_weight,
    norm_weight,
    router_cache,
    num_tokens: tl.constexpr,
    active_idx: tl.constexpr,
    rms_norm_eps,
    hidden_size: tl.constexpr,
    block_k: tl.constexpr,
):
    token = tl.program_id(0)
    k = tl.arange(0, block_k)
    mask = k < hidden_size

    x_predict = tl.load(
        hidden_states + (active_idx * num_tokens + token) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x_correct = tl.load(
        activated + token * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    norm = tl.load(norm_weight + k, mask=mask, other=0.0).to(tl.float32)

    rstd_predict = tl.rsqrt(
        tl.sum(x_predict * x_predict, axis=0) / hidden_size + rms_norm_eps
    )
    rstd_correct = tl.rsqrt(
        tl.sum(x_correct * x_correct, axis=0) / hidden_size + rms_norm_eps
    )

    router_scale = 1.0 / 2304.0
    scaled_predict = x_predict * rstd_predict * norm * router_scale
    scaled_correct = x_correct * rstd_correct * norm * router_scale

    tl.store(router_cache + token * 8, rstd_predict)
    tl.store(router_cache + token * 8 + 1, rstd_correct)

    for q in tl.static_range(0, 3):
        router = tl.load(
            router_weight + q * hidden_size + k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        routed_predict = tl.sum(scaled_predict * router, axis=0)
        routed_correct = tl.sum(scaled_correct * router, axis=0)
        modality_predict = 2.0 * tl.sigmoid(2.0 * routed_predict) - 1.0
        modality_correct = 2.0 * tl.sigmoid(2.0 * routed_correct) - 1.0
        tl.store(router_cache + token * 8 + 2 + q, modality_predict)
        tl.store(router_cache + token * 8 + 5 + q, modality_correct)


@triton.jit
def _token_reduction_kernel(
    grad_corrected,
    hidden_states,
    activated,
    prediction_coef_weight,
    correction_coef_weight,
    router_weight,
    norm_weight,
    router_cache,
    token_cache,
    num_tokens: tl.constexpr,
    active_idx: tl.constexpr,
    hidden_size: tl.constexpr,
    block_k: tl.constexpr,
):
    token = tl.program_id(0)
    k = tl.arange(0, block_k)
    mask = k < hidden_size

    mp0 = tl.load(router_cache + token * 8 + 2)
    mp1 = tl.load(router_cache + token * 8 + 3)
    mp2 = tl.load(router_cache + token * 8 + 4)
    mc0 = tl.load(router_cache + token * 8 + 5)
    mc1 = tl.load(router_cache + token * 8 + 6)
    mc2 = tl.load(router_cache + token * 8 + 7)

    active_row = active_idx * 9
    active_coef0 = (
        mp0 * tl.load(prediction_coef_weight + active_row).to(tl.float32)
        + mp1 * tl.load(prediction_coef_weight + active_row + 1).to(tl.float32)
        + mp2 * tl.load(prediction_coef_weight + active_row + 2).to(tl.float32)
    )
    active_coef1 = (
        mp0 * tl.load(prediction_coef_weight + active_row + 3).to(tl.float32)
        + mp1 * tl.load(prediction_coef_weight + active_row + 4).to(tl.float32)
        + mp2 * tl.load(prediction_coef_weight + active_row + 5).to(tl.float32)
    )
    active_coef2 = (
        mp0 * tl.load(prediction_coef_weight + active_row + 6).to(tl.float32)
        + mp1 * tl.load(prediction_coef_weight + active_row + 7).to(tl.float32)
        + mp2 * tl.load(prediction_coef_weight + active_row + 8).to(tl.float32)
    )

    c0 = 1.0 + (
        mc0 * tl.load(correction_coef_weight).to(tl.float32)
        + mc1 * tl.load(correction_coef_weight + 1).to(tl.float32)
        + mc2 * tl.load(correction_coef_weight + 2).to(tl.float32)
    )
    c1 = 1.0 + (
        mc0 * tl.load(correction_coef_weight + 3).to(tl.float32)
        + mc1 * tl.load(correction_coef_weight + 4).to(tl.float32)
        + mc2 * tl.load(correction_coef_weight + 5).to(tl.float32)
    )
    c2 = 1.0 + (
        mc0 * tl.load(correction_coef_weight + 6).to(tl.float32)
        + mc1 * tl.load(correction_coef_weight + 7).to(tl.float32)
        + mc2 * tl.load(correction_coef_weight + 8).to(tl.float32)
    )

    h0 = tl.load(
        hidden_states + token * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    h1 = tl.load(
        hidden_states + (num_tokens + token) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    h2 = tl.load(
        hidden_states + (2 * num_tokens + token) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    if active_idx == 0:
        active_hidden = h0
    elif active_idx == 1:
        active_hidden = h1
    else:
        active_hidden = h2

    prediction_active = (
        h0 * active_coef0
        + h1 * active_coef1
        + h2 * active_coef2
        + active_hidden
    )

    activated_value = tl.load(
        activated + token * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    innovation = activated_value - prediction_active

    g0 = tl.load(
        grad_corrected + token * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    g1 = tl.load(
        grad_corrected + (num_tokens + token) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    g2 = tl.load(
        grad_corrected + (2 * num_tokens + token) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    cg0 = tl.sum(g0 * innovation, axis=0)
    cg1 = tl.sum(g1 * innovation, axis=0)
    cg2 = tl.sum(g2 * innovation, axis=0)

    tl.store(token_cache + token * 20 + 9, cg0)
    tl.store(token_cache + token * 20 + 10, cg1)
    tl.store(token_cache + token * 20 + 11, cg2)

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

    pg00 = tl.sum(h0 * pg0, axis=0)
    pg01 = tl.sum(h1 * pg0, axis=0)
    pg02 = tl.sum(h2 * pg0, axis=0)
    pg10 = tl.sum(h0 * pg1, axis=0)
    pg11 = tl.sum(h1 * pg1, axis=0)
    pg12 = tl.sum(h2 * pg1, axis=0)
    pg20 = tl.sum(h0 * pg2, axis=0)
    pg21 = tl.sum(h1 * pg2, axis=0)
    pg22 = tl.sum(h2 * pg2, axis=0)

    tl.store(token_cache + token * 20, pg00)
    tl.store(token_cache + token * 20 + 1, pg01)
    tl.store(token_cache + token * 20 + 2, pg02)
    tl.store(token_cache + token * 20 + 3, pg10)
    tl.store(token_cache + token * 20 + 4, pg11)
    tl.store(token_cache + token * 20 + 5, pg12)
    tl.store(token_cache + token * 20 + 6, pg20)
    tl.store(token_cache + token * 20 + 7, pg21)
    tl.store(token_cache + token * 20 + 8, pg22)

    gp0 = (
        pg00 * tl.load(prediction_coef_weight).to(tl.float32)
        + pg01 * tl.load(prediction_coef_weight + 3).to(tl.float32)
        + pg02 * tl.load(prediction_coef_weight + 6).to(tl.float32)
        + pg10 * tl.load(prediction_coef_weight + 9).to(tl.float32)
        + pg11 * tl.load(prediction_coef_weight + 12).to(tl.float32)
        + pg12 * tl.load(prediction_coef_weight + 15).to(tl.float32)
        + pg20 * tl.load(prediction_coef_weight + 18).to(tl.float32)
        + pg21 * tl.load(prediction_coef_weight + 21).to(tl.float32)
        + pg22 * tl.load(prediction_coef_weight + 24).to(tl.float32)
    ) * (1.0 - mp0 * mp0)

    gp1 = (
        pg00 * tl.load(prediction_coef_weight + 1).to(tl.float32)
        + pg01 * tl.load(prediction_coef_weight + 4).to(tl.float32)
        + pg02 * tl.load(prediction_coef_weight + 7).to(tl.float32)
        + pg10 * tl.load(prediction_coef_weight + 10).to(tl.float32)
        + pg11 * tl.load(prediction_coef_weight + 13).to(tl.float32)
        + pg12 * tl.load(prediction_coef_weight + 16).to(tl.float32)
        + pg20 * tl.load(prediction_coef_weight + 19).to(tl.float32)
        + pg21 * tl.load(prediction_coef_weight + 22).to(tl.float32)
        + pg22 * tl.load(prediction_coef_weight + 25).to(tl.float32)
    ) * (1.0 - mp1 * mp1)

    gp2 = (
        pg00 * tl.load(prediction_coef_weight + 2).to(tl.float32)
        + pg01 * tl.load(prediction_coef_weight + 5).to(tl.float32)
        + pg02 * tl.load(prediction_coef_weight + 8).to(tl.float32)
        + pg10 * tl.load(prediction_coef_weight + 11).to(tl.float32)
        + pg11 * tl.load(prediction_coef_weight + 14).to(tl.float32)
        + pg12 * tl.load(prediction_coef_weight + 17).to(tl.float32)
        + pg20 * tl.load(prediction_coef_weight + 20).to(tl.float32)
        + pg21 * tl.load(prediction_coef_weight + 23).to(tl.float32)
        + pg22 * tl.load(prediction_coef_weight + 26).to(tl.float32)
    ) * (1.0 - mp2 * mp2)

    gc0 = (
        cg0 * tl.load(correction_coef_weight).to(tl.float32)
        + cg1 * tl.load(correction_coef_weight + 3).to(tl.float32)
        + cg2 * tl.load(correction_coef_weight + 6).to(tl.float32)
    ) * (1.0 - mc0 * mc0)

    gc1 = (
        cg0 * tl.load(correction_coef_weight + 1).to(tl.float32)
        + cg1 * tl.load(correction_coef_weight + 4).to(tl.float32)
        + cg2 * tl.load(correction_coef_weight + 7).to(tl.float32)
    ) * (1.0 - mc1 * mc1)

    gc2 = (
        cg0 * tl.load(correction_coef_weight + 2).to(tl.float32)
        + cg1 * tl.load(correction_coef_weight + 5).to(tl.float32)
        + cg2 * tl.load(correction_coef_weight + 8).to(tl.float32)
    ) * (1.0 - mc2 * mc2)

    tl.store(token_cache + token * 20 + 12, gp0)
    tl.store(token_cache + token * 20 + 13, gp1)
    tl.store(token_cache + token * 20 + 14, gp2)
    tl.store(token_cache + token * 20 + 15, gc0)
    tl.store(token_cache + token * 20 + 16, gc1)
    tl.store(token_cache + token * 20 + 17, gc2)

    norm = tl.load(norm_weight + k, mask=mask, other=0.0).to(tl.float32)
    router0 = tl.load(router_weight + k, mask=mask, other=0.0).to(tl.float32)
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
    grad_normalized_predict = (
        gp0 * router0 + gp1 * router1 + gp2 * router2
    ) * router_scale * norm
    grad_normalized_correct = (
        gc0 * router0 + gc1 * router1 + gc2 * router2
    ) * router_scale * norm

    mean_predict = (
        tl.sum(grad_normalized_predict * active_hidden, axis=0) / hidden_size
    )
    mean_correct = (
        tl.sum(grad_normalized_correct * activated_value, axis=0) / hidden_size
    )

    tl.store(token_cache + token * 20 + 18, mean_predict)
    tl.store(token_cache + token * 20 + 19, mean_correct)


@triton.jit
def _input_and_dense_param_kernel(
    grad_corrected,
    hidden_states,
    activated,
    prediction_coef_weight,
    correction_coef_weight,
    router_weight,
    norm_weight,
    router_cache,
    token_cache,
    grad_hidden_states,
    grad_activated,
    grad_router_weight,
    grad_norm_weight,
    num_tokens: tl.constexpr,
    active_idx: tl.constexpr,
    hidden_size: tl.constexpr,
    block_t: tl.constexpr,
    block_k: tl.constexpr,
):
    token_offsets = tl.program_id(0) * block_t + tl.arange(0, block_t)
    k_offsets = tl.program_id(1) * block_k + tl.arange(0, block_k)

    token_mask = token_offsets < num_tokens
    k_mask = k_offsets < hidden_size
    mask = token_mask[:, None] & k_mask[None, :]

    t = token_offsets[:, None]
    k = k_offsets[None, :]

    rstd_predict = tl.load(
        router_cache + token_offsets * 8,
        mask=token_mask,
        other=0.0,
    )
    rstd_correct = tl.load(
        router_cache + token_offsets * 8 + 1,
        mask=token_mask,
        other=0.0,
    )
    mean_predict = tl.load(
        token_cache + token_offsets * 20 + 18,
        mask=token_mask,
        other=0.0,
    )
    mean_correct = tl.load(
        token_cache + token_offsets * 20 + 19,
        mask=token_mask,
        other=0.0,
    )

    mp0 = tl.load(
        router_cache + token_offsets * 8 + 2,
        mask=token_mask,
        other=0.0,
    )
    mp1 = tl.load(
        router_cache + token_offsets * 8 + 3,
        mask=token_mask,
        other=0.0,
    )
    mp2 = tl.load(
        router_cache + token_offsets * 8 + 4,
        mask=token_mask,
        other=0.0,
    )
    mc0 = tl.load(
        router_cache + token_offsets * 8 + 5,
        mask=token_mask,
        other=0.0,
    )
    mc1 = tl.load(
        router_cache + token_offsets * 8 + 6,
        mask=token_mask,
        other=0.0,
    )
    mc2 = tl.load(
        router_cache + token_offsets * 8 + 7,
        mask=token_mask,
        other=0.0,
    )

    pw = tl.load(
        prediction_coef_weight + tl.arange(0, 27),
    ).to(tl.float32)
    cw = tl.load(
        correction_coef_weight + tl.arange(0, 9),
    ).to(tl.float32)

    p0 = mp0 * pw[0] + mp1 * pw[1] + mp2 * pw[2]
    p1 = mp0 * pw[3] + mp1 * pw[4] + mp2 * pw[5]
    p2 = mp0 * pw[6] + mp1 * pw[7] + mp2 * pw[8]
    p3 = mp0 * pw[9] + mp1 * pw[10] + mp2 * pw[11]
    p4 = mp0 * pw[12] + mp1 * pw[13] + mp2 * pw[14]
    p5 = mp0 * pw[15] + mp1 * pw[16] + mp2 * pw[17]
    p6 = mp0 * pw[18] + mp1 * pw[19] + mp2 * pw[20]
    p7 = mp0 * pw[21] + mp1 * pw[22] + mp2 * pw[23]
    p8 = mp0 * pw[24] + mp1 * pw[25] + mp2 * pw[26]

    c0 = 1.0 + mc0 * cw[0] + mc1 * cw[1] + mc2 * cw[2]
    c1 = 1.0 + mc0 * cw[3] + mc1 * cw[4] + mc2 * cw[5]
    c2 = 1.0 + mc0 * cw[6] + mc1 * cw[7] + mc2 * cw[8]

    h0 = tl.load(
        hidden_states + t * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    h1 = tl.load(
        hidden_states + (num_tokens + t) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    h2 = tl.load(
        hidden_states + (2 * num_tokens + t) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    if active_idx == 0:
        active_hidden = h0
    elif active_idx == 1:
        active_hidden = h1
    else:
        active_hidden = h2

    activated_value = tl.load(
        activated + t * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    g0 = tl.load(
        grad_corrected + t * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    g1 = tl.load(
        grad_corrected + (num_tokens + t) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    g2 = tl.load(
        grad_corrected + (2 * num_tokens + t) * hidden_size + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    grad_innovation = (
        g0 * c0[:, None] + g1 * c1[:, None] + g2 * c2[:, None]
    )

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

    norm = tl.load(
        norm_weight + k,
        mask=k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    router0 = tl.load(
        router_weight + k,
        mask=k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    router1 = tl.load(
        router_weight + hidden_size + k,
        mask=k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    router2 = tl.load(
        router_weight + 2 * hidden_size + k,
        mask=k_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    router_scale = 1.0 / 2304.0
    grad_normed_predict = (
        gp0[:, None] * router0
        + gp1[:, None] * router1
        + gp2[:, None] * router2
    ) * router_scale
    grad_normed_correct = (
        gc0[:, None] * router0
        + gc1[:, None] * router1
        + gc2[:, None] * router2
    ) * router_scale

    grad_normalized_predict = grad_normed_predict * norm
    grad_normalized_correct = grad_normed_correct * norm

    rstd_predict_cubed = (
        rstd_predict[:, None]
        * rstd_predict[:, None]
        * rstd_predict[:, None]
    )
    rstd_correct_cubed = (
        rstd_correct[:, None]
        * rstd_correct[:, None]
        * rstd_correct[:, None]
    )

    grad_active_router = (
        grad_normalized_predict * rstd_predict[:, None]
        - active_hidden * rstd_predict_cubed * mean_predict[:, None]
    )
    grad_activated_router = (
        grad_normalized_correct * rstd_correct[:, None]
        - activated_value * rstd_correct_cubed * mean_correct[:, None]
    )

    hidden_grad_0 = (
        pg0
        + pg0 * p0[:, None]
        + pg1 * p3[:, None]
        + pg2 * p6[:, None]
    )
    hidden_grad_1 = (
        pg1
        + pg0 * p1[:, None]
        + pg1 * p4[:, None]
        + pg2 * p7[:, None]
    )
    hidden_grad_2 = (
        pg2
        + pg0 * p2[:, None]
        + pg1 * p5[:, None]
        + pg2 * p8[:, None]
    )

    if active_idx == 0:
        hidden_grad_0 += grad_active_router
    elif active_idx == 1:
        hidden_grad_1 += grad_active_router
    else:
        hidden_grad_2 += grad_active_router

    tl.store(
        grad_hidden_states + t * hidden_size + k,
        hidden_grad_0,
        mask=mask,
    )
    tl.store(
        grad_hidden_states + (num_tokens + t) * hidden_size + k,
        hidden_grad_1,
        mask=mask,
    )
    tl.store(
        grad_hidden_states + (2 * num_tokens + t) * hidden_size + k,
        hidden_grad_2,
        mask=mask,
    )
    tl.store(
        grad_activated + t * hidden_size + k,
        grad_innovation + grad_activated_router,
        mask=mask,
    )

    normalized_predict = active_hidden * rstd_predict[:, None]
    normalized_correct = activated_value * rstd_correct[:, None]
    scaled_predict = normalized_predict * norm * router_scale
    scaled_correct = normalized_correct * norm * router_scale

    router_sum0 = tl.sum(
        gp0[:, None] * scaled_predict + gc0[:, None] * scaled_correct,
        axis=0,
    )
    router_sum1 = tl.sum(
        gp1[:, None] * scaled_predict + gc1[:, None] * scaled_correct,
        axis=0,
    )
    router_sum2 = tl.sum(
        gp2[:, None] * scaled_predict + gc2[:, None] * scaled_correct,
        axis=0,
    )
    norm_sum = tl.sum(
        grad_normed_predict * normalized_predict
        + grad_normed_correct * normalized_correct,
        axis=0,
    )

    tl.atomic_add(
        grad_router_weight + k_offsets,
        router_sum0,
        mask=k_mask,
    )
    tl.atomic_add(
        grad_router_weight + hidden_size + k_offsets,
        router_sum1,
        mask=k_mask,
    )
    tl.atomic_add(
        grad_router_weight + 2 * hidden_size + k_offsets,
        router_sum2,
        mask=k_mask,
    )
    tl.atomic_add(
        grad_norm_weight + k_offsets,
        norm_sum,
        mask=k_mask,
    )


@triton.jit
def _coefficient_gradient_kernel(
    router_cache,
    token_cache,
    grad_prediction_coef_weight,
    grad_correction_coef_weight,
    num_tokens: tl.constexpr,
    block_t: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)

    token_offsets = chunk * block_t + tl.arange(0, block_t)
    token_mask = token_offsets < num_tokens

    is_prediction = row < 9
    modality_offset = tl.where(is_prediction, 2, 5)

    coefficient_grad = tl.load(
        token_cache + token_offsets * 20 + row,
        mask=token_mask,
        other=0.0,
    )
    modality0 = tl.load(
        router_cache + token_offsets * 8 + modality_offset,
        mask=token_mask,
        other=0.0,
    )
    modality1 = tl.load(
        router_cache + token_offsets * 8 + modality_offset + 1,
        mask=token_mask,
        other=0.0,
    )
    modality2 = tl.load(
        router_cache + token_offsets * 8 + modality_offset + 2,
        mask=token_mask,
        other=0.0,
    )

    value0 = tl.sum(coefficient_grad * modality0, axis=0)
    value1 = tl.sum(coefficient_grad * modality1, axis=0)
    value2 = tl.sum(coefficient_grad * modality2, axis=0)

    output_row = tl.where(is_prediction, row, row - 9)
    output_base = tl.where(
        is_prediction,
        grad_prediction_coef_weight + output_row * 3,
        grad_correction_coef_weight + output_row * 3,
    )

    tl.atomic_add(output_base, value0)
    tl.atomic_add(output_base + 1, value1)
    tl.atomic_add(output_base + 2, value2)


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

    router_cache = torch.empty(
        (num_tokens, 8),
        device=device,
        dtype=torch.float32,
    )
    token_cache = torch.empty(
        (num_tokens, 20),
        device=device,
        dtype=torch.float32,
    )

    _router_forward_kernel[(num_tokens,)](
        hidden_states,
        activated,
        router_weight,
        norm_weight,
        router_cache,
        num_tokens,
        altup_active_idx,
        rms_norm_eps,
        hidden_size=_HIDDEN_SIZE,
        block_k=4096,
        num_warps=8,
    )

    _token_reduction_kernel[(num_tokens,)](
        grad_corrected,
        hidden_states,
        activated,
        prediction_coef_weight,
        correction_coef_weight,
        router_weight,
        norm_weight,
        router_cache,
        token_cache,
        num_tokens,
        altup_active_idx,
        hidden_size=_HIDDEN_SIZE,
        block_k=4096,
        num_warps=8,
    )

    block_t = 32 if num_tokens >= 16384 else (16 if num_tokens >= 4096 else 8)
    block_k = 64

    _input_and_dense_param_kernel[
        (
            triton.cdiv(num_tokens, block_t),
            triton.cdiv(_HIDDEN_SIZE, block_k),
        )
    ](
        grad_corrected,
        hidden_states,
        activated,
        prediction_coef_weight,
        correction_coef_weight,
        router_weight,
        norm_weight,
        router_cache,
        token_cache,
        grad_hidden_states,
        grad_activated,
        grad_router_weight,
        grad_norm_weight,
        num_tokens,
        altup_active_idx,
        hidden_size=_HIDDEN_SIZE,
        block_t=block_t,
        block_k=block_k,
        num_warps=4,
    )

    coefficient_block_t = 1024
    coefficient_chunks = triton.cdiv(num_tokens, coefficient_block_t)

    _coefficient_gradient_kernel[(12, coefficient_chunks)](
        router_cache,
        token_cache,
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        num_tokens,
        block_t=coefficient_block_t,
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