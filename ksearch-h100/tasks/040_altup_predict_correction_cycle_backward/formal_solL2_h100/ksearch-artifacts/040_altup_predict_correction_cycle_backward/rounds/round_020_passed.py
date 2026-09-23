# solution=GPT-5.6-Sol_040_altup_predict_correction_cycle_backward_triton_optimized_r20 score=8.296711940051255 passed=True
import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 2304
_ROUTER_SCALE = tl.constexpr(1.0 / 2304.0)


@triton.jit
def _forward_state_kernel(
    hidden_states, activated, prediction_coef_weight, correction_coef_weight,
    router_weight, norm_weight, state, num_tokens, rms_norm_eps,
    ACTIVE: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    sum_sq_predict = 0.0
    sum_sq_correct = 0.0
    predict_dot_0 = 0.0
    predict_dot_1 = 0.0
    predict_dot_2 = 0.0
    correct_dot_0 = 0.0
    correct_dot_1 = 0.0
    correct_dot_2 = 0.0

    for start in range(0, K, BLOCK_K):
        k = start + offsets
        mask = k < K
        predict_x = tl.load(
            hidden_states + (ACTIVE * num_tokens + token) * K + k,
            mask=mask, other=0.0,
        ).to(tl.float32)
        correct_x = tl.load(
            activated + token * K + k,
            mask=mask, other=0.0,
        ).to(tl.float32)
        norm = tl.load(norm_weight + k, mask=mask, other=0.0).to(tl.float32)
        router_0 = tl.load(
            router_weight + k, mask=mask, other=0.0,
        ).to(tl.float32)
        router_1 = tl.load(
            router_weight + K + k, mask=mask, other=0.0,
        ).to(tl.float32)
        router_2 = tl.load(
            router_weight + 2 * K + k, mask=mask, other=0.0,
        ).to(tl.float32)

        weighted_predict = predict_x * norm
        weighted_correct = correct_x * norm
        sum_sq_predict += tl.sum(predict_x * predict_x, axis=0)
        sum_sq_correct += tl.sum(correct_x * correct_x, axis=0)
        predict_dot_0 += tl.sum(weighted_predict * router_0, axis=0)
        predict_dot_1 += tl.sum(weighted_predict * router_1, axis=0)
        predict_dot_2 += tl.sum(weighted_predict * router_2, axis=0)
        correct_dot_0 += tl.sum(weighted_correct * router_0, axis=0)
        correct_dot_1 += tl.sum(weighted_correct * router_1, axis=0)
        correct_dot_2 += tl.sum(weighted_correct * router_2, axis=0)

    rstd_predict = tl.rsqrt(sum_sq_predict / K + rms_norm_eps)
    rstd_correct = tl.rsqrt(sum_sq_correct / K + rms_norm_eps)

    routed_predict_0 = predict_dot_0 * rstd_predict * _ROUTER_SCALE
    routed_predict_1 = predict_dot_1 * rstd_predict * _ROUTER_SCALE
    routed_predict_2 = predict_dot_2 * rstd_predict * _ROUTER_SCALE
    routed_correct_0 = correct_dot_0 * rstd_correct * _ROUTER_SCALE
    routed_correct_1 = correct_dot_1 * rstd_correct * _ROUTER_SCALE
    routed_correct_2 = correct_dot_2 * rstd_correct * _ROUTER_SCALE

    modality_predict_0 = 2.0 * tl.sigmoid(2.0 * routed_predict_0) - 1.0
    modality_predict_1 = 2.0 * tl.sigmoid(2.0 * routed_predict_1) - 1.0
    modality_predict_2 = 2.0 * tl.sigmoid(2.0 * routed_predict_2) - 1.0
    modality_correct_0 = 2.0 * tl.sigmoid(2.0 * routed_correct_0) - 1.0
    modality_correct_1 = 2.0 * tl.sigmoid(2.0 * routed_correct_1) - 1.0
    modality_correct_2 = 2.0 * tl.sigmoid(2.0 * routed_correct_2) - 1.0

    base = state + token * 26
    tl.store(base + 0, rstd_predict)
    tl.store(base + 1, rstd_correct)
    tl.store(base + 2, modality_predict_0)
    tl.store(base + 3, modality_predict_1)
    tl.store(base + 4, modality_predict_2)
    tl.store(base + 5, modality_correct_0)
    tl.store(base + 6, modality_correct_1)
    tl.store(base + 7, modality_correct_2)

    for output in range(9):
        weight_base = prediction_coef_weight + output * 3
        coefficient = (
            tl.load(weight_base + 0).to(tl.float32) * modality_predict_0
            + tl.load(weight_base + 1).to(tl.float32) * modality_predict_1
            + tl.load(weight_base + 2).to(tl.float32) * modality_predict_2
        )
        tl.store(base + 8 + output, coefficient)

    for output in range(3):
        weight_base = correction_coef_weight + output * 3
        coefficient = (
            tl.load(weight_base + 0).to(tl.float32) * modality_correct_0
            + tl.load(weight_base + 1).to(tl.float32) * modality_correct_1
            + tl.load(weight_base + 2).to(tl.float32) * modality_correct_2
            + 1.0
        )
        tl.store(base + 17 + output, coefficient)

    tl.store(base + 20, routed_predict_0)
    tl.store(base + 21, routed_predict_1)
    tl.store(base + 22, routed_predict_2)
    tl.store(base + 23, routed_correct_0)
    tl.store(base + 24, routed_correct_1)
    tl.store(base + 25, routed_correct_2)


@triton.jit
def _token_gradient_state_kernel(
    grad_corrected, hidden_states, activated, prediction_coef_weight,
    correction_coef_weight, state, gradient_state, num_tokens,
    ACTIVE: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    state_base = state + token * 26

    pc00 = tl.load(state_base + 8)
    pc01 = tl.load(state_base + 9)
    pc02 = tl.load(state_base + 10)
    pc10 = tl.load(state_base + 11)
    pc11 = tl.load(state_base + 12)
    pc12 = tl.load(state_base + 13)
    pc20 = tl.load(state_base + 14)
    pc21 = tl.load(state_base + 15)
    pc22 = tl.load(state_base + 16)
    cc0 = tl.load(state_base + 17)
    cc1 = tl.load(state_base + 18)
    cc2 = tl.load(state_base + 19)

    grad_pc00 = 0.0
    grad_pc01 = 0.0
    grad_pc02 = 0.0
    grad_pc10 = 0.0
    grad_pc11 = 0.0
    grad_pc12 = 0.0
    grad_pc20 = 0.0
    grad_pc21 = 0.0
    grad_pc22 = 0.0
    grad_cc0 = 0.0
    grad_cc1 = 0.0
    grad_cc2 = 0.0
    offsets = tl.arange(0, BLOCK_K)

    for start in range(0, K, BLOCK_K):
        k = start + offsets
        mask = k < K

        h0 = tl.load(
            hidden_states + token * K + k, mask=mask, other=0.0,
        ).to(tl.float32)
        h1 = tl.load(
            hidden_states + (num_tokens + token) * K + k,
            mask=mask, other=0.0,
        ).to(tl.float32)
        h2 = tl.load(
            hidden_states + (2 * num_tokens + token) * K + k,
            mask=mask, other=0.0,
        ).to(tl.float32)
        gc0 = tl.load(
            grad_corrected + token * K + k, mask=mask, other=0.0,
        ).to(tl.float32)
        gc1 = tl.load(
            grad_corrected + (num_tokens + token) * K + k,
            mask=mask, other=0.0,
        ).to(tl.float32)
        gc2 = tl.load(
            grad_corrected + (2 * num_tokens + token) * K + k,
            mask=mask, other=0.0,
        ).to(tl.float32)
        act = tl.load(
            activated + token * K + k, mask=mask, other=0.0,
        ).to(tl.float32)

        if ACTIVE == 0:
            prediction_active = h0 + h0 * pc00 + h1 * pc01 + h2 * pc02
        elif ACTIVE == 1:
            prediction_active = h1 + h0 * pc10 + h1 * pc11 + h2 * pc12
        else:
            prediction_active = h2 + h0 * pc20 + h1 * pc21 + h2 * pc22

        innovation = act - prediction_active
        grad_innovation = gc0 * cc0 + gc1 * cc1 + gc2 * cc2

        if ACTIVE == 0:
            gp0 = gc0 - grad_innovation
            gp1 = gc1
            gp2 = gc2
        elif ACTIVE == 1:
            gp0 = gc0
            gp1 = gc1 - grad_innovation
            gp2 = gc2
        else:
            gp0 = gc0
            gp1 = gc1
            gp2 = gc2 - grad_innovation

        grad_cc0 += tl.sum(gc0 * innovation, axis=0)
        grad_cc1 += tl.sum(gc1 * innovation, axis=0)
        grad_cc2 += tl.sum(gc2 * innovation, axis=0)
        grad_pc00 += tl.sum(gp0 * h0, axis=0)
        grad_pc01 += tl.sum(gp0 * h1, axis=0)
        grad_pc02 += tl.sum(gp0 * h2, axis=0)
        grad_pc10 += tl.sum(gp1 * h0, axis=0)
        grad_pc11 += tl.sum(gp1 * h1, axis=0)
        grad_pc12 += tl.sum(gp1 * h2, axis=0)
        grad_pc20 += tl.sum(gp2 * h0, axis=0)
        grad_pc21 += tl.sum(gp2 * h1, axis=0)
        grad_pc22 += tl.sum(gp2 * h2, axis=0)

    gradient_base = gradient_state + token * 20
    tl.store(gradient_base + 0, grad_pc00)
    tl.store(gradient_base + 1, grad_pc01)
    tl.store(gradient_base + 2, grad_pc02)
    tl.store(gradient_base + 3, grad_pc10)
    tl.store(gradient_base + 4, grad_pc11)
    tl.store(gradient_base + 5, grad_pc12)
    tl.store(gradient_base + 6, grad_pc20)
    tl.store(gradient_base + 7, grad_pc21)
    tl.store(gradient_base + 8, grad_pc22)
    tl.store(gradient_base + 9, grad_cc0)
    tl.store(gradient_base + 10, grad_cc1)
    tl.store(gradient_base + 11, grad_cc2)

    grad_modality_predict_0 = 0.0
    grad_modality_predict_1 = 0.0
    grad_modality_predict_2 = 0.0
    for output in range(9):
        grad_coefficient = tl.load(gradient_base + output)
        weight_base = prediction_coef_weight + output * 3
        grad_modality_predict_0 += (
            grad_coefficient * tl.load(weight_base + 0).to(tl.float32)
        )
        grad_modality_predict_1 += (
            grad_coefficient * tl.load(weight_base + 1).to(tl.float32)
        )
        grad_modality_predict_2 += (
            grad_coefficient * tl.load(weight_base + 2).to(tl.float32)
        )

    grad_modality_correct_0 = 0.0
    grad_modality_correct_1 = 0.0
    grad_modality_correct_2 = 0.0
    for output in range(3):
        grad_coefficient = tl.load(gradient_base + 9 + output)
        weight_base = correction_coef_weight + output * 3
        grad_modality_correct_0 += (
            grad_coefficient * tl.load(weight_base + 0).to(tl.float32)
        )
        grad_modality_correct_1 += (
            grad_coefficient * tl.load(weight_base + 1).to(tl.float32)
        )
        grad_modality_correct_2 += (
            grad_coefficient * tl.load(weight_base + 2).to(tl.float32)
        )

    modality_predict_0 = tl.load(state_base + 2)
    modality_predict_1 = tl.load(state_base + 3)
    modality_predict_2 = tl.load(state_base + 4)
    modality_correct_0 = tl.load(state_base + 5)
    modality_correct_1 = tl.load(state_base + 6)
    modality_correct_2 = tl.load(state_base + 7)

    grad_routed_predict_0 = grad_modality_predict_0 * (
        1.0 - modality_predict_0 * modality_predict_0
    )
    grad_routed_predict_1 = grad_modality_predict_1 * (
        1.0 - modality_predict_1 * modality_predict_1
    )
    grad_routed_predict_2 = grad_modality_predict_2 * (
        1.0 - modality_predict_2 * modality_predict_2
    )
    grad_routed_correct_0 = grad_modality_correct_0 * (
        1.0 - modality_correct_0 * modality_correct_0
    )
    grad_routed_correct_1 = grad_modality_correct_1 * (
        1.0 - modality_correct_1 * modality_correct_1
    )
    grad_routed_correct_2 = grad_modality_correct_2 * (
        1.0 - modality_correct_2 * modality_correct_2
    )

    tl.store(gradient_base + 12, grad_routed_predict_0)
    tl.store(gradient_base + 13, grad_routed_predict_1)
    tl.store(gradient_base + 14, grad_routed_predict_2)
    tl.store(gradient_base + 15, grad_routed_correct_0)
    tl.store(gradient_base + 16, grad_routed_correct_1)
    tl.store(gradient_base + 17, grad_routed_correct_2)

    predict_routed_dot = (
        grad_routed_predict_0 * tl.load(state_base + 20)
        + grad_routed_predict_1 * tl.load(state_base + 21)
        + grad_routed_predict_2 * tl.load(state_base + 22)
    )
    correct_routed_dot = (
        grad_routed_correct_0 * tl.load(state_base + 23)
        + grad_routed_correct_1 * tl.load(state_base + 24)
        + grad_routed_correct_2 * tl.load(state_base + 25)
    )
    tl.store(gradient_base + 18, predict_routed_dot)
    tl.store(gradient_base + 19, correct_routed_dot)


@triton.jit
def _coefficient_gradient_kernel(
    state, gradient_state, grad_prediction_coef_weight,
    grad_correction_coef_weight, num_tokens, BLOCK_T: tl.constexpr,
):
    output = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_T)
    accumulator = 0.0

    if output < 27:
        coefficient = output // 3
        modality = output - coefficient * 3
        for start in tl.range(0, num_tokens, BLOCK_T):
            tokens = start + offsets
            mask = tokens < num_tokens
            grad_coefficient = tl.load(
                gradient_state + tokens * 20 + coefficient,
                mask=mask, other=0.0,
            )
            modality_value = tl.load(
                state + tokens * 26 + 2 + modality,
                mask=mask, other=0.0,
            )
            accumulator += tl.sum(grad_coefficient * modality_value, axis=0)
        tl.store(
            grad_prediction_coef_weight + coefficient * 3 + modality,
            accumulator,
        )
    else:
        local_output = output - 27
        coefficient = local_output // 3
        modality = local_output - coefficient * 3
        for start in tl.range(0, num_tokens, BLOCK_T):
            tokens = start + offsets
            mask = tokens < num_tokens
            grad_coefficient = tl.load(
                gradient_state + tokens * 20 + 9 + coefficient,
                mask=mask, other=0.0,
            )
            modality_value = tl.load(
                state + tokens * 26 + 5 + modality,
                mask=mask, other=0.0,
            )
            accumulator += tl.sum(grad_coefficient * modality_value, axis=0)
        tl.store(
            grad_correction_coef_weight + coefficient * 3 + modality,
            accumulator,
        )


@triton.jit
def _router_norm_gradient_kernel(
    hidden_states, activated, router_weight, norm_weight, state,
    gradient_state, grad_router_weight, grad_norm_weight, num_tokens,
    ACTIVE: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    k_block = tl.program_id(0)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k < K

    norm = tl.load(norm_weight + k, mask=k_mask, other=0.0).to(tl.float32)
    router_0 = tl.load(
        router_weight + k, mask=k_mask, other=0.0,
    ).to(tl.float32)
    router_1 = tl.load(
        router_weight + K + k, mask=k_mask, other=0.0,
    ).to(tl.float32)
    router_2 = tl.load(
        router_weight + 2 * K + k, mask=k_mask, other=0.0,
    ).to(tl.float32)

    grad_router_0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    grad_router_1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    grad_router_2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    grad_norm = tl.zeros((BLOCK_K,), dtype=tl.float32)
    token_offsets = tl.arange(0, BLOCK_T)

    for start in tl.range(0, num_tokens, BLOCK_T):
        tokens = start + token_offsets
        token_mask = tokens < num_tokens
        matrix_mask = token_mask[:, None] & k_mask[None, :]

        predict_x = tl.load(
            hidden_states
            + (ACTIVE * num_tokens + tokens[:, None]) * K
            + k[None, :],
            mask=matrix_mask, other=0.0,
        ).to(tl.float32)
        correct_x = tl.load(
            activated + tokens[:, None] * K + k[None, :],
            mask=matrix_mask, other=0.0,
        ).to(tl.float32)

        rstd_predict = tl.load(
            state + tokens * 26, mask=token_mask, other=0.0,
        )
        rstd_correct = tl.load(
            state + tokens * 26 + 1, mask=token_mask, other=0.0,
        )
        gp0 = tl.load(
            gradient_state + tokens * 20 + 12,
            mask=token_mask, other=0.0,
        )
        gp1 = tl.load(
            gradient_state + tokens * 20 + 13,
            mask=token_mask, other=0.0,
        )
        gp2 = tl.load(
            gradient_state + tokens * 20 + 14,
            mask=token_mask, other=0.0,
        )
        gc0 = tl.load(
            gradient_state + tokens * 20 + 15,
            mask=token_mask, other=0.0,
        )
        gc1 = tl.load(
            gradient_state + tokens * 20 + 16,
            mask=token_mask, other=0.0,
        )
        gc2 = tl.load(
            gradient_state + tokens * 20 + 17,
            mask=token_mask, other=0.0,
        )

        normalized_predict = predict_x * rstd_predict[:, None]
        normalized_correct = correct_x * rstd_correct[:, None]
        scaled_predict = normalized_predict * norm[None, :] * _ROUTER_SCALE
        scaled_correct = normalized_correct * norm[None, :] * _ROUTER_SCALE

        grad_router_0 += tl.sum(
            gp0[:, None] * scaled_predict
            + gc0[:, None] * scaled_correct,
            axis=0,
        )
        grad_router_1 += tl.sum(
            gp1[:, None] * scaled_predict
            + gc1[:, None] * scaled_correct,
            axis=0,
        )
        grad_router_2 += tl.sum(
            gp2[:, None] * scaled_predict
            + gc2[:, None] * scaled_correct,
            axis=0,
        )

        predict_router_grad = (
            gp0[:, None] * router_0[None, :]
            + gp1[:, None] * router_1[None, :]
            + gp2[:, None] * router_2[None, :]
        )
        correct_router_grad = (
            gc0[:, None] * router_0[None, :]
            + gc1[:, None] * router_1[None, :]
            + gc2[:, None] * router_2[None, :]
        )
        grad_norm += tl.sum(
            (
                predict_router_grad * normalized_predict
                + correct_router_grad * normalized_correct
            ) * _ROUTER_SCALE,
            axis=0,
        )

    tl.store(grad_router_weight + k, grad_router_0, mask=k_mask)
    tl.store(grad_router_weight + K + k, grad_router_1, mask=k_mask)
    tl.store(grad_router_weight + 2 * K + k, grad_router_2, mask=k_mask)
    tl.store(grad_norm_weight + k, grad_norm, mask=k_mask)


@triton.jit
def _dense_input_gradient_kernel(
    grad_corrected, hidden_states, activated, router_weight, norm_weight,
    state, gradient_state, grad_hidden_states, grad_activated, num_tokens,
    ACTIVE: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    k = block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < K
    state_base = state + token * 26
    gradient_base = gradient_state + token * 20

    gc0 = tl.load(
        grad_corrected + token * K + k, mask=mask, other=0.0,
    ).to(tl.float32)
    gc1 = tl.load(
        grad_corrected + (num_tokens + token) * K + k,
        mask=mask, other=0.0,
    ).to(tl.float32)
    gc2 = tl.load(
        grad_corrected + (2 * num_tokens + token) * K + k,
        mask=mask, other=0.0,
    ).to(tl.float32)

    cc0 = tl.load(state_base + 17)
    cc1 = tl.load(state_base + 18)
    cc2 = tl.load(state_base + 19)
    grad_innovation = gc0 * cc0 + gc1 * cc1 + gc2 * cc2

    if ACTIVE == 0:
        gp0 = gc0 - grad_innovation
        gp1 = gc1
        gp2 = gc2
    elif ACTIVE == 1:
        gp0 = gc0
        gp1 = gc1 - grad_innovation
        gp2 = gc2
    else:
        gp0 = gc0
        gp1 = gc1
        gp2 = gc2 - grad_innovation

    pc00 = tl.load(state_base + 8)
    pc01 = tl.load(state_base + 9)
    pc02 = tl.load(state_base + 10)
    pc10 = tl.load(state_base + 11)
    pc11 = tl.load(state_base + 12)
    pc12 = tl.load(state_base + 13)
    pc20 = tl.load(state_base + 14)
    pc21 = tl.load(state_base + 15)
    pc22 = tl.load(state_base + 16)

    grad_h0 = gp0 + gp0 * pc00 + gp1 * pc10 + gp2 * pc20
    grad_h1 = gp1 + gp0 * pc01 + gp1 * pc11 + gp2 * pc21
    grad_h2 = gp2 + gp0 * pc02 + gp1 * pc12 + gp2 * pc22

    router_0 = tl.load(
        router_weight + k, mask=mask, other=0.0,
    ).to(tl.float32)
    router_1 = tl.load(
        router_weight + K + k, mask=mask, other=0.0,
    ).to(tl.float32)
    router_2 = tl.load(
        router_weight + 2 * K + k, mask=mask, other=0.0,
    ).to(tl.float32)
    norm = tl.load(norm_weight + k, mask=mask, other=0.0).to(tl.float32)

    predict_x = tl.load(
        hidden_states + (ACTIVE * num_tokens + token) * K + k,
        mask=mask, other=0.0,
    ).to(tl.float32)
    correct_x = tl.load(
        activated + token * K + k, mask=mask, other=0.0,
    ).to(tl.float32)
    rstd_predict = tl.load(state_base + 0)
    rstd_correct = tl.load(state_base + 1)

    grad_routed_predict_0 = tl.load(gradient_base + 12)
    grad_routed_predict_1 = tl.load(gradient_base + 13)
    grad_routed_predict_2 = tl.load(gradient_base + 14)
    grad_routed_correct_0 = tl.load(gradient_base + 15)
    grad_routed_correct_1 = tl.load(gradient_base + 16)
    grad_routed_correct_2 = tl.load(gradient_base + 17)

    predict_projection_grad = (
        grad_routed_predict_0 * router_0
        + grad_routed_predict_1 * router_1
        + grad_routed_predict_2 * router_2
    ) * _ROUTER_SCALE
    correct_projection_grad = (
        grad_routed_correct_0 * router_0
        + grad_routed_correct_1 * router_1
        + grad_routed_correct_2 * router_2
    ) * _ROUTER_SCALE

    predict_routed_dot = tl.load(gradient_base + 18)
    correct_routed_dot = tl.load(gradient_base + 19)

    router_grad_predict = (
        predict_projection_grad * norm * rstd_predict
        - predict_x * rstd_predict * rstd_predict
        * (predict_routed_dot / K)
    )
    router_grad_correct = (
        correct_projection_grad * norm * rstd_correct
        - correct_x * rstd_correct * rstd_correct
        * (correct_routed_dot / K)
    )

    if ACTIVE == 0:
        grad_h0 += router_grad_predict
    elif ACTIVE == 1:
        grad_h1 += router_grad_predict
    else:
        grad_h2 += router_grad_predict

    tl.store(grad_hidden_states + token * K + k, grad_h0, mask=mask)
    tl.store(
        grad_hidden_states + (num_tokens + token) * K + k,
        grad_h1,
        mask=mask,
    )
    tl.store(
        grad_hidden_states + (2 * num_tokens + token) * K + k,
        grad_h2,
        mask=mask,
    )
    tl.store(
        grad_activated + token * K + k,
        grad_innovation + router_grad_correct,
        mask=mask,
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
    num_tokens = hidden_states.shape[1] * hidden_states.shape[2]
    device = hidden_states.device
    active = int(altup_active_idx)

    state = torch.empty(
        (num_tokens, 26), device=device, dtype=torch.float32,
    )
    gradient_state = torch.empty(
        (num_tokens, 20), device=device, dtype=torch.float32,
    )
    grad_hidden_states = torch.empty_like(hidden_states)
    grad_activated = torch.empty_like(activated)
    grad_prediction_coef_weight = torch.empty(
        prediction_coef_weight.shape, device=device, dtype=torch.float32,
    )
    grad_correction_coef_weight = torch.empty(
        correction_coef_weight.shape, device=device, dtype=torch.float32,
    )
    grad_router_weight = torch.empty(
        router_weight.shape, device=device, dtype=torch.float32,
    )
    grad_norm_weight = torch.empty(
        norm_weight.shape, device=device, dtype=torch.float32,
    )

    small_warps = 4 if num_tokens < 512 else 8

    _forward_state_kernel[(num_tokens,)](
        hidden_states,
        activated,
        prediction_coef_weight,
        correction_coef_weight,
        router_weight,
        norm_weight,
        state,
        num_tokens,
        rms_norm_eps,
        ACTIVE=active,
        K=_HIDDEN_SIZE,
        BLOCK_K=1024,
        num_warps=small_warps,
        num_stages=2,
    )

    _token_gradient_state_kernel[(num_tokens,)](
        grad_corrected,
        hidden_states,
        activated,
        prediction_coef_weight,
        correction_coef_weight,
        state,
        gradient_state,
        num_tokens,
        ACTIVE=active,
        K=_HIDDEN_SIZE,
        BLOCK_K=256,
        num_warps=small_warps,
        num_stages=2,
    )

    coefficient_block_t = 512 if num_tokens >= 512 else 256
    _coefficient_gradient_kernel[(36,)](
        state,
        gradient_state,
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        num_tokens,
        BLOCK_T=coefficient_block_t,
        num_warps=4,
        num_stages=2,
    )

    router_block_t = 64 if num_tokens >= 4096 else 32
    _router_norm_gradient_kernel[
        (triton.cdiv(_HIDDEN_SIZE, 32),)
    ](
        hidden_states,
        activated,
        router_weight,
        norm_weight,
        state,
        gradient_state,
        grad_router_weight,
        grad_norm_weight,
        num_tokens,
        ACTIVE=active,
        K=_HIDDEN_SIZE,
        BLOCK_K=32,
        BLOCK_T=router_block_t,
        num_warps=4,
        num_stages=2,
    )

    _dense_input_gradient_kernel[
        (num_tokens, triton.cdiv(_HIDDEN_SIZE, 512))
    ](
        grad_corrected,
        hidden_states,
        activated,
        router_weight,
        norm_weight,
        state,
        gradient_state,
        grad_hidden_states,
        grad_activated,
        num_tokens,
        ACTIVE=active,
        K=_HIDDEN_SIZE,
        BLOCK_K=512,
        num_warps=small_warps,
        num_stages=2,
    )

    return (
        grad_hidden_states,
        grad_activated,
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        grad_router_weight,
        grad_norm_weight,
    )