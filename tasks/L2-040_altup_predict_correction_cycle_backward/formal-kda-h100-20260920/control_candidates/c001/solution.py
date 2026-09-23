import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # Numerically stable tanh, avoids libdevice version differences.
    e = tl.exp(-2.0 * tl.abs(x))
    y = (1.0 - e) / (1.0 + e)
    return tl.where(x >= 0, y, -y)


@triton.jit
def _altup_bwd_kernel(
    grad_corrected_ptr,      # [N, M, H] bf16
    hidden_states_ptr,       # [N, M, H] bf16
    activated_ptr,           # [M, H]    bf16
    pcw_ptr,                 # [9, 3]    bf16 (prediction_coef_weight)
    ccw_ptr,                 # [3, 3]    bf16 (correction_coef_weight)
    rw_ptr,                  # [3, H]    bf16 (router_weight)
    nw_ptr,                  # [H]       bf16 (norm_weight)
    grad_hidden_states_ptr,  # [N, M, H] bf16
    grad_activated_ptr,      # [M, H]    bf16
    grad_pcw_ptr,            # [9, 3]    fp32
    grad_ccw_ptr,            # [3, 3]    fp32
    grad_rw_ptr,             # [3, H]    fp32
    grad_nw_ptr,             # [H]       fp32
    M, H,
    eps,
    rs,                      # router_scale = 1/H
    ACTIVE_IDX: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)

    h = tl.arange(0, BLOCK_H)
    hmask = h < H

    stride_n = M * H  # N-dim stride in [N, M, H]
    row = t * H

    # ---- big-vector loads (upcast to fp32) ----
    h0 = tl.load(hidden_states_ptr + 0 * stride_n + row + h, mask=hmask, other=0.0).to(tl.float32)
    h1 = tl.load(hidden_states_ptr + 1 * stride_n + row + h, mask=hmask, other=0.0).to(tl.float32)
    h2 = tl.load(hidden_states_ptr + 2 * stride_n + row + h, mask=hmask, other=0.0).to(tl.float32)
    xa = tl.load(activated_ptr + row + h, mask=hmask, other=0.0).to(tl.float32)
    gc0 = tl.load(grad_corrected_ptr + 0 * stride_n + row + h, mask=hmask, other=0.0).to(tl.float32)
    gc1 = tl.load(grad_corrected_ptr + 1 * stride_n + row + h, mask=hmask, other=0.0).to(tl.float32)
    gc2 = tl.load(grad_corrected_ptr + 2 * stride_n + row + h, mask=hmask, other=0.0).to(tl.float32)

    nw = tl.load(nw_ptr + h, mask=hmask, other=0.0).to(tl.float32)
    rw0 = tl.load(rw_ptr + 0 * H + h, mask=hmask, other=0.0).to(tl.float32)
    rw1 = tl.load(rw_ptr + 1 * H + h, mask=hmask, other=0.0).to(tl.float32)
    rw2 = tl.load(rw_ptr + 2 * H + h, mask=hmask, other=0.0).to(tl.float32)

    h_list = [h0, h1, h2]
    gc_list = [gc0, gc1, gc2]
    rw_list = [rw0, rw1, rw2]

    # small coef weights as scalars (compile-time unrolled)
    pcw = [[tl.load(pcw_ptr + c * 3 + n).to(tl.float32) for n in range(3)] for c in range(9)]
    ccw = [[tl.load(ccw_ptr + n * 3 + m).to(tl.float32) for m in range(3)] for n in range(3)]

    Hf = H.to(tl.float32)

    # ===================== FORWARD RECOMPUTE =====================
    x_p = h_list[ACTIVE_IDX]

    # predict RMSNorm
    var_p = tl.sum(x_p * x_p, axis=0) / Hf
    rstd_p = 1.0 / tl.sqrt(var_p + eps)
    normalized_p = x_p * rstd_p
    scaled_p = normalized_p * nw * rs

    # correct RMSNorm
    var_c = tl.sum(xa * xa, axis=0) / Hf
    rstd_c = 1.0 / tl.sqrt(var_c + eps)
    normalized_c = xa * rstd_c
    scaled_c = normalized_c * nw * rs

    # router forward (fp32 matvec over H, no tl.dot)
    routed_p = [tl.sum(scaled_p * rw_list[n], axis=0) for n in range(3)]
    routed_c = [tl.sum(scaled_c * rw_list[n], axis=0) for n in range(3)]
    mod_p = [_tanh(routed_p[n]) for n in range(3)]
    mod_c = [_tanh(routed_c[n]) for n in range(3)]

    # prediction coefficients: coefs_flat[c] = sum_n mod_p[n]*pcw[c,n]
    coefs_flat = [mod_p[0] * pcw[c][0] + mod_p[1] * pcw[c][1] + mod_p[2] * pcw[c][2] for c in range(9)]

    # correction coefficients: coefs_correct[n] = sum_m mod_c[m]*ccw[n,m] + 1
    coefs_correct = [mod_c[0] * ccw[n][0] + mod_c[1] * ccw[n][1] + mod_c[2] * ccw[n][2] + 1.0 for n in range(3)]

    # predictions[idx] = sum_k h_k * coefs_flat[3*idx + k] + h_idx
    predictions_idx = h_list[ACTIVE_IDX]
    for k in range(3):
        predictions_idx = predictions_idx + h_list[k] * coefs_flat[3 * ACTIVE_IDX + k]
    innovation = xa - predictions_idx

    # ===================== CORRECT BACKWARD =====================
    grad_innovation = coefs_correct[0] * gc0 + coefs_correct[1] * gc1 + coefs_correct[2] * gc2
    grad_coefs_correct = [tl.sum(gc_list[k] * innovation, axis=0) for k in range(3)]

    # grad_correction_coef_weight[n,m] += grad_coefs_correct[n]*mod_c[m]
    for n in range(3):
        for m in range(3):
            tl.atomic_add(grad_ccw_ptr + n * 3 + m, grad_coefs_correct[n] * mod_c[m])

    grad_mod_c = [grad_coefs_correct[0] * ccw[0][m] + grad_coefs_correct[1] * ccw[1][m] + grad_coefs_correct[2] * ccw[2][m] for m in range(3)]
    grad_routed_c = [grad_mod_c[m] * (1.0 - mod_c[m] * mod_c[m]) for m in range(3)]

    grad_scaled_c = grad_routed_c[0] * rw0 + grad_routed_c[1] * rw1 + grad_routed_c[2] * rw2
    grad_normed_c = grad_scaled_c * rs
    grad_normalized_c = grad_normed_c * nw
    mean_c = tl.sum(grad_normalized_c * xa, axis=0) / Hf
    rstd_c3 = rstd_c * rstd_c * rstd_c
    grad_act_router = grad_normalized_c * rstd_c - xa * rstd_c3 * mean_c
    grad_activated = grad_innovation + grad_act_router
    tl.store(grad_activated_ptr + row + h, grad_activated.to(tl.bfloat16), mask=hmask)

    # grad_predictions
    gp = [gc0, gc1, gc2]
    gp[ACTIVE_IDX] = gp[ACTIVE_IDX] - grad_innovation

    # ===================== PREDICT BACKWARD =====================
    # grad_h_permuted[:,i] = sum_j gp[j]*coefs_flat[3*j + i]
    grad_h_perm = [gp[0] * coefs_flat[0 * 3 + i] + gp[1] * coefs_flat[1 * 3 + i] + gp[2] * coefs_flat[2 * 3 + i] for i in range(3)]

    # grad_all_coefs_flat[c] = sum_h h_{c%3}*gp_{c//3}
    grad_acf = [tl.sum(h_list[c % 3] * gp[c // 3], axis=0) for c in range(9)]

    # grad_prediction_coef_weight[c,n] += grad_acf[c]*mod_p[n]
    for c in range(9):
        for n in range(3):
            tl.atomic_add(grad_pcw_ptr + c * 3 + n, grad_acf[c] * mod_p[n])

    grad_mod_p = [sum(grad_acf[c] * pcw[c][n] for c in range(9)) for n in range(3)]
    grad_routed_p = [grad_mod_p[n] * (1.0 - mod_p[n] * mod_p[n]) for n in range(3)]

    grad_scaled_p = grad_routed_p[0] * rw0 + grad_routed_p[1] * rw1 + grad_routed_p[2] * rw2
    grad_normed_p = grad_scaled_p * rs
    grad_normalized_p = grad_normed_p * nw
    mean_p = tl.sum(grad_normalized_p * x_p, axis=0) / Hf
    rstd_p3 = rstd_p * rstd_p * rstd_p
    grad_active_input = grad_normalized_p * rstd_p - x_p * rstd_p3 * mean_p

    # grad_hidden_states[k] = gp[k] + grad_h_perm[k] ; idx row += grad_active_input
    ghs = [gp[k] + grad_h_perm[k] for k in range(3)]
    ghs[ACTIVE_IDX] = ghs[ACTIVE_IDX] + grad_active_input
    for k in range(3):
        tl.store(grad_hidden_states_ptr + k * stride_n + row + h, ghs[k].to(tl.bfloat16), mask=hmask)

    # ---- router / norm weight grads (predict + correct combined) ----
    for m in range(3):
        grw = grad_routed_c[m] * scaled_c + grad_routed_p[m] * scaled_p
        tl.atomic_add(grad_rw_ptr + m * H + h, grw, mask=hmask)

    gnw = grad_normed_c * normalized_c + grad_normed_p * normalized_p
    tl.atomic_add(grad_nw_ptr + h, gnw, mask=hmask)


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
    N = 3
    H = 2304
    assert hidden_states.shape[0] == N
    B = hidden_states.shape[1]
    S = hidden_states.shape[2]
    assert hidden_states.shape[3] == H
    M = B * S

    gc = grad_corrected.contiguous()
    hs = hidden_states.contiguous()
    act = activated.contiguous()
    pcw = prediction_coef_weight.contiguous()
    ccw = correction_coef_weight.contiguous()
    rw = router_weight.contiguous()
    nw = norm_weight.contiguous()

    grad_hidden_states = torch.empty_like(hs)
    grad_activated = torch.empty_like(act)
    grad_pcw = torch.zeros((N * N, N), dtype=torch.float32, device=hs.device)
    grad_ccw = torch.zeros((N, N), dtype=torch.float32, device=hs.device)
    grad_rw = torch.zeros((N, H), dtype=torch.float32, device=hs.device)
    grad_nw = torch.zeros((H,), dtype=torch.float32, device=hs.device)

    BLOCK_H = triton.next_power_of_2(H)
    grid = (M,)
    _altup_bwd_kernel[grid](
        gc, hs, act, pcw, ccw, rw, nw,
        grad_hidden_states, grad_activated,
        grad_pcw, grad_ccw, grad_rw, grad_nw,
        M, H,
        float(rms_norm_eps),
        1.0 / H,
        ACTIVE_IDX=int(altup_active_idx),
        BLOCK_H=BLOCK_H,
        num_warps=8,
    )

    return (
        grad_hidden_states,
        grad_activated,
        grad_pcw,
        grad_ccw,
        grad_rw,
        grad_nw,
    )
