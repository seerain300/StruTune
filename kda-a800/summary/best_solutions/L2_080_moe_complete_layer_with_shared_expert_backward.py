# KDA A800 best solution: L2/080_moe_complete_layer_with_shared_expert_backward
# candidate: c001  |  feedback: 3.88x  |  final (authoritative): 4.43x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 1
# source: tasks/formal-kda-20260916--sol_execbench--L2-080_moe_complete_layer_with_shared_expert_backward/control/candidates/c001/solution.py (sha256-frozen snapshot)

"""
Candidate c001 — D0 correctness anchor (pure-Triton) for
L2/080 MoE complete-layer + shared-expert backward.

All compute is in Triton kernels. PyTorch is used ONLY for tensor
metadata (shape/stride/dtype) and output allocation / kernel launch.
No Torch / CPU / NumPy / CUDA-extension computational fallback.

Reference op decomposition (must be reproduced op-for-op):

  go = grad_output                                         # [T,H] bf16
  G1: grad_shared_activated = go @ down_w                  # [T,H]@[H,I] -> [T,I] bf16
  SW: SwiGLU backward (f32 intermediates -> bf16 stores):
        grad_gate_silu = grad_shared_activated * up_out
        grad_up_output = grad_shared_activated * silu(gate_out)
        silu'(x) = sig*(1 + x*(1-sig)),  sig = sigmoid(x)
        grad_gate_output = grad_gate_silu * silu'(gate_out)
  G3: grad_hidden += grad_up_output   @ up_w               # [T,I]@[I,H] -> [T,H]
  G4: grad_hidden += grad_gate_output @ gate_w             # [T,I]@[I,H] -> [T,H]
  R : router prologue -> grad_router_logits[T,E] (f32)
        g          = rowsum(go.f32^2) / K       (same for all K slots)
        S          = rowsum(topk_weights) + 1e-20
        sum_grad   = g * rowsum(topk_weights) / S
        gb         = (g - sum_grad) / S
        grad_scores[t, idx_k] = gb ; * score_mask
        grad_router_logits = grad_scores * scores * (1 - scores)
  G7: grad_hidden += grad_router_logits.bf16 @ router_w    # [T,E]@[E,H] -> [T,H]
  G2: g_down_w = go.T @ shared_activated                   # [H,T]@[T,I] -> [H,I] -> bf16   (output #5)
  G5: g_up_w   = grad_up_output.T @ hidden                 # [I,T]@[T,H] -> [I,H] -> bf16   (output #4)
  G6: g_gate_w = grad_gate_output.T @ hidden               # [I,T]@[T,H] -> [I,H] -> bf16   (output #3)
  G8: g_router_w = grad_router_logits.T @ hidden           # [E,T]@[T,H] -> [E,H] -> f32     (output #2)

Notes:
* All GEMMs use bf16 tensor-core inputs with fp32 accumulation. For the
  reference's "cast-to-f32-then-matmul" weight-grad GEMMs (G2/G5/G6), the
  bf16 operands are exactly representable in f32, so bf16-in/fp32-acc
  reproduces the f32 matmul products exactly before the bf16 output cast.
* The router path is numerically ~0 (isotropic grad + normalization
  Jacobian cancellation); it is still reproduced faithfully.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Generic tiled GEMM: C = A @ B  (fp32 accumulate, bf16 tensor-core compute).
# Logical transposes of A are expressed purely through strides.
# ACCUMULATE=True adds into the existing C tile (for grad_hidden fan-in).
# ---------------------------------------------------------------------------
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, Kdim,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACCUMULATE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, Kdim, BK):
        k_remaining = Kdim - k
        a = tl.load(a_ptrs, mask=m_mask & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & n_mask, other=0.0)
        acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask & n_mask
    if ACCUMULATE:
        prev = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        acc += prev
    tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=c_mask)


# ---------------------------------------------------------------------------
# SwiGLU backward elementwise over [T, I].
#   grad_up_output   = grad_shared_activated * silu(gate)
#   grad_gate_output = (grad_shared_activated * up) * silu'(gate)
# Computed in fp32, stored bf16.
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_bwd_kernel(
    GSA, UP, GATE, GUP, GGATE,
    M, Ncol,
    stride_m, stride_n,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < Ncol)

    gsa = tl.load(GSA + ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + ptrs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + ptrs, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    silu = gate * sig
    silu_p = sig * (1.0 + gate * (1.0 - sig))

    grad_gate_silu = gsa * up
    gup = gsa * silu
    ggate = grad_gate_silu * silu_p

    tl.store(GUP + ptrs, gup.to(tl.bfloat16), mask=mask)
    tl.store(GGATE + ptrs, ggate.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Router prologue: one program per token row -> grad_router_logits[t, :] (f32).
# ---------------------------------------------------------------------------
@triton.jit
def _router_kernel(
    GO, TOPKW, TOPKI, SCORES, SMASK, GRL,
    T, H, E, K,
    stride_go_t, stride_go_h,
    stride_tw_t, stride_tw_k,
    stride_ti_t, stride_ti_k,
    stride_sc_t, stride_sc_e,
    stride_sm_t, stride_sm_e,
    stride_grl_t, stride_grl_e,
    BH: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
):
    t = tl.program_id(0)

    # grad_norm_sq = sum_h go[t,h]^2 ; g = grad_norm_sq / K
    nsq = 0.0
    for h0 in range(0, H, BH):
        offs_h = h0 + tl.arange(0, BH)
        go = tl.load(GO + t * stride_go_t + offs_h * stride_go_h,
                     mask=offs_h < H, other=0.0).to(tl.float32)
        nsq += tl.sum(go * go, axis=0)
    g = nsq / K

    # S = sum_k topk_weights[t,k] + 1e-20
    offs_k = tl.arange(0, BK)
    tw = tl.load(TOPKW + t * stride_tw_t + offs_k * stride_tw_k,
                 mask=offs_k < K, other=0.0)
    sum_tw = tl.sum(tw, axis=0)
    S = sum_tw + 1e-20
    sum_grad = (g * sum_tw) / S
    gb = (g - sum_grad) / S  # grad_before_norm, identical across selected slots

    # grad_scores over E (only selected experts nonzero)
    offs_e = tl.arange(0, BE)
    gs = tl.zeros([BE], dtype=tl.float32)
    for k in range(0, K):
        idx = tl.load(TOPKI + t * stride_ti_t + k * stride_ti_k).to(tl.int32)
        gs += tl.where(offs_e == idx, gb, 0.0)

    smask = tl.load(SMASK + t * stride_sm_t + offs_e * stride_sm_e,
                    mask=offs_e < E, other=0.0)
    sc = tl.load(SCORES + t * stride_sc_t + offs_e * stride_sc_e,
                 mask=offs_e < E, other=0.0)
    grl = (gs * smask) * sc * (1.0 - sc)
    tl.store(GRL + t * stride_grl_t + offs_e * stride_grl_e, grl, mask=offs_e < E)


# ---------------------------------------------------------------------------
# Host-side launch helpers (metadata / plumbing only).
# ---------------------------------------------------------------------------
def _gemm(A, B, C, M, N, Kd,
          sam, sak, sbk, sbn, scm, scn,
          accumulate, out_dtype,
          BM=64, BN=64, BK=32, num_warps=4, num_stages=3):
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm_kernel[grid](
        A, B, C, M, N, Kd,
        sam, sak, sbk, sbn, scm, scn,
        ACCUMULATE=accumulate, OUT_DTYPE=out_dtype,
        BM=BM, BN=BN, BK=BK,
        num_warps=num_warps, num_stages=num_stages,
    )


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    router_weight,
    e_score_correction_bias,
    router_logits,
    scores,
    topk_indices,
    topk_weights,
    score_mask,
    shared_expert_gate_weight,
    shared_expert_up_weight,
    shared_expert_down_weight,
    shared_gate_output,
    shared_up_output,
    shared_activated,
):
    T, H = hidden_states.shape
    I = shared_expert_gate_weight.shape[0]
    E = router_weight.shape[0]
    K = topk_indices.shape[1]
    device = hidden_states.device
    bf16 = torch.bfloat16
    f32 = torch.float32

    go = grad_output

    # ---- allocations (outputs + intermediates) ----
    grad_shared_activated = torch.empty((T, I), dtype=bf16, device=device)
    grad_up_output = torch.empty((T, I), dtype=bf16, device=device)
    grad_gate_output = torch.empty((T, I), dtype=bf16, device=device)
    grad_router_logits = torch.empty((T, E), dtype=f32, device=device)

    grad_hidden_states = torch.empty((T, H), dtype=bf16, device=device)      # output #1
    grad_router_weight = torch.empty((E, H), dtype=f32, device=device)       # output #2
    grad_shared_expert_gate_weight = torch.empty((I, H), dtype=bf16, device=device)  # output #3
    grad_shared_expert_up_weight = torch.empty((I, H), dtype=bf16, device=device)    # output #4
    grad_shared_expert_down_weight = torch.empty((H, I), dtype=bf16, device=device)  # output #5

    # ---- G1: grad_shared_activated = go @ down_w  ([T,H]@[H,I]) ----
    _gemm(
        go, shared_expert_down_weight, grad_shared_activated,
        T, I, H,
        go.stride(0), go.stride(1),
        shared_expert_down_weight.stride(0), shared_expert_down_weight.stride(1),
        grad_shared_activated.stride(0), grad_shared_activated.stride(1),
        accumulate=False, out_dtype=tl.bfloat16,
    )

    # ---- SW: SwiGLU backward -> grad_up_output, grad_gate_output ----
    BM_E, BN_E = 64, 64
    grid_sw = (triton.cdiv(T, BM_E), triton.cdiv(I, BN_E))
    _swiglu_bwd_kernel[grid_sw](
        grad_shared_activated, shared_up_output, shared_gate_output,
        grad_up_output, grad_gate_output,
        T, I,
        grad_shared_activated.stride(0), grad_shared_activated.stride(1),
        BM=BM_E, BN=BN_E,
    )

    # ---- G3: grad_hidden = grad_up_output @ up_w   ([T,I]@[I,H]) ----
    _gemm(
        grad_up_output, shared_expert_up_weight, grad_hidden_states,
        T, H, I,
        grad_up_output.stride(0), grad_up_output.stride(1),
        shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
        grad_hidden_states.stride(0), grad_hidden_states.stride(1),
        accumulate=False, out_dtype=tl.bfloat16,
    )
    # ---- G4: grad_hidden += grad_gate_output @ gate_w ----
    _gemm(
        grad_gate_output, shared_expert_gate_weight, grad_hidden_states,
        T, H, I,
        grad_gate_output.stride(0), grad_gate_output.stride(1),
        shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
        grad_hidden_states.stride(0), grad_hidden_states.stride(1),
        accumulate=True, out_dtype=tl.bfloat16,
    )

    # ---- R: router prologue -> grad_router_logits ----
    _router_kernel[(T,)](
        go, topk_weights, topk_indices, scores, score_mask, grad_router_logits,
        T, H, E, K,
        go.stride(0), go.stride(1),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
        scores.stride(0), scores.stride(1),
        score_mask.stride(0), score_mask.stride(1),
        grad_router_logits.stride(0), grad_router_logits.stride(1),
        BH=1024, BE=triton.next_power_of_2(E), BK=triton.next_power_of_2(K),
    )

    # ---- G7: grad_hidden += grad_router_logits @ router_w ([T,E]@[E,H]) ----
    _gemm(
        grad_router_logits, router_weight, grad_hidden_states,
        T, H, E,
        grad_router_logits.stride(0), grad_router_logits.stride(1),
        router_weight.stride(0), router_weight.stride(1),
        grad_hidden_states.stride(0), grad_hidden_states.stride(1),
        accumulate=True, out_dtype=tl.bfloat16,
    )

    # ---- G2: g_down_w = go.T @ shared_activated ([H,T]@[T,I]) ----
    _gemm(
        go, shared_activated, grad_shared_expert_down_weight,
        H, I, T,
        go.stride(1), go.stride(0),                       # transposed A: [H,T]
        shared_activated.stride(0), shared_activated.stride(1),
        grad_shared_expert_down_weight.stride(0), grad_shared_expert_down_weight.stride(1),
        accumulate=False, out_dtype=tl.bfloat16,
    )

    # ---- G5: g_up_w = grad_up_output.T @ hidden ([I,T]@[T,H]) ----
    _gemm(
        grad_up_output, hidden_states, grad_shared_expert_up_weight,
        I, H, T,
        grad_up_output.stride(1), grad_up_output.stride(0),   # transposed A: [I,T]
        hidden_states.stride(0), hidden_states.stride(1),
        grad_shared_expert_up_weight.stride(0), grad_shared_expert_up_weight.stride(1),
        accumulate=False, out_dtype=tl.bfloat16,
    )

    # ---- G6: g_gate_w = grad_gate_output.T @ hidden ([I,T]@[T,H]) ----
    _gemm(
        grad_gate_output, hidden_states, grad_shared_expert_gate_weight,
        I, H, T,
        grad_gate_output.stride(1), grad_gate_output.stride(0),  # transposed A: [I,T]
        hidden_states.stride(0), hidden_states.stride(1),
        grad_shared_expert_gate_weight.stride(0), grad_shared_expert_gate_weight.stride(1),
        accumulate=False, out_dtype=tl.bfloat16,
    )

    # ---- G8: g_router_w = grad_router_logits.T @ hidden ([E,T]@[T,H]) -> f32 ----
    _gemm(
        grad_router_logits, hidden_states, grad_router_weight,
        E, H, T,
        grad_router_logits.stride(1), grad_router_logits.stride(0),  # transposed A: [E,T]
        hidden_states.stride(0), hidden_states.stride(1),
        grad_router_weight.stride(0), grad_router_weight.stride(1),
        accumulate=False, out_dtype=tl.float32,
    )

    return (
        grad_hidden_states,
        grad_router_weight,
        grad_shared_expert_gate_weight,
        grad_shared_expert_up_weight,
        grad_shared_expert_down_weight,
    )
