"""L2/012 MoE expert batched execution with capacity factor.

Candidate c002 — correctness-first Triton baseline with int64 pointer fix.

Change from c001 (single attributable variable): in both Triton kernels the
per-expert base-pointer offset is now computed in int64. c001 failed 0/5 with
catastrophic errors (max_abs ~100+) because ``pid_e * stride_we`` was computed
in int32; the gate/up/down weight tensors are [E,H,I]=[160,6144,2560] so
``stride_we = H*I = 15,728,640`` and the product overflows int32 for
``pid_e >= 137``, making experts 137..159 read garbage weights. Casting
``pid_e`` to int64 promotes all base-offset arithmetic to 64-bit. Everything
else is byte-for-byte identical to c001 (still the correctness anchor; the
fused-tail optimization originally scoped for c002 is deferred to c003 to keep
one variable per candidate).

Design (see docs/plan.md §2 c001):
  * Admission metadata (capacity, stable sort, within-expert position, capacity
    mask, gather/scatter of admitted rows) is computed in torch. These are
    tensor-metadata / data-movement plumbing, explicitly allowed. This
    guarantees a drop set identical to the reference.
  * The full expert FFN compute — the three GEMMs and the SwiGLU activation —
    runs entirely in Triton with fp32 accumulators:
      kernel A: gate = X @ Wg ; up = X @ Wu ; h = silu(gate) * up   -> h[E,cap,I]
      kernel B: y = h @ Wd                                           -> y[E,cap,H]
  * Weighted aggregation back to [N, H] uses torch index_add_ (a scatter-add of
    the admitted rows), mirroring the reference exactly.

No Torch/CPU/NumPy/CUDA-extension computational fallback for the FFN math.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: gate & up projections + fused SwiGLU  ->  h[E, capacity, I]
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_swiglu_kernel(
    x_ptr,          # expert_inputs [E, CAP, H] bf16
    wg_ptr,         # expert_gate_weights [E, H, I] bf16
    wu_ptr,         # expert_up_weights   [E, H, I] bf16
    h_ptr,          # out h [E, CAP, I] bf16
    CAP, H, I,
    stride_xe, stride_xm, stride_xh,
    stride_we, stride_wh, stride_wi,
    stride_he, stride_hm, stride_hi,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Promote expert id to int64: base offsets pid_e*stride can exceed int32
    # (stride_we = H*I = 15,728,640; overflows for pid_e >= 137).
    pid_e = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_base = x_ptr + pid_e * stride_xe
    wg_base = wg_ptr + pid_e * stride_we
    wu_base = wu_ptr + pid_e * stride_we

    x_ptrs = x_base + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xh
    wg_ptrs = wg_base + offs_k[:, None] * stride_wh + offs_n[None, :] * stride_wi
    wu_ptrs = wu_base + offs_k[:, None] * stride_wh + offs_n[None, :] * stride_wi

    m_mask = offs_m < CAP
    n_mask = offs_n < I

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_mask = (k + offs_k) < H
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wg = tl.load(wg_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        wu = tl.load(wu_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc_g += tl.dot(x, wg)
        acc_u += tl.dot(x, wu)
        x_ptrs += BLOCK_K * stride_xh
        wg_ptrs += BLOCK_K * stride_wh
        wu_ptrs += BLOCK_K * stride_wh

    # SwiGLU: silu(gate) * up, computed in fp32.
    silu_g = acc_g * tl.sigmoid(acc_g)
    h = silu_g * acc_u

    h_ptrs = h_ptr + pid_e * stride_he + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hi
    tl.store(h_ptrs, h.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel B: down projection  ->  y[E, capacity, H]
# ---------------------------------------------------------------------------
@triton.jit
def _down_kernel(
    h_ptr,          # h [E, CAP, I] bf16
    wd_ptr,         # expert_down_weights [E, I, H] bf16
    y_ptr,          # out y [E, CAP, H] bf16
    CAP, I, H,
    stride_he, stride_hm, stride_hi,
    stride_we, stride_wi, stride_wh,
    stride_ye, stride_ym, stride_yh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Promote expert id to int64 (see kernel A note): wd stride = I*H overflows int32.
    pid_e = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    h_base = h_ptr + pid_e * stride_he
    wd_base = wd_ptr + pid_e * stride_we

    h_ptrs = h_base + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hi
    wd_ptrs = wd_base + offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wh

    m_mask = offs_m < CAP
    n_mask = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, I, BLOCK_K):
        k_mask = (k + offs_k) < I
        h = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wd = tl.load(wd_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(h, wd)
        h_ptrs += BLOCK_K * stride_hi
        wd_ptrs += BLOCK_K * stride_wi

    y_ptrs = y_ptr + pid_e * stride_ye + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yh
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    # ---- static self-checks (fail loudly, do not silently mismatch) --------
    assert hidden_states.is_cuda, "hidden_states must be on CUDA"
    assert hidden_states.dtype == torch.bfloat16
    assert expert_gate_weights.dtype == torch.bfloat16
    assert expert_up_weights.dtype == torch.bfloat16
    assert expert_down_weights.dtype == torch.bfloat16

    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, moe_intermediate_size = expert_gate_weights.shape
    num_experts_per_tok = selected_experts.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    H = hidden_size
    I = moe_intermediate_size
    E = num_experts
    K = num_experts_per_tok

    assert expert_gate_weights.shape == (E, H, I)
    assert expert_up_weights.shape == (E, H, I)
    assert expert_down_weights.shape == (E, I, H)

    # ---- admission metadata (torch; identical drop set to the reference) ---
    capacity = max(int((num_tokens * K / E) * 1.25), 1)

    flat_experts = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    flat_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(K)

    sorted_experts, sorted_indices = flat_experts.sort(stable=True)
    sorted_weights = flat_weights[sorted_indices]
    sorted_token_ids = flat_token_ids[sorted_indices]

    counts = torch.bincount(sorted_experts, minlength=E)
    starts = torch.zeros(E, dtype=torch.long, device=device)
    starts[1:] = counts[:-1].cumsum(0)
    within_pos = torch.arange(len(sorted_experts), device=device) - starts[sorted_experts]

    valid = within_pos < capacity
    v_exp = sorted_experts[valid]
    v_pos = within_pos[valid]
    v_tok = sorted_token_ids[valid]
    v_wt = sorted_weights[valid]

    # Scatter admitted rows into padded per-expert batches (data movement).
    expert_inputs = torch.zeros(E, capacity, H, dtype=dtype, device=device)
    expert_inputs[v_exp, v_pos] = hidden_states[v_tok]

    # Ensure contiguity for kernels.
    wg = expert_gate_weights.contiguous()
    wu = expert_up_weights.contiguous()
    wd = expert_down_weights.contiguous()

    # ---- kernel A: gate + up + fused SwiGLU -> h[E, cap, I] ----------------
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 4
    num_stages = 3

    h = torch.empty(E, capacity, I, dtype=dtype, device=device)
    grid_a = (E, triton.cdiv(capacity, BLOCK_M), triton.cdiv(I, BLOCK_N))
    _gate_up_swiglu_kernel[grid_a](
        expert_inputs, wg, wu, h,
        capacity, H, I,
        expert_inputs.stride(0), expert_inputs.stride(1), expert_inputs.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    # ---- kernel B: down projection -> y[E, cap, H] ------------------------
    y = torch.empty(E, capacity, H, dtype=dtype, device=device)
    grid_b = (E, triton.cdiv(capacity, BLOCK_M), triton.cdiv(H, BLOCK_N))
    _down_kernel[grid_b](
        h, wd, y,
        capacity, I, H,
        h.stride(0), h.stride(1), h.stride(2),
        wd.stride(0), wd.stride(1), wd.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    # ---- weighted aggregation back to [N, H] (scatter-add plumbing) -------
    valid_out = y[v_exp, v_pos]              # (num_valid, H)
    weighted_out = v_wt.unsqueeze(1) * valid_out
    result = torch.zeros(num_tokens, H, dtype=dtype, device=device)
    result.index_add_(0, v_tok, weighted_out)

    return result
