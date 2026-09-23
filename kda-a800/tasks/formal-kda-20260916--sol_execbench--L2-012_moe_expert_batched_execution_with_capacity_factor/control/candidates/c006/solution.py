"""L2/012 MoE expert batched execution with capacity factor.

Candidate c006 — deepen cp.async pipelining (num_stages 3 -> 4).

Change from c005 (single attributable lever: pipeline depth): both kernels are
launched with num_stages=4 instead of 3. All avoidable HBM traffic has been
removed (c002-c005), so the memory-bound four are now bounded by the ~15 GB
weight-streaming floor; deeper cp.async software pipelining is the draft's
single most important knob for hiding that weight-load latency behind the
tensor-core math. Shared-memory budget check: kernel A per-stage tiles are
x[128,32]+wg[128,32]+wu[128,32] bf16 ~= 24 KB, so 4 stages ~= 96 KB < 164 KB/SM.
Everything else (both kernel bodies, drop set, tile shape, fp32 accumulation) is
byte-for-byte identical to c005.

No Torch/CPU/NumPy/CUDA-extension computational fallback for the FFN math.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: gather X from hidden_states + gate & up projections + fused SwiGLU
#           ->  h[E, capacity, I]
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_swiglu_kernel(
    hs_ptr,         # hidden_states [N, H] bf16
    rtok_ptr,       # row_token [E, CAP] int32 (sentinel = N for padding)
    wg_ptr,         # expert_gate_weights [E, H, I] bf16
    wu_ptr,         # expert_up_weights   [E, H, I] bf16
    h_ptr,          # out h [E, CAP, I] bf16
    CAP, H, I, N,
    stride_hs_n, stride_hs_h,
    stride_re, stride_rm,
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

    m_mask = offs_m < CAP
    n_mask = offs_n < I

    # Gather this tile's token ids (padding rows -> sentinel N).
    tok = tl.load(rtok_ptr + pid_e * stride_re + offs_m * stride_rm, mask=m_mask, other=N)
    row_valid = m_mask & (tok < N)
    tok = tl.where(row_valid, tok, 0).to(tl.int64)  # keep pointer in bounds when masked

    wg_base = wg_ptr + pid_e * stride_we
    wu_base = wu_ptr + pid_e * stride_we

    # On-the-fly gather: x row m = hidden_states[tok[m], :]
    x_ptrs = hs_ptr + tok[:, None] * stride_hs_n + offs_k[None, :] * stride_hs_h
    wg_ptrs = wg_base + offs_k[:, None] * stride_wh + offs_n[None, :] * stride_wi
    wu_ptrs = wu_base + offs_k[:, None] * stride_wh + offs_n[None, :] * stride_wi

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_mask = (k + offs_k) < H
        x = tl.load(x_ptrs, mask=row_valid[:, None] & k_mask[None, :], other=0.0)
        wg = tl.load(wg_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        wu = tl.load(wu_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc_g += tl.dot(x, wg)
        acc_u += tl.dot(x, wu)
        x_ptrs += BLOCK_K * stride_hs_h
        wg_ptrs += BLOCK_K * stride_wh
        wu_ptrs += BLOCK_K * stride_wh

    # SwiGLU: silu(gate) * up, computed in fp32.
    silu_g = acc_g * tl.sigmoid(acc_g)
    h = silu_g * acc_u

    h_ptrs = h_ptr + pid_e * stride_he + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hi
    tl.store(h_ptrs, h.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel B': down projection + weighted scatter-add into fp32 result[N, H]
# ---------------------------------------------------------------------------
@triton.jit
def _down_scatter_kernel(
    h_ptr,          # h [E, CAP, I] bf16
    wd_ptr,         # expert_down_weights [E, I, H] bf16
    rtok_ptr,       # row_token [E, CAP] int32 (sentinel = N for padding)
    rwt_ptr,        # row_weight [E, CAP] fp32
    res_ptr,        # result_f32 [N, H] fp32
    CAP, I, H, N,
    stride_he, stride_hm, stride_hi,
    stride_we, stride_wi, stride_wh,
    stride_re, stride_rm,
    stride_res_n, stride_res_h,
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

    # Per-row token id + routing weight (padding rows -> sentinel N, weight 0).
    r_offs = pid_e * stride_re + offs_m * stride_rm
    tok = tl.load(rtok_ptr + r_offs, mask=m_mask, other=N)
    wt = tl.load(rwt_ptr + r_offs, mask=m_mask, other=0.0)

    # Weight each row (fp32) then scatter-add into result by token id.
    acc = acc * wt[:, None]

    row_valid = m_mask & (tok < N)
    tok = tl.where(row_valid, tok, 0).to(tl.int64)  # keep pointer in bounds when masked

    res_ptrs = res_ptr + tok[:, None] * stride_res_n + offs_n[None, :] * stride_res_h
    tl.atomic_add(res_ptrs, acc, mask=row_valid[:, None] & n_mask[None, :])


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

    # Padded per-row token id (sentinel N) + routing weight (0). Used by kernel A
    # (on-the-fly input gather) and kernel B' (weighted scatter-add). No
    # expert_inputs[E,cap,H] scatter is materialized anymore.
    row_token = torch.full((E, capacity), num_tokens, dtype=torch.int32, device=device)
    row_token[v_exp, v_pos] = v_tok.to(torch.int32)
    row_weight = torch.zeros(E, capacity, dtype=torch.float32, device=device)
    row_weight[v_exp, v_pos] = v_wt.to(torch.float32)

    # Ensure contiguity for kernels.
    hs = hidden_states.contiguous()
    wg = expert_gate_weights.contiguous()
    wu = expert_up_weights.contiguous()
    wd = expert_down_weights.contiguous()

    # ---- kernel A: gather X + gate + up + fused SwiGLU -> h[E, cap, I] ------
    # BLOCK_M=128 makes ceil(cap/BM)=1 for the memory-bound four (cap 84-98),
    # so each expert's weight tiles are streamed from HBM once, not twice.
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 8
    num_stages = 4

    h = torch.empty(E, capacity, I, dtype=dtype, device=device)
    grid_a = (E, triton.cdiv(capacity, BLOCK_M), triton.cdiv(I, BLOCK_N))
    _gate_up_swiglu_kernel[grid_a](
        hs, row_token, wg, wu, h,
        capacity, H, I, num_tokens,
        hs.stride(0), hs.stride(1),
        row_token.stride(0), row_token.stride(1),
        wg.stride(0), wg.stride(1), wg.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    # ---- kernel B': down projection + weighted scatter-add -> result[N,H] --
    result_f32 = torch.zeros(num_tokens, H, dtype=torch.float32, device=device)
    grid_b = (E, triton.cdiv(capacity, BLOCK_M), triton.cdiv(H, BLOCK_N))
    _down_scatter_kernel[grid_b](
        h, wd, row_token, row_weight, result_f32,
        capacity, I, H, num_tokens,
        h.stride(0), h.stride(1), h.stride(2),
        wd.stride(0), wd.stride(1), wd.stride(2),
        row_token.stride(0), row_token.stride(1),
        result_f32.stride(0), result_f32.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    return result_f32.to(dtype)
