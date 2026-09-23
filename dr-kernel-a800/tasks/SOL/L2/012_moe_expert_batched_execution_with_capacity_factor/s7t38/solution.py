import math
import torch
import triton
import triton.language as tl


# Kernel 1: Flatten and sort by keys (selected_experts) with stable order.
# Inputs: selected_experts flattened [T], routing_weights flattened [T], token_ids [T].
# Outputs: sorted_experts [T], sorted_weights [T], sorted_token_ids [T].
@triton.jit
def sort_by_keys_stable_kernel(keys_ptr, vals_ptr, tok_ptr,
                               out_keys_ptr, out_vals_ptr, out_tok_ptr,
                               T: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    # Load with mask; pad keys with max int64 so they go to the end of sort
    a = tl.load(keys_ptr + idx, mask=idx < T, other=tl.max_int64)  # int64 keys
    b = tl.load(vals_ptr + idx, mask=idx < T, other=0.0)          # bfloat16 weights
    c = tl.load(tok_ptr + idx, mask=idx < T, other=0)             # int64 token ids

    # Bitonic sort network over BLOCK lanes to sort ascending by keys (int64).
    # Stable tie-break for equal keys: preserve original idx order by using idx as tie-breaker.
    for size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        if size > BLOCK:
            break
        for stride in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            if stride > size:
                break
            i = idx
            j = i ^ (stride // 2)
            asc = (i & size) == 0
            a_i = a[i]
            a_j = a[j]
            # Swap when ascending and a_i > a_j, or descending and a_i < a_j.
            # For equal keys, swap when idx_i > idx_j to maintain original order (stable).
            cond = tl.where(asc, a_i > a_j, a_i < a_j)
            cond |= (a_i == a_j) & (i > j)
            ai_new = tl.where(cond, a_j, a_i)
            bi_new = tl.where(cond, b[j], b[i])
            ci_new = tl.where(cond, c[j], c[i])
            a = tl.where(i == idx, ai_new, a)
            b = tl.where(i == idx, bi_new, b)
            c = tl.where(i == idx, ci_new, c)

    tl.store(out_keys_ptr + idx, a, mask=idx < T)
    tl.store(out_vals_ptr + idx, b, mask=idx < T)
    tl.store(out_tok_ptr + idx, c, mask=idx < T)


# Kernel 2: Per-expert counts (bincount) of sorted_experts. Input: keys (sorted_experts), Output: counts[num_experts].
@triton.jit
def bincount_kernel(keys_ptr, counts_ptr, T: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    vals = tl.load(keys_ptr + idx, mask=idx < T, other=0)  # int64
    # Each lane counts how many times its value appears
    cnt = tl.zeros((), dtype=tl.int32)
    # Loop over all elements
    for k in range(0, T):
        vk = vals[k]  # scalar load of one key
        # increment counts for lane where vals == vk
        mask = (vals == vk) & (idx < T)
        cnt += tl.sum(mask.to(tl.int32))
    # Atomic add to global counts
    tl.atomic_add(counts_ptr + vals, cnt)
    # NOTE: This is a Triton-only reduction; torch is not used.


# Kernel 3: Prefix sum (cumsum) of counts to get starts. Input: counts[num_experts], Output: starts[num_experts].
@triton.jit
def cumsum_starts_kernel(counts_ptr, starts_ptr, NUM_EXPERTS: tl.constexpr):
    # Serial scan; small size. Launch NUM_EXPERTS programs; each computes its start.
    e = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.int64)
    cnt = tl.load(counts_ptr + e)
    acc = acc + cnt
    tl.store(starts_ptr + e, acc)


# Kernel 4: Compute within_pos = index - starts[sorted_experts[index]] for all T elements.
@triton.jit
def within_pos_kernel(sorted_keys_ptr, starts_ptr, within_ptr, T: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    sk = tl.load(sorted_keys_ptr + idx, mask=idx < T, other=0)  # int64
    starts = tl.load(starts_ptr + sk, mask=sk < NUM_EXPERTS, other=0)  # int64
    t = idx
    within = t - starts
    tl.store(within_ptr + t, within, mask=t < T)


# Kernel 5: Build mask_valid = within_pos < capacity. Inputs: within_ptr, capacity, Output: mask_valid_ptr (int8 0/1).
@triton.jit
def build_mask_valid_kernel(within_ptr, capacity, mask_ptr, T: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    w = tl.load(within_ptr + idx, mask=idx < T, other=0)
    m = w < capacity
    m_i8 = tl.where(m, 1, 0).to(tl.int8)
    tl.store(mask_ptr + idx, m_i8, mask=idx < T)


# Kernel 6: Scatter hidden_states into expert_inputs at positions where mask_valid is 1.
# Inputs: hidden_states [T], mask_valid [T], sorted_keys [T], token_ids [T], expert_inputs [NUM_EXPERTS*CAP*H].
@triton.jit
def scatter_inputs_kernel(hidden_ptr, mask_ptr, sorted_keys_ptr, token_ids_ptr, inputs_ptr,
                           T: tl.constexpr, BLOCK: tl.constexpr, NUM_EXPERTS: tl.constexpr, CAP: tl.constexpr, H: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    m = tl.load(mask_ptr + idx, mask=idx < T, other=0).to(tl.int32)  # 0/1
    valid = m > 0
    hs = tl.load(hidden_ptr + idx, mask=valid & (idx < T), other=0.0)  # bfloat16
    sk = tl.load(sorted_keys_ptr + idx, mask=valid & (idx < T), other=0)  # int64 expert id
    tok = tl.load(token_ids_ptr + idx, mask=valid & (idx < T), other=0)  # int64 token id
    # Compute destination pointer in inputs: base = sk * (CAP * H) + idx * H; only store for valid
    # For BLOCK vectors, we can't have per-lane dynamic offsets easily; instead process in chunks here:
    # We mask and store per idx
    base = sk * (CAP * H) + idx * H
    dest_ptr = inputs_ptr + base
    # Store only for valid lanes
    tl.store(dest_ptr, hs, mask=valid & (idx < T))


# Kernel 7: Batched matmul gate_out = A @ B, A: [M, K], B: [K, J], C: [M, J] (we write per (e,n) row).
# A is inputs [NUM_EXPERTS*CAP*H], B is gate weights [NUM_EXPERTS, H, J]. We launch grid over (e,n).
@triton.jit
def bmm_gate_kernel(A_ptr, B_ptr, C_ptr,
                    NUM_EXPERTS: tl.constexpr, CAP: tl.constexpr, H: tl.constexpr, J: tl.constexpr,
                    stride_A_m, stride_A_k,
                    stride_B_k, stride_B_j,
                    stride_C_e, stride_C_n, stride_C_j,
                    BLOCK_K: tl.constexpr, BLOCK_J: tl.constexpr):
    e = tl.program_id(0)
    n = tl.program_id(1)
    m = e * CAP + n
    base_c = e * (CAP * J) + n * J
    for j0 in range(0, J, BLOCK_J):
        j_idx = j0 + tl.arange(0, BLOCK_J)
        acc = tl.zeros((BLOCK_J,), dtype=tl.bfloat16)
        for k0 in range(0, H, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < H
            a_ptrs = A_ptr + m * H + k_idx * stride_A_k
            a = tl.load(a_ptrs, mask=mask_k, other=0.0)
            b_ptrs = B_ptr + e * stride_B_k + k_idx[:, None] * stride_B_k + j_idx[None, :] * stride_B_j
            b = tl.load(b_ptrs, mask=(mask_k[:, None]), other=0.0)
            acc += tl.sum(b * a[:, None], axis=0)
        c_ptrs = C_ptr + base_c + j_idx * stride_C_j
        tl.store(c_ptrs, acc, mask=(j_idx < J))


# Kernel 8: Same as gate, compute up_out with B = up weights.
@triton.jit
def bmm_up_kernel(A_ptr, B_ptr, C_ptr,
                  NUM_EXPERTS: tl.constexpr, CAP: tl.constexpr, H: tl.constexpr, J: tl.constexpr,
                  stride_A_m, stride_A_k,
                  stride_B_k, stride_B_j,
                  stride_C_e, stride_C_n, stride_C_j,
                  BLOCK_K: tl.constexpr, BLOCK_J: tl.constexpr):
    e = tl.program_id(0)
    n = tl.program_id(1)
    m = e * CAP + n
    base_c = e * (CAP * J) + n * J
    for j0 in range(0, J, BLOCK_J):
        j_idx = j0 + tl.arange(0, BLOCK_J)
        acc = tl.zeros((BLOCK_J,), dtype=tl.bfloat16)
        for k0 in range(0, H, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < H
            a_ptrs = A_ptr + m * H + k_idx * stride_A_k
            a = tl.load(a_ptrs, mask=mask_k, other=0.0)
            b_ptrs = B_ptr + e * stride_B_k + k_idx[:, None] * stride_B_k + j_idx[None, :] * stride_B_j
            b = tl.load(b_ptrs, mask=(mask_k[:, None]), other=0.0)
            acc += tl.sum(b * a[:, None], axis=0)
        c_ptrs = C_ptr + base_c + j_idx * stride_C_j
        tl.store(c_ptrs, acc, mask=(j_idx < J))


# Kernel 9: Elementwise SiLU and multiply: activated = SiLU(gate_out) * up_out
# Inputs: gate_out flattened [M_total * J], up_out flattened [M_total * J], Output: activated flattened.
@triton.jit
def activated_silu_mul_kernel(gate_ptr, up_ptr, activated_ptr, TOT: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < TOT
    g = tl.load(gate_ptr + idx, mask=mask, other=0.0)
    u = tl.load(up_ptr + idx, mask=mask, other=0.0)
    # SiLU(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-g))
    silu = g * s
    a = silu * u
    tl.store(activated_ptr + idx, a, mask=mask)


# Kernel 10: Batched matmul expert_outputs = activated @ down_weights
# Inputs: activated [NUM_EXPERTS * CAP * J], down_weights [NUM_EXPERTS, J, H], Output: expert_outputs_all [NUM_EXPERTS * CAP * H]
@triton.jit
def bmm_down_kernel(activated_ptr, down_ptr, outputs_ptr,
                    NUM_EXPERTS: tl.constexpr, CAP: tl.constexpr, J: tl.constexpr, H: tl.constexpr,
                    stride_Act_m, stride_Act_j,
                    stride_D_j, stride_D_h,
                    stride_Out_e, stride_Out_n, stride_Out_h,
                    BLOCK_J: tl.constexpr, BLOCK_H: tl.constexpr):
    e = tl.program_id(0)
    n = tl.program_id(1)
    m = e * CAP + n
    base_out = e * (CAP * H) + n * H
    for h0 in range(0, H, BLOCK_H):
        h_idx = h0 + tl.arange(0, BLOCK_H)
        acc = tl.zeros((BLOCK_H,), dtype=tl.bfloat16)
        for j0 in range(0, J, BLOCK_J):
            j_idx = j0 + tl.arange(0, BLOCK_J)
            a_ptrs = activated_ptr + m * J + j_idx * stride_Act_m
            a = tl.load(a_ptrs, mask=(j_idx < J), other=0.0)
            d_ptrs = down_ptr + e * stride_D_j + j_idx[:, None] * stride_D_j + h_idx[None, :] * stride_D_h
            d = tl.load(d_ptrs, mask=(j_idx[:, None] < J), other=0.0)
            acc += tl.sum(d * a[:, None], axis=0)
        out_ptrs = outputs_ptr + base_out + h_idx * stride_Out_h
        tl.store(out_ptrs, acc, mask=(h_idx < H))


# Kernel 11: Compute token_id = t // num_experts_per_tok for flattened index t and store to tok_ptr[t].
@triton.jit
def compute_token_ids_kernel(flattened_t_ptr, tok_ptr, num_tokens: tl.constexpr, num_experts_per_tok: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    t = idx  # flattened index 0..T-1
    mask = t < T
    token_id = t // num_experts_per_tok
    tl.store(tok_ptr + t, token_id, mask=mask)


# Kernel 12: Final weighted scatter-add into result [num_tokens, hidden_size].
# Inputs: v_exp [M_total], v_pos [M_total], v_tok [M_total], v_wt [M_total], expert_outputs_all [NUM_EXPERTS*CAP*H], result [num_tokens,H].
@triton.jit
def scatter_weighted_add_kernel(v_exp_ptr, v_pos_ptr, v_tok_ptr, v_wt_ptr, outputs_ptr, result_ptr,
                                NUM_EXPERTS: tl.constexpr, CAP: tl.constexpr, H: tl.constexpr,
                                BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < (NUM_EXPERTS * CAP)
    e = tl.load(v_exp_ptr + idx, mask=mask, other=0)  # int64
    n = tl.load(v_pos_ptr + idx, mask=mask, other=0)  # int32
    tok = tl.load(v_tok_ptr + idx, mask=mask, other=0)  # int64
    wt = tl.load(v_wt_ptr + idx, mask=mask, other=0.0)  # bfloat16
    # For each (e,n), compute row offset in outputs and scale by weight
    row_offset = e * (CAP * H) + n * H
    vals = tl.load(outputs_ptr + row_offset + tl.arange(0, H), mask=tl.arange(0, H) < H, other=0.0)
    vals = vals * wt
    # Scatter-add into result[tok, :]
    for h in range(0, H):
        ptr = result_ptr + tok * H + h
        old = tl.load(ptr)
        new = old + vals[h]
        tl.store(ptr, new)


class ModelNew(torch.nn.Module):
    def forward(self,
                hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure inputs are CUDA and contiguous
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda, "All tensors must be on CUDA."
        assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16."
        device = hidden_states.device

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts = expert_gate_weights.shape[0]
        intermediate_size = expert_gate_weights.shape[2]
        num_experts_per_tok = selected_experts.shape[1]

        # Flatten selections and routing weights
        T = num_tokens * num_experts_per_tok
        selected_experts_flat = selected_experts.reshape(-1).contiguous()
        routing_flat = routing_weights.reshape(-1).contiguous()
        token_ids_flat = torch.arange(T, device=device, dtype=torch.int64).contiguous()

        # Stable sort by selected_experts
        BLOCK = 4096  # must be >= T; can be next power of 2
        sorted_experts = torch.empty(T, device=device, dtype=torch.int64)
        sorted_weights = torch.empty(T, device=device, dtype=torch.bfloat16)
        sorted_token_ids = torch.empty(T, device=device, dtype=torch.int64)
        sort_by_keys_stable_kernel[(1,)](
            selected_experts_flat, routing_flat, token_ids_flat,
            sorted_experts, sorted_weights, sorted_token_ids,
            T=T, BLOCK=BLOCK
        )

        # Per-expert counts (bincount) and prefix sums (cumsum) in Triton
        counts = torch.zeros(num_experts, device=device, dtype=torch.int32)
        bincount_kernel[(1,)](sorted_experts, counts, T=T, BLOCK=BLOCK)
        # cumsum: starts[e] = sum_{i<e} counts[i]
        starts = torch.empty(num_experts, device=device, dtype=torch.int64)
        cumsum_starts_kernel[(num_experts,)](counts, starts, NUM_EXPERTS=num_experts)

        # Compute within_pos and capacity
        capacity = int(math.ceil(1.25 * T / num_experts))
        capacity = max(capacity, 1)
        within_pos = torch.empty(T, device=device, dtype=torch.int32)
        within_pos_kernel[(1,)](sorted_experts, starts, within_pos, T=T, BLOCK=BLOCK)

        # Build mask_valid = within_pos < capacity
        mask_valid = torch.empty(T, device=device, dtype=torch.int8)
        build_mask_valid_kernel[(1,)](within_pos, capacity, mask_valid, T=T, BLOCK=BLOCK)

        # Scatter hidden states into expert_inputs for valid positions
        expert_inputs = torch.empty(num_experts * capacity * hidden_size, device=device, dtype=torch.bfloat16)
        scatter_inputs_kernel[(1,)](
            hidden_states.reshape(-1), mask_valid, sorted_experts, sorted_token_ids, expert_inputs,
            T=T, BLOCK=BLOCK, NUM_EXPERTS=num_experts, CAP=capacity, H=hidden_size
        )

        # Compute gate_out, up_out, activated, and expert_outputs_all using batched matmuls
        # Gate bmm: [e,n,H] @ [e,H,J] -> [e,n,J]
        gate_out = torch.empty(num_experts * capacity * intermediate_size, device=device, dtype=torch.bfloat16)
        bmm_gate_kernel[(num_experts, capacity)](
            expert_inputs, expert_gate_weights, gate_out,
            NUM_EXPERTS=num_experts, CAP=capacity, H=hidden_size, J=intermediate_size,
            stride_A_m=hidden_size, stride_A_k=1,
            stride_B_k=hidden_size, stride_B_j=intermediate_size,
            stride_C_e=num_experts, stride_C_n=capacity, stride_C_j=intermediate_size,
            BLOCK_K=64, BLOCK_J=64
        )

        up_out = torch.empty(num_experts * capacity * intermediate_size, device=device, dtype=torch.bfloat16)
        bmm_up_kernel[(num_experts, capacity)](
            expert_inputs, expert_up_weights, up_out,
            NUM_EXPERTS=num_experts, CAP=capacity, H=hidden_size, J=intermediate_size,
            stride_A_m=hidden_size, stride_A_k=1,
            stride_B_k=hidden_size, stride_B_j=intermediate_size,
            stride_C_e=num_experts, stride_C_n=capacity, stride_C_j=intermediate_size,
            BLOCK_K=64, BLOCK_J=64
        )

        activated = torch.empty(num_experts * capacity * intermediate_size, device=device, dtype=torch.bfloat16)
        activated_silu_mul_kernel[(1,)](
            gate_out, up_out, activated,
            TOT=num_experts * capacity * intermediate_size,
            BLOCK=1024
        )

        expert_outputs_all = torch.empty(num_experts * capacity * hidden_size, device=device, dtype=torch.bfloat16)
        bmm_down_kernel[(num_experts, capacity)](
            activated, expert_down_weights, expert_outputs_all,
            NUM_EXPERTS=num_experts, CAP=capacity, J=intermediate_size, H=hidden_size,
            stride_Act_m=intermediate_size, stride_Act_j=1,
            stride_D_j=intermediate_size, stride_D_h=hidden_size,
            stride_Out_e=num_experts, stride_Out_n=capacity, stride_Out_h=hidden_size,
            BLOCK_J=64, BLOCK_H=64
        )

        # Final weighted scatter-add into result
        result = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

        # Prepare v_exp, v_pos, v_wt (from sorted arrays)
        v_exp = sorted_experts
        v_pos = within_pos
        v_wt = sorted_weights

        # Launch scatter_weighted_add_kernel: grid = (ceil_div(NUM_EXPERTS*CAP, BLOCK),)
        grid = (triton.cdiv(num_experts * capacity, 1024),)
        scatter_weighted_add_kernel[grid](v_exp, v_pos, sorted_token_ids, v_wt, expert_outputs_all, result,
                                          NUM_EXPERTS=num_experts, CAP=capacity, H=hidden_size,
                                          BLOCK=1024)

        return result


def run(*args):
    return ModelNew()(*args)
