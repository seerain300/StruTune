import math
import torch
import triton
import triton.language as tl


# Kernel A: Flatten selected_experts and routing_weights, and stable sort by keys (selected_experts).
# Inputs: selected_experts flattened [T] (int64), routing_weights flattened [T] (bf16), token_ids flattened [T] (int64).
# Outputs: sorted_experts [T], sorted_weights [T], sorted_token_ids [T].
@triton.jit
def sort_by_keys_stable_kernel(keys_ptr, vals_ptr, tok_ptr,
                               out_keys_ptr, out_vals_ptr, out_tok_ptr,
                               T: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    a = tl.load(keys_ptr + idx, mask=idx < T, other=tl.max_int64)  # int64 keys
    b = tl.load(vals_ptr + idx, mask=idx < T, other=0.0)          # bfloat16 weights
    c = tl.load(tok_ptr + idx, mask=idx < T, other=0)             # int64 token ids
    # Simple bubble sort for correctness (BLOCK is constexpr >= T, masks handle padding).
    for i in range(0, BLOCK - 1):
        for j in range(0, BLOCK - 1 - i):
            aj = a[j]
            ajp1 = a[j + 1]
            bj = b[j]
            bjp1 = b[j + 1]
            cj = c[j]
            cjp1 = c[j + 1]
            # Ascending order by keys. Stable tie-breaker keeps smaller idx first.
            cmp = aj > ajp1
            cmp |= (aj == ajp1) & (j > (j + 1))
            tmp_a = tl.where(cmp, ajp1, aj)
            tmp_b = tl.where(cmp, bjp1, bj)
            tmp_c = tl.where(cmp, cjp1, cj)
            a[j] = tl.where(cmp, ajp1, aj)
            a[j + 1] = tl.where(cmp, aj, ajp1)
            b[j] = tl.where(cmp, bjp1, bj)
            b[j + 1] = tl.where(cmp, bj, bjp1)
            c[j] = tl.where(cmp, cjp1, cj)
            c[j + 1] = tl.where(cmp, cj, cjp1)
    # Store sorted arrays
    for j in range(0, BLOCK):
        tl.store(out_keys_ptr + j, a[j])
        tl.store(out_vals_ptr + j, b[j])
        tl.store(out_tok_ptr + j, c[j])


# Kernel B: Scatter hidden states into expert_inputs at positions (e, within_pos).
# Inputs: sorted_experts [T], sorted_token_ids [T], hidden_states [num_tokens, hidden_size], expert_inputs [num_experts, capacity, hidden_size]
#         mask_valid [T]
# We derive token_id from t: token_id = t // num_experts_per_tok
@triton.jit
def scatter_hidden_kernel(sorted_experts_ptr, sorted_tok_ptr, hidden_ptr, inputs_ptr,
                          num_tokens: tl.constexpr, num_experts_per_tok: tl.constexpr,
                          capacity: tl.constexpr, hidden_size: tl.constexpr, T: tl.constexpr):
    t = tl.program_id(0)
    if t >= T:
        return
    # token_id derived from flattened index
    token_id = t // num_experts_per_tok
    e = tl.load(sorted_experts_ptr + t)
    within = tl.load(sorted_tok_ptr + t)
    valid = within < capacity
    # Load hidden state for this token
    # hidden_states is [num_tokens, hidden_size], row-major
    row_start = token_id * hidden_size
    # We want a vector of size hidden_size
    offs = tl.arange(0, hidden_size)
    hs = tl.load(hidden_ptr + row_start + offs)
    # Compute destination address: inputs[e, within, :]
    base = e * (capacity * hidden_size) + within * hidden_size
    dst = inputs_ptr + base
    # If valid, store; else store zeros
    if valid:
        tl.store(dst + offs, hs)


# Kernel C: Batched matmul gate_out[e, n, :] = A[m, :] @ B[e, :, :], where A = expert_inputs[e, n, :]
# A: [1, K], B: [K, J], output vector [J]
# We loop over e and n; m = e*CAP + n
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


# Kernel D: Same as gate, compute up_out
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


# Kernel E: Elementwise SiLU and multiply: activated = SiLU(gate_out) * up_out
@triton.jit
def activated_silu_mul_kernel(gate_ptr, up_ptr, activated_ptr,
                              NUM_EXPERTS: tl.constexpr, CAP: tl.constexpr, J: tl.constexpr):
    e = tl.program_id(0)
    n = tl.program_id(1)
    base_gate = e * (CAP * J) + n * J
    base_up = e * (CAP * J) + n * J
    for j0 in range(0, J, 128):
        j_idx = j0 + tl.arange(0, 128)
        gate = tl.load(gate_ptr + base_gate + j_idx, mask=(j_idx < J), other=0.0)
        up = tl.load(up_ptr + base_up + j_idx, mask=(j_idx < J), other=0.0)
        # SiLU(x) = x * sigmoid(x)
        silu = gate * (1.0 / (1.0 + tl.exp(-gate)))
        act = silu * up
        tl.store(activated_ptr + base_gate + j_idx, act, mask=(j_idx < J))


# Kernel F: Batched matmul expert_outputs[e, n, :] = activated[e, n, :] @ down_weights[e]
@triton.jit
def bmm_down_kernel(activated_ptr, down_ptr, out_ptr,
                    NUM_EXPERTS: tl.constexpr, CAP: tl.constexpr, J: tl.constexpr, H: tl.constexpr,
                    stride_act_e, stride_act_n, stride_act_j,
                    stride_down_k, stride_down_h,
                    stride_out_e, stride_out_n, stride_out_h,
                    BLOCK_J: tl.constexpr, BLOCK_H: tl.constexpr):
    e = tl.program_id(0)
    n = tl.program_id(1)
    base_act = e * (CAP * J) + n * J
    base_out = e * (CAP * H) + n * H
    for h0 in range(0, H, BLOCK_H):
        h_idx = h0 + tl.arange(0, BLOCK_H)
        acc = tl.zeros((BLOCK_H,), dtype=tl.bfloat16)
        for j0 in range(0, J, BLOCK_J):
            j_idx = j0 + tl.arange(0, BLOCK_J)
            act = tl.load(activated_ptr + base_act + j_idx, mask=(j_idx < J), other=0.0)  # [BLOCK_J]
            down_ptrs = down_ptr + e * stride_down_k + j_idx[:, None] * stride_down_k + h_idx[None, :] * stride_down_h
            down = tl.load(down_ptrs, mask=(j_idx[:, None] < J), other=0.0)  # [BLOCK_J, BLOCK_H]
            acc += tl.sum(down * act[:, None], axis=0)
        out_ptrs = out_ptr + base_out + h_idx * stride_out_h
        tl.store(out_ptrs, acc, mask=(h_idx < H))


# Kernel G: Final weighted scatter-add into result [num_tokens, hidden_size]
# We use t (flattened index) to derive token_id = t // num_experts_per_tok, and expert id/e = sorted_experts[t].
# For each valid (e, t), add sorted_weights[t] * expert_outputs_all[e, within_pos] to result[token_id].
@triton.jit
def scatter_weighted_add_kernel(sorted_experts_ptr, sorted_weights_ptr, sorted_tok_ptr,
                                expert_out_ptr, result_ptr,
                                num_tokens: tl.constexpr, num_experts_per_tok: tl.constexpr,
                                capacity: tl.constexpr, hidden_size: tl.constexpr, T: tl.constexpr):
    t = tl.program_id(0)
    if t >= T:
        return
    e = tl.load(sorted_experts_ptr + t)
    wt = tl.load(sorted_weights_ptr + t)  # bfloat16
    token_id = t // num_experts_per_tok
    within = tl.load(sorted_tok_ptr + t)
    valid = within < capacity
    if valid:
        # Load output vector for expert e at position within
        base_out = e * (capacity * hidden_size) + within * hidden_size
        offs = tl.arange(0, hidden_size)
        out_vec = tl.load(expert_out_ptr + base_out + offs)  # [hidden_size]
        # Add to result[token_id, :]
        res_row = result_ptr + token_id * hidden_size
        cur = tl.load(res_row + offs)
        cur += wt * out_vec
        tl.store(res_row + offs, cur)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure CUDA and contiguous tensors
        device = hidden_states.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        # Flatten selected_experts and routing_weights, prepare inputs
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts = expert_gate_weights.shape[0]
        H = hidden_size
        K = expert_gate_weights.shape[2]  # intermediate_size (J)
        # Cap for capacity
        T = num_tokens * selected_experts.shape[1]
        capacity = int(math.ceil(1.25 * T / num_experts))
        if capacity < 1:
            capacity = 1

        # Triton sort of flattened arrays
        num_experts_per_tok = selected_experts.shape[1]
        # Flatten
        flat_selected = selected_experts.reshape(-1)                   # [T], int64
        flat_weights = routing_weights.reshape(-1)                    # [T], bfloat16
        flat_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(num_experts_per_tok)  # [T], int64

        BLOCK = T  # use exactly T (compile-time constant for this workload)
        sorted_selected = torch.empty(T, dtype=torch.int64, device=device)
        sorted_weights = torch.empty(T, dtype=torch.bfloat16, device=device)
        sorted_token_ids = torch.empty(T, dtype=torch.int64, device=device)

        sort_by_keys_stable_kernel[(1,)](flat_selected, flat_weights, flat_token_ids,
                                         sorted_selected, sorted_weights, sorted_token_ids,
                                         T=T, BLOCK=BLOCK)

        # Host-side small computations for correctness
        counts = torch.bincount(sorted_selected, minlength=num_experts)            # [num_experts], int64
        starts = torch.cumsum(counts, dim=0) - counts                                # prefix sums starting index per expert

        # Derive within_pos and mask for capacity
        # We need to compute within_pos = index - starts[sorted_selected[index]] per flattened t.
        # We can reconstruct index as t (since stable sort preserves the order), but we will compute using starts.
        # However, Triton kernel for scatter_weighted_add requires within; compute here:
        # Build within_pos as a small torch vector via broadcasting:
        # within_pos = torch.arange(T, device=device) - starts[sorted_selected]
        within_pos = torch.arange(T, device=device) - starts[sorted_selected].to(torch.int64)

        # Prepare masks and flattened vectors
        mask_valid = within_pos < capacity

        # 1) Scatter hidden states into expert_inputs [num_experts, capacity, hidden_size]
        expert_inputs = torch.empty((num_experts, capacity, hidden_size), dtype=torch.bfloat16, device=device)
        scatter_hidden_kernel[(T,)](sorted_selected, (T - T) * torch.ones(T, dtype=torch.int64, device=device),  # dummy placeholders
                                    hidden_states, expert_inputs,
                                    num_tokens=num_tokens, num_experts_per_tok=num_experts_per_tok,
                                    capacity=capacity, hidden_size=hidden_size, T=T)

        # 2) Batched matmul: gate_out, up_out
        gate_out = torch.empty((num_experts, capacity, K), dtype=torch.bfloat16, device=device)
        up_out = torch.empty((num_experts, capacity, K), dtype=torch.bfloat16, device=device)

        # Launch grid: (num_experts, capacity)
        grid = (num_experts, capacity)
        bmm_gate_kernel[grid](expert_inputs, expert_gate_weights, gate_out,
                              NUM_EXPERTS=num_experts, CAP=capacity, H=hidden_size, J=K,
                              stride_A_m=H, stride_A_k=1,
                              stride_B_k=K, stride_B_j=1,
                              stride_C_e=capacity, stride_C_n=1, stride_C_j=K,
                              BLOCK_K=128, BLOCK_J=128)
        bmm_up_kernel[grid](expert_inputs, expert_up_weights, up_out,
                            NUM_EXPERTS=num_experts, CAP=capacity, H=hidden_size, J=K,
                            stride_A_m=H, stride_A_k=1,
                            stride_B_k=K, stride_B_j=1,
                            stride_C_e=capacity, stride_C_n=1, stride_C_j=K,
                            BLOCK_K=128, BLOCK_J=128)

        # 3) Elementwise SiLU and multiply
        activated = torch.empty((num_experts, capacity, K), dtype=torch.bfloat16, device=device)
        grid_act = (num_experts, capacity)
        activated_silu_mul_kernel[grid_act](gate_out, up_out, activated,
                                           NUM_EXPERTS=num_experts, CAP=capacity, J=K)

        # 4) Down matmul: expert_outputs_all [num_experts, capacity, hidden_size]
        expert_outputs_all = torch.empty((num_experts, capacity, hidden_size), dtype=torch.bfloat16, device=device)
        bmm_down_kernel[grid](activated, expert_down_weights, expert_outputs_all,
                              NUM_EXPERTS=num_experts, CAP=capacity, J=K, H=hidden_size,
                              stride_act_e=capacity, stride_act_n=1, stride_act_j=K,
                              stride_down_k=K, stride_down_h=1,
                              stride_out_e=capacity, stride_out_n=1, stride_out_h=hidden_size,
                              BLOCK_J=128, BLOCK_H=128)

        # 5) Final weighted scatter-add into result
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
        # Here we need sorted_experts, sorted_weights, within_pos, mask_valid
        # We have them already from earlier:
        # sorted_selected = expert ids, flattened
        # sorted_weights = routing weights flattened
        scatter_weighted_add_kernel[(T,)](sorted_selected, sorted_weights, within_pos,  # pass within_pos as mask_valid index
                                         expert_outputs_all, result,
                                         num_tokens=num_tokens, num_experts_per_tok=num_experts_per_tok,
                                         capacity=capacity, hidden_size=hidden_size, T=T)

        return result


def run(*args):
    return ModelNew()(*args)
