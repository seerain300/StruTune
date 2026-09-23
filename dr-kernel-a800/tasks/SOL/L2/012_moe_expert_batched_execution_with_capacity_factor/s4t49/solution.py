import math
import torch
import triton
import triton.language as tl


# ----------------------------
# Triton kernels for preprocessing and compute
# ----------------------------

@triton.jit
def triton_sort_experts_flat(experts_ptr,       # *int64, flattened [num_tokens * num_experts_per_tok]
                              sorted_ptr,       # *int64, flattened [num_tokens * num_experts_per_tok]
                              sorted_ids_ptr,   # *int32, flattened [num_tokens * num_experts_per_tok]
                              num_rows: tl.constexpr,      # num_tokens
                              col_count: tl.constexpr,     # num_experts_per_tok
                              BLOCK: tl.constexpr):        # bitonic block size (must be >= col_count)
    pid = tl.program_id(axis=0)
    base = pid * col_count
    idx = base + tl.arange(0, BLOCK)  # [BLOCK], may have masked lanes
    mask = idx < (base + col_count)

    # Load original expert indices
    orig = tl.load(experts_ptr + idx, mask=mask, other=0).to(tl.int64)
    # Positions (stable: use original positions as tie-breaker)
    pos = idx - base  # [BLOCK], int32
    pos = pos.to(tl.int32)

    # Bitonic sort (stable by original pos): ascending by value
    k = 2
    while k <= BLOCK:
        j = k // 2
        while j >= 1:
            partner = idx ^ j
            # Create sub-tuples (value, pos) for pair compare
            orig_i = orig
            pos_i = pos
            orig_p = tl.load(experts_ptr + partner, mask=partner < (base + col_count), other=0).to(tl.int64)
            pos_p = partner - base
            pos_p = pos_p.to(tl.int32)

            # Determine direction: for ascending, swap if orig_i > orig_p or equal and pos_i > pos_p
            greater = orig_i > orig_p
            equal = orig_i == orig_p
            swap = greater | (equal & (pos_i > pos_p))

            # Compute new values for this lane
            new_orig = tl.where(swap, orig_p, orig_i)
            new_pos = tl.where(swap, pos_p, pos_i)

            # Assign min/max back to i and partner positions
            is_lower = (idx & j) == 0
            # For lower half: take min; for upper half: take max
            take_min = is_lower
            take_max = not take_min

            # Only one side writes per pair; use XOR to ensure write once per pair
            write_mask = (idx < partner)  # only the lower index in the pair writes

            # min/max computed by both sides
            min_orig = tl.where(new_orig < orig_i, new_orig, orig_i)
            max_orig = tl.where(new_orig > orig_i, new_orig, orig_i)
            min_pos = tl.where(new_pos < pos_i, new_pos, pos_i)
            max_pos = tl.where(new_pos > pos_i, new_pos, pos_i)

            # Select for this lane based on take_min/max
            new_orig = tl.where(take_min, min_orig, max_orig)
            new_pos = tl.where(take_min, min_pos, max_pos)

            # Only write for lower index
            tl.store(orig, new_orig, mask=write_mask & mask)
            # Also store pos (not used further, but keep consistent)
            tl.store(pos, new_pos, mask=write_mask & mask)

            j //= 2
        k *= 2

    # After sorting, pos now contains stable positions within the token's vector.
    # Store sorted values and their indices.
    sorted_vals = orig
    sorted_ids = pos  # original positions within the token's [num_experts_per_tok] vector
    # Write to output
    tl.store(sorted_ptr + idx, sorted_vals, mask=mask)
    tl.store(sorted_ids_ptr + idx, sorted_ids, mask=mask)


@triton.jit
def triton_bincount(counts_ptr,           # *int64, output [num_experts]
                     sorted_ids_ptr,       # *int32, flattened [num_tokens * num_experts_per_tok]
                     num_rows: tl.constexpr,      # num_tokens
                     col_count: tl.constexpr,     # num_experts_per_tok
                     E: tl.constexpr):            # num_experts
    # Each program handles one expert id
    pid = tl.program_id(axis=0)
    expert_id = pid
    acc = tl.zeros((), dtype=tl.int64)

    # Iterate all tokens
    for t in range(0, num_rows):
        base = t * col_count
        for k in range(0, col_count):
            idx = base + k
            sid = tl.load(sorted_ids_ptr + idx)  # int32
            if sid == expert_id:
                acc += 1
    tl.store(counts_ptr + expert_id, acc)


@triton.jit
def triton_cumsum(starts_ptr,            # *int64, output [num_experts]
                  counts_ptr,             # *int64, input [num_experts]
                  E: tl.constexpr):
    # Prefix sum: starts[i] = sum_{j=0..i} counts[j]
    # Launch one program per expert
    pid = tl.program_id(axis=0)
    i = pid
    sum_val = tl.zeros((), dtype=tl.int64)
    for j in range(0, i + 1):
        cnt = tl.load(counts_ptr + j)
        sum_val += cnt
    tl.store(starts_ptr + i, sum_val)


@triton.jit
def triton_row_matmul(C_ptr,        # *bfloat16, flattened output vector [H]
                      A_ptr,         # *bfloat16, input row [H]
                      B_ptr,         # *bfloat16, matrix [H, H], row-major
                      H: tl.constexpr,
                      BLOCK: tl.constexpr):
    # Compute C = A_row @ B
    out_base = tl.program_id(axis=0) * H
    acc = tl.zeros([H], dtype=tl.float32)

    # Iterate over K in chunks of BLOCK
    for k in range(0, H, BLOCK):
        k_offsets = k + tl.arange(0, BLOCK)  # [BLOCK]
        k_mask = k_offsets < H

        # Load A_row[k:k+BLOCK]
        a = tl.load(A_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK]

        # Load corresponding columns of B: shape [BLOCK, H]
        b = tl.zeros([BLOCK, H], dtype=tl.float32)
        for jj in range(0, H, BLOCK):
            j_offsets = jj + tl.arange(0, BLOCK)  # [BLOCK]
            j_mask = j_offsets < H
            b_part = tl.load(B_ptr + k_offsets[:, None] * H + j_offsets[None, :],
                             mask=k_mask[:, None] & j_mask[None, :],
                             other=0.0).to(tl.float32)
            b += b_part

        # Accumulate dot products
        acc += tl.sum(a[:, None] * b, axis=0)

    # Store result
    tl.store(C_ptr + out_base, acc.to(tl.bfloat16))


@triton.jit
def triton_silu(x_ptr,          # *bfloat16, [H]
                out_ptr,         # *bfloat16, [H]
                H: tl.constexpr):
    for i in range(0, H):
        x = tl.load(x_ptr + i).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(out_ptr + i, y.to(tl.bfloat16))


@triton.jit
def triton_mul(a_ptr, b_ptr, out_ptr, H: tl.constexpr):
    for i in range(0, H):
        a = tl.load(a_ptr + i).to(tl.float32)
        b = tl.load(b_ptr + i).to(tl.float32)
        tl.store(out_ptr + i, (a * b).to(tl.bfloat16))


@triton.jit
def triton_atomic_weighted_add(weight_ptr,  # *bfloat16, [N] = num_tokens * hidden_size
                               vec_ptr,     # *bfloat16, [N]
                               out_ptr,     # *bfloat16, [num_tokens * hidden_size]
                               N: tl.constexpr):
    for i in range(0, N):
        w = tl.load(weight_ptr + i).to(tl.float32)
        v = tl.load(vec_ptr + i).to(tl.float32)
        o = tl.load(out_ptr + i)
        o += w * v
        tl.store(out_ptr + i, o)


# ----------------------------
# ModelNew: Triton-only forward
# ----------------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]
        device = hidden_states.device

        # Ensure dtype bfloat16 for inputs to kernels
        # selected_experts: int64, routing_weights: bfloat16, hidden and expert weights: bfloat16
        # Flatten selected_experts for sort
        flat_experts = selected_experts.reshape(-1).to(torch.int64).contiguous()
        N_pairs = num_tokens * num_experts_per_tok
        BLOCK = num_experts_per_tok  # bitonic sort block size equals K

        # Allocate sorted and sorted_ids
        sorted_experts = torch.empty(N_pairs, dtype=torch.int64, device=device)
        sorted_ids = torch.empty(N_pairs, dtype=torch.int32, device=device)

        # Launch Triton sort (bitonic per row-vector)
        grid_sort = (num_tokens,)
        triton_sort_experts_flat[grid_sort](
            flat_experts, sorted_experts, sorted_ids, num_rows=num_tokens, col_count=num_experts_per_tok, BLOCK=BLOCK
        )

        # Bincount per expert using Triton
        counts = torch.empty(num_experts, dtype=torch.int64, device=device)
        grid_bc = (num_experts,)
        triton_bincount[grid_bc](
            counts, sorted_ids, num_rows=num_tokens, col_count=num_experts_per_tok, E=num_experts
        )

        # Cumsum starts using Triton
        starts = torch.empty(num_experts, dtype=torch.int64, device=device)
        grid_cs = (num_experts,)
        triton_cumsum[grid_cs](
            starts, counts, E=num_experts
        )

        # Compute capacity (same as original) in host; pass to Triton as int32
        capacity = max(int((num_tokens * num_experts_per_tok) * 1.25 // num_experts), 1)
        cap_i32 = torch.tensor(capacity, dtype=torch.int32, device=device)

        # Output result: [num_tokens, hidden_size], bfloat16, initialized to zero
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

        # Iterate over all token-expert pairs; launch Triton kernels for compute and atomic add
        for t in range(0, num_tokens):
            # Base pointers for this token
            base_hs = hidden_states[t]  # [hidden_size], bfloat16
            for j in range(0, num_experts_per_tok):
                # Find expert id from sorted_experts
                idx = t * num_experts_per_tok + j
                expert_id = int(tl.load(sorted_experts + idx).item())  # expert_id for this token's j-th selected

                # Load routing weight for this token-expert
                weight = float(tl.load(routing_weights[t, j].data_ptr) if hasattr(routing_weights[t, j], 'data_ptr') else tl.load(routing_weights[t, j]).to(tl.float32))
                # Note: Triton does not allow accessing .data_ptr of torch.Tensor; instead pass scalar weight directly.
                # We will pass weight as a tensor to the atomic kernel by computing weight * expert_out per pair.
                # Compute hidden_input = base_hs as Triton input (bf16 vector)
                hidden_vec = base_hs.contiguous()

                # Gate and Up GEMMs: compute gate_out and up_out
                # gate_out = hidden_vec @ expert_gate_weights[expert_id]
                gate_out = torch.empty(moe_intermediate_size, dtype=torch.bfloat16, device=device)
                triton_row_matmul[(1,)](
                    gate_out, hidden_vec, expert_gate_weights[expert_id].contiguous(), H=moe_intermediate_size, BLOCK=64, num_warps=2
                )

                # up_out = hidden_vec @ expert_up_weights[expert_id]
                up_out = torch.empty(moe_intermediate_size, dtype=torch.bfloat16, device=device)
                triton_row_matmul[(1,)](
                    up_out, hidden_vec, expert_up_weights[expert_id].contiguous(), H=moe_intermediate_size, BLOCK=64, num_warps=2
                )

                # SiLU on gate_out
                gate_silu = torch.empty(moe_intermediate_size, dtype=torch.bfloat16, device=device)
                triton_silu[(moe_intermediate_size,)](gate_out, gate_silu, H=moe_intermediate_size)

                # Elementwise multiply: activated = SiLU(gate_out) * up_out
                activated = torch.empty(moe_intermediate_size, dtype=torch.bfloat16, device=device)
                triton_mul[(moe_intermediate_size,)](
                    gate_silu, up_out, activated, H=moe_intermediate_size
                )

                # expert_outputs = activated @ expert_down_weights[expert_id]
                expert_out = torch.empty(hidden_size, dtype=torch.bfloat16, device=device)
                triton_row_matmul[(1,)](
                    expert_out, activated, expert_down_weights[expert_id].contiguous(), H=hidden_size, BLOCK=64, num_warps=2
                )

                # Atomic add: weight * expert_out into result[t, :]
                # Prepare weight vector: replicate expert_out length (or directly use scalar)
                weight_vec = torch.full((hidden_size,), weight, dtype=torch.bfloat16, device=device)
                weight_expert_out = torch.empty(hidden_size, dtype=torch.bfloat16, device=device)
                triton_mul[(hidden_size,)](
                    weight_vec, expert_out, weight_expert_out, H=hidden_size
                )
                # Atomic add into flattened result
                triton_atomic_weighted_add[(num_tokens * hidden_size,)](
                    weight_expert_out, torch.zeros(num_tokens * hidden_size, dtype=torch.bfloat16, device=device),
                    result.view(-1), N=num_tokens * hidden_size
                )

        return result


def run(*args):
    return ModelNew()(*args)
