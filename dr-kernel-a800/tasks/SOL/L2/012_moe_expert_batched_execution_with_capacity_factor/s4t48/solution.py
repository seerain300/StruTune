import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_row_matmul(C_ptr,        # *bfloat16, flattened result buffer
                      A_ptr,         # *bfloat16, row vector [H]
                      B_ptr,         # *bfloat16, matrix [H, H] row-major
                      H: tl.constexpr,
                      BLOCK: tl.constexpr):
    # One program handles one row-block and writes its C[H]
    # A_ptr is a row vector of length H, B_ptr is HxH, C_ptr is contiguous of length H
    # We implement: C = A @ B (row A times matrix B => vector C)
    out = tl.zeros([H], dtype=tl.float32)
    for k in range(0, H, BLOCK):
        k_offsets = k + tl.arange(0, BLOCK)
        a_k = tl.load(A_ptr + k_offsets, mask=k_offsets < H, other=0.0).to(tl.float32)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for kk in range(0, H, BLOCK):
            kk_offsets = kk + tl.arange(0, BLOCK)
            b_tile = tl.load(B_ptr + kk_offsets[:, None] * H + k_offsets[None, :],
                             mask=(kk_offsets[:, None] < H) & (k_offsets[None, :] < H),
                             other=0.0).to(tl.float32)
            acc += tl.sum(b_tile * a_k[None, :], axis=1)
        out += acc
    # Store out to C_ptr
    for i in range(0, H):
        tl.store(C_ptr + i, out[i].to(tl.bfloat16))


@triton.jit
def triton_silu(x_ptr, out_ptr, H: tl.constexpr):
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
def triton_atomic_add_weighted(weight_ptr,  # *bfloat16, vector to add (length N)
                               out_ptr,     # *bfloat16, flattened output (length N)
                               N: tl.constexpr):
    # Atomic add weight[i] into out[i] for i in [0, N)
    for i in range(0, N):
        w = tl.load(weight_ptr + i).to(tl.float32)
        out = tl.load(out_ptr + i)
        out += w
        tl.store(out_ptr + i, out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_w_H, m = expert_gate_weights.shape  # gate_w_H == hidden_size, m == moe_intermediate_size
        # Preprocess exactly as original (use torch for metadata, no torch compute on results)
        # 1) Flatten selected_experts: (num_tokens * K,)
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        flat_token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(selected_experts.shape[1])

        # 2) Stable sort by selected_experts to match original ordering
        sorted_experts, sorted_indices = torch.sort(flat_experts, stable=True)
        sorted_weights = flat_weights[sorted_indices]
        sorted_token_ids = flat_token_ids[sorted_indices]

        # 3) Bincount per expert to compute capacity
        counts = torch.bincount(sorted_experts, minlength=num_experts)
        starts = torch.zeros(num_experts, dtype=torch.long, device=hidden_states.device)
        starts[1:] = counts[:-1].cumsum(0)
        # capacity = ceil(1.25 * average occupancy)
        # average occupancy = (num_tokens * K) / num_experts
        avg = (num_tokens * selected_experts.shape[1]) / num_experts
        capacity = int(math.ceil(1.25 * avg))
        capacity = max(capacity, 1)

        # 4) Compute within-expert positions
        idx = torch.arange(sorted_experts.shape[0], device=hidden_states.device)
        within_pos = idx - starts[sorted_experts]

        # 5) Apply capacity mask
        valid = within_pos < capacity
        v_exp = sorted_experts[valid]
        v_pos = within_pos[valid].to(v_exp.dtype)  # positions in [0, capacity)
        v_tok = sorted_token_ids[valid]
        v_wt = sorted_weights[valid]

        # 6) Build padded expert inputs: expert_inputs[exp, pos, :] = hidden_states[token]
        # We will directly access hidden_states[v_tok] in forward and call Triton per pair
        # Prepare output result
        result = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

        # Iterate all valid pairs and launch Triton kernels for compute + atomic add
        # Note: We must ensure we only aggregate valid pairs (per original algorithm).
        # However, Triton atomic_add will allow duplicates; to match original exactly, we rely on valid mask above.
        # For each pair (v_exp, v_pos, v_tok, v_wt), compute and add.
        for k in range(0, v_exp.numel()):
            exp = int(v_exp[k].item())  # index of expert
            pos = int(v_pos[k].item())  # position within this expert's group (always < capacity)
            tok = int(v_tok[k].item())  # token index
            w = float(v_wt[k].item())   # routing weight

            # hidden_input for this token
            hidden_input = hidden_states[tok].to(torch.bfloat16).contiguous()  # length H
            # gate weight and up weight for this expert: shape [H, m]
            gate_w = expert_gate_weights[exp].contiguous()  # [H, m]
            up_w = expert_up_weights[exp].contiguous()      # [H, m]
            down_w = expert_down_weights[exp].contiguous()  # [m, H]

            # Compute gate_out = hidden_input @ gate_w  -> [H]
            # Compute up_out = hidden_input @ up_w      -> [H]
            # Triton row_matmul for gate_out
            gate_out = torch.empty(hidden_size, dtype=torch.bfloat16, device=hidden_states.device)
            triton_row_matmul[(1,)](
                gate_out, hidden_input, gate_w, H=hidden_size, BLOCK=64, num_warps=4
            )

            # Triton row_matmul for up_out
            up_out = torch.empty(hidden_size, dtype=torch.bfloat16, device=hidden_states.device)
            triton_row_matmul[(1,)](
                up_out, hidden_input, up_w, H=hidden_size, BLOCK=64, num_warps=4
            )

            # Elementwise SiLU on gate_out
            gate_silu = torch.empty(hidden_size, dtype=torch.bfloat16, device=hidden_states.device)
            triton_silu[(hidden_size,)](gate_out, gate_silu, H=hidden_size)

            # Elementwise multiply: activated = SiLU(gate_out) * up_out
            activated = torch.empty(hidden_size, dtype=torch.bfloat16, device=hidden_states.device)
            triton_mul[(hidden_size,)](gate_silu, up_out, activated, H=hidden_size)

            # Compute expert_outputs = activated @ down_w -> scalar vector length H
            expert_out = torch.empty(hidden_size, dtype=torch.bfloat16, device=hidden_states.device)
            triton_row_matmul[(1,)](
                expert_out, activated, down_w, H=hidden_size, BLOCK=64, num_warps=4
            )

            # Atomic add weight * expert_out into result[tok, :]
            # result[tok, :] += w * expert_out
            # Flatten result for atomic adds
            triton_atomic_add_weighted[(num_tokens * hidden_size,)](
                (w * expert_out).to(torch.bfloat16), result.view(-1), N=num_tokens * hidden_size
            )

        return result


def run(*args):
    return ModelNew()(*args)
