import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_beta_kernel(
    A_log_ptr,        # [H] float32
    a_ptr,            # [B,H] float32
    dt_bias_ptr,      # [H] float32
    b_ptr,            # [B,H] float32
    g_out_ptr,        # [B,H] float32
    beta_out_ptr,     # [B,H] float32
    H: tl.constexpr,  # number of heads
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b_idx * H + h_idx)  # float32
    dt_val = tl.load(dt_bias_ptr + h_idx)       # float32
    b_val = tl.load(b_ptr + b_idx * H + h_idx)  # float32

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)

    # Load A_log[h]
    exp_A = tl.load(A_log_ptr + h_idx)  # float32

    g = tl.exp(-exp_A * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    state_ptr,        # [B,H,V,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    v_ptr,            # [B,H,V] float32
    k_exp_ptr,        # [B,H,K] float32
    beta_ptr,         # [B,H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,  # set to 128
    K: tl.constexpr,  # set to 128
    stride_b,         # int
    stride_h,         # int
    stride_v,         # int
    stride_k,         # int
    stride_new_b,     # int
    stride_new_h,     # int
    stride_new_i,     # int
    stride_new_j,     # int
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # Load beta and compute effective factor
    beta_val = tl.load(beta_ptr + b_idx * H + h_idx)  # float32
    one_minus_beta = 1.0 - beta_val

    # Iterate over i in [0..V-1]
    for i in tl.static_range(0, V):
        # Compute old_v = sum_j k_exp[j] * state[b,h,i,j]
        old_v = 0.0
        for j in tl.static_range(0, K):
            k_elem = tl.load(k_exp_ptr + b_idx * H * K + h_idx * K + j)  # [B,H,K] flattened
            addr = b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k
            state_elem = tl.load(state_ptr + addr)
            old_v += k_elem * state_elem

        # Compute new_v[h,i] = beta * v[b,h,i] + (1-beta) * old_v
        addr_v = b_idx * stride_b + h_idx * stride_h + i * stride_v
        v_elem = tl.load(v_ptr + addr_v)
        new_v = beta_val * v_elem + one_minus_beta * old_v

        # Update new_state[b,h,i,j] for all j
        for j in tl.static_range(0, K):
            k_elem = tl.load(k_exp_ptr + b_idx * H * K + h_idx * K + j)
            addr_state = b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k
            state_old = tl.load(state_ptr + addr_state)
            new_val = state_old - old_v + k_elem * new_v
            addr_new = b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + j * stride_new_j
            tl.store(new_state_ptr + addr_new, new_val)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,        # [B,H,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    out_ptr,          # [B,H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,  # set to 128
    K: tl.constexpr,  # set to 128
    scale,            # float32
    stride_b_q,       # int
    stride_h_q,       # int
    stride_k_q,       # int
    stride_b_ns,      # int
    stride_h_ns,      # int
    stride_i_ns,      # int
    stride_j_ns,      # int
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    acc = 0.0
    # sum_i q_exp[b,h,i] * new_state[b,h,i,0]
    for i in tl.static_range(0, V):
        addr_q = b_idx * stride_b_q + h_idx * stride_h_q + i * stride_k_q
        q_elem = tl.load(q_exp_ptr + addr_q)  # [B,H,K]
        addr_ns = b_idx * stride_b_ns + h_idx * stride_h_ns + i * stride_i_ns + 0 * stride_j_ns
        ns_elem = tl.load(new_state_ptr + addr_ns)
        acc += q_elem * ns_elem

    out_val = acc * scale
    tl.store(out_ptr + b_idx * H + h_idx, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Cast and make inputs contiguous, work in float32
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()   # [B,1,Q,K]
        k_f32 = k.to(torch.float32).contiguous()   # [B,1,Q,K]
        v_f32 = v.to(torch.float32).contiguous()   # [B,1,V,K]
        state_f32 = state.to(torch.float32).contiguous()  # [B,H,V,K]
        A_log_f32 = A_log.to(torch.float32).contiguous()  # [H]
        a_f32 = a.to(torch.float32).contiguous()         # [B,1,H]
        dt_bias_f32 = dt_bias.to(torch.float32).contiguous()  # [H]
        b_f32 = b.to(torch.float32).contiguous()         # [B,1,H]

        # Expand q and k heads by repeat_interleave ratio = num_v_heads // num_q_heads = 2
        q_exp = q_f32.repeat_interleave(2, dim=1)  # [B,8,K]
        k_exp = k_f32.repeat_interleave(2, dim=1)  # [B,8,K]
        B, H, K = q_exp.shape
        _, _, V, _ = v_f32.shape
        assert K == 128 and V == 128, "K and V must be 128 for this kernel."

        # Allocate outputs for g and beta
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta over (B*H)
        grid = (B * H,)
        compute_g_beta_kernel[grid](
            A_log_f32, a_f32.squeeze(1), dt_bias_f32, b_f32.squeeze(1), g_out, beta_out,
            H=H,
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Get strides
        stride_b = state_f32.stride(0)
        stride_h = state_f32.stride(1)
        stride_v = state_f32.stride(2)
        stride_k = state_f32.stride(3)

        stride_new_b = new_state.stride(0)
        stride_new_h = new_state.stride(1)
        stride_new_i = new_state.stride(2)
        stride_new_j = new_state.stride(3)

        # Launch Triton state update kernel
        state_update_kernel[grid](
            state_f32, new_state, v_f32.squeeze(1), k_exp, beta_out,
            B=B, H=H, V=128, K=128,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Compute output dot product: out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        q_exp_flat = q_exp.reshape(B * H, K).contiguous()  # [B*H,K]
        stride_b_q = q_exp_flat.stride(0)
        stride_h_q = q_exp_flat.stride(1) if q_exp_flat.dim() == 2 else 1
        stride_k_q = q_exp_flat.stride(1) if q_exp_flat.dim() == 2 else 1

        stride_b_ns = new_state.stride(0)
        stride_h_ns = new_state.stride(1)
        stride_i_ns = new_state.stride(2)
        stride_j_ns = new_state.stride(3)

        out = torch.empty((B, H), dtype=torch.float32, device=device)

        output_dot_kernel[grid](
            q_exp_flat, new_state, out,
            B=B, H=H, V=128, K=128, scale=scale,
            stride_b_q=stride_b_q, stride_h_q=stride_h_q, stride_k_q=stride_k_q,
            stride_b_ns=stride_b_ns, stride_h_ns=stride_h_ns, stride_i_ns=stride_i_ns, stride_j_ns=stride_j_ns,
        )

        # Prepare outputs: output [B,1,H,1] bfloat16; new_state [B,H,V,K] float32
        output_bf16 = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        return (output_bf16, new_state)


# Optional: helper functions for the evaluation harness (not required by the task, but kept for completeness).
def get_inputs():
    q = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    state = torch.randn([1, 8, 128, 128], dtype=torch.float32, device='cuda')
    A_log = torch.randn([8], dtype=torch.float32, device='cuda')
    a = torch.randn([1, 1, 8], dtype=torch.bfloat16, device='cuda')
    dt_bias = torch.randn([8], dtype=torch.float32, device='cuda')
    b = torch.randn([1, 1, 8], dtype=torch.bfloat16, device='cuda')
    scale = 1.0
    return [q, k, v, state, A_log, a, dt_bias, b, scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8)
    # Return a list/tuple to satisfy evaluator unpacking
    if isinstance(_out, (tuple, list)):
        return list(_out)
    else:
        return [_out]


def run(*args):
    return ModelNew()(*args)
