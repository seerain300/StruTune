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
    H: tl.constexpr,  # number of heads (e.g., 8)
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b_idx * H + h_idx)       # float32
    dt_val = tl.load(dt_bias_ptr + h_idx)            # float32
    b_val = tl.load(b_ptr + b_idx * H + h_idx)       # float32

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for numerical stability
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)

    exp_A = tl.exp(A_log_ptr + h_idx)                # float32
    g = tl.exp(-exp_A * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    state_ptr,        # [B,H,V,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    k_exp_ptr,        # [B,H,K] float32
    v_ptr,            # [B,H,V] float32
    beta_ptr,         # [B,H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    stride_b,         # int: stride for B in state_ptr
    stride_h,         # int: stride for H in state_ptr
    stride_v,         # int: stride for V in state_ptr
    stride_k,         # int: stride for K in state_ptr
    stride_new_b,     # int: stride for B in new_state_ptr
    stride_new_h,     # int: stride for H in new_state_ptr
    stride_new_i,     # int: stride for V in new_state_ptr
    stride_new_j,     # int: stride for K in new_state_ptr
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # For each row i in V and each column j in K
    for i in tl.static_range(0, V):
        # old_v = dot(k_exp[b,h,:], state[b,h,i,:])
        acc = 0.0
        for j in tl.static_range(0, K):
            k_elem = tl.load(k_exp_ptr + b_idx * H * K + h_idx * K + j)  # [B,H,K] flattened address: b*H*K + h*K + j
            state_elem = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            acc += k_elem * state_elem
        # new_v = beta * v[b,h,i] + (1 - beta) * old_v
        beta_val = tl.load(beta_ptr + b_idx * H + h_idx)
        v_addr = b_idx * (H * V) + h_idx * V + i
        v_elem = tl.load(v_ptr + v_addr)
        new_v = beta_val * v_elem + (1.0 - beta_val) * acc

        # Update new_state[b,h,i,j] = state - old_v + k[j] * new_v
        for jj in tl.static_range(0, K):
            k_elem = tl.load(k_exp_ptr + b_idx * H * K + h_idx * K + jj)
            state_elem = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + jj * stride_k)
            new_state_elem = state_elem - acc + k_elem * new_v
            tl.store(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + jj * stride_new_j, new_state_elem)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,        # [B,H,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    out_ptr,          # [B,H] float32
    scale,            # float32
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,  # needed for addressing new_state with V dimension
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    acc = 0.0
    for j in tl.static_range(0, K):
        q_elem = tl.load(q_exp_ptr + b_idx * H * K + h_idx * K + j)
        # address for new_state[b,h,0,j] when V=1: b*H*V*K + h*V*K + 0*K + j
        ns_elem = tl.load(new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + 0 * K + j)
        acc += q_elem * ns_elem
    out_val = acc * scale
    tl.store(out_ptr + b_idx * H + h_idx, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward:
        - Computes g and beta via Triton
        - Updates new_state via Triton
        - Computes output via Triton
        Returns:
          - output_bf16: [B, 1, H, 1] bfloat16
          - new_state_f32: [B, H, V, K] float32
        """
        # Ensure device and dtype
        device = q.device
        B, _, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        _, _, H, _ = state.shape
        ratio = num_v_heads // num_q_heads

        # Cast to float32 and make contiguous for Triton
        q_f32 = q.to(torch.float32).contiguous()        # [B,1,4,K]
        k_f32 = k.to(torch.float32).contiguous()        # [B,1,4,K]
        v_f32 = v.to(torch.float32).contiguous()        # [B,1,8,V]
        state_f32 = state.to(torch.float32).contiguous()  # [B,H,V,K]

        # Expand q and k to H=num_v_heads via repeat_interleave
        q_exp = q_f32.repeat_interleave(ratio, dim=1)   # [B,8,K]
        k_exp = k_f32.repeat_interleave(ratio, dim=1)   # [B,8,K]
        v_exp = v_f32.repeat_interleave(ratio, dim=1)   # [B,8,V]

        # Allocate outputs for g and beta
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta over (B*H)
        grid = (B * H,)
        compute_g_beta_kernel[grid](
            A_log.to(torch.float32), a.squeeze(1).to(torch.float32), dt_bias.to(torch.float32), b.squeeze(1).to(torch.float32),
            g_out, beta_out,
            H=H,
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Get strides for tensors (in elements)
        stride_b = state_f32.stride(0)
        stride_h = state_f32.stride(1)
        stride_v = state_f32.stride(2)
        stride_k = state_f32.stride(3)

        stride_new_b = new_state.stride(0)
        stride_new_h = new_state.stride(1)
        stride_new_i = new_state.stride(2)
        stride_new_j = new_state.stride(3)

        # Launch Triton state update kernel over (B*H)
        state_update_kernel[grid](
            state_f32, new_state, k_exp, v_exp, beta_out,
            B=B, H=H, V=V, K=K,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h]) via Triton, output as [B,1,H,1] bfloat16
        out = torch.empty((B, H), dtype=torch.float32, device=device)
        q_exp = q_exp.contiguous()
        grid_out = (B * H,)
        output_dot_kernel[grid_out](
            q_exp, new_state, out, float(scale),
            B=B, H=H, K=K, V=V,
        )

        # Prepare outputs as required by evaluator: [B,1,H,1] bfloat16 and [B,H,V,K] float32
        out_bf16 = out.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        # Return a list for the first output and the state for the second
        return [out_bf16], new_state


def run(*args):
    return ModelNew()(*args)
