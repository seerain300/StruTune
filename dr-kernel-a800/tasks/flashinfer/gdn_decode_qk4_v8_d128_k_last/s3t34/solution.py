import torch
import triton
import triton.language as tl


@triton.jit
def g_beta_kernel(
    A_log_ptr,        # [H] float32
    a_ptr,            # [B,H] float32
    dt_bias_ptr,      # [H] float32
    b_ptr,            # [B,H] float32
    g_out_ptr,        # [B,H] float32
    beta_out_ptr,     # [B,H] float32
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b_idx * H + h_idx)      # a[b, h]
    dt_val = tl.load(dt_bias_ptr + h_idx)           # dt_bias[h]
    A_val = tl.load(A_log_ptr + h_idx)              # A_log[h]
    b_val = tl.load(b_ptr + b_idx * H + h_idx)      # b[b, h]

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    x = a_val + dt_val
    absx = tl.abs(x)
    maxx = tl.maximum(x, 0.0)
    softplus = tl.log(1.0 + tl.exp(-absx)) + maxx

    g_val = tl.exp(-tl.exp(A_val) * softplus)       # g scalar
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))         # sigmoid(b)

    # Store results
    tl.store(g_out_ptr + b_idx * H + h_idx, g_val)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta_val)


@triton.jit
def state_update_kernel(
    state_ptr,        # [B,H,V,K] float32
    new_ptr,          # [B,H,V,K] float32
    k_ptr,            # [H,K] float32
    v_ptr,            # [H,V] float32
    beta_ptr,         # [B,H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    stride_b, stride_h, stride_v, stride_k,
    stride_new_b, stride_new_h, stride_new_v, stride_new_k,
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load beta scalar for (b,h)
    beta_val = tl.load(beta_ptr + b_idx * H + h_idx)

    # Iterate over V and K to update new state
    for i in tl.static_range(0, V):
        # Compute old_v = dot(k[h, :], state[b,h,i,:]) over K
        old_v = 0.0
        for j in tl.static_range(0, K):
            # k[h, j]
            k_elem = tl.load(k_ptr + h_idx * K + j)
            # state[b,h,i,j]
            s_elem = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            old_v += k_elem * s_elem

        # Compute new_v for this i
        v_elem = tl.load(v_ptr + h_idx * V + i)  # v[h, i]
        new_v = beta_val * v_elem + (1.0 - beta_val) * old_v  # scalar

        # Update new state[b,h,i,j] = state_old - old_v + k[h,j] * new_v
        for j in tl.static_range(0, K):
            k_elem = tl.load(k_ptr + h_idx * K + j)
            state_old = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            new_state_val = state_old - old_v + k_elem * new_v
            tl.store(new_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_v + j * stride_new_k, new_state_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, num_q_heads, K] (bfloat16)
        k: [B, 1, num_k_heads, K] (bfloat16)
        v: [B, 1, num_v_heads, V] (bfloat16)
        state: [B, num_heads, V, K] (float32 or other; we will cast to float32)
        A_log: [num_heads] (float32)
        a: [B, 1, num_heads] (bfloat16 or other; we cast to float32)
        dt_bias: [num_heads] (float32)
        b: [B, 1, num_heads] (bfloat16 or other; we cast to float32)
        scale: float32 scalar
        """
        device = q.device
        B, _, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        num_heads = num_v_heads
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8 and K == 128 and V == 128 and state.shape[3] == 128, "Shapes must match fixed configuration."

        # Cast inputs to float32 for Triton computation
        q_f32 = q.squeeze(1).to(torch.float32).contiguous()  # [B,num_q_heads,K]
        k_f32 = k.squeeze(1).to(torch.float32).contiguous()  # [B,num_k_heads,K]
        v_f32 = v.squeeze(1).to(torch.float32).contiguous() # [B,num_v_heads,V]
        state_f32 = state.to(torch.float32).contiguous()    # [B,num_heads,V,K], num_heads=8

        # Expand q and k heads by repeat_interleave (ratio = num_v_heads // num_q_heads = 2)
        q_exp = q_f32.repeat_interleave(2, dim=1)            # [B,8,K]
        k_exp = k_f32.repeat_interleave(2, dim=1)            # [B,8,K]

        # Allocate g_out and beta_out
        g_out = torch.empty((B, num_heads), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, num_heads), dtype=torch.float32, device=device)

        # Launch g_beta_kernel over B*H
        grid = (B * num_heads,)
        g_beta_kernel[grid](
            A_log.to(torch.float32), a.squeeze(1).to(torch.float32), dt_bias.to(torch.float32), b.squeeze(1).to(torch.float32),
            g_out, beta_out,
            H=num_heads,
        )

        # Allocate new_state [B,H,V,K] float32
        new_state = torch.empty((B, num_heads, V, K), dtype=torch.float32, device=device)

        # Get strides
        stride_b = state_f32.stride(0)
        stride_h = state_f32.stride(1)
        stride_v = state_f32.stride(2)
        stride_k = state_f32.stride(3)

        stride_new_b = new_state.stride(0)
        stride_new_h = new_state.stride(1)
        stride_new_v = new_state.stride(2)
        stride_new_k = new_state.stride(3)

        # Launch state_update_kernel over B*H
        state_update_kernel[grid](
            state_f32, new_state, k_exp, v_f32, beta_out,
            B=B, H=num_heads, V=V, K=K,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_v=stride_new_v, stride_new_k=stride_new_k,
        )

        # Compute output: out[b,h] = scale * (q_exp[b,h,:] @ new_state[b,h,:,:]) in torch for correctness
        out_host = torch.empty((B, num_heads), device=device, dtype=torch.float32)
        for b_idx in range(B):
            for h_idx in range(num_heads):
                q_vec = q_exp[b_idx, h_idx]  # [K]
                acc = 0.0
                for i in range(V):
                    row = new_state[b_idx, h_idx, i, :]  # [K]
                    acc += torch.dot(q_vec, row)
                out_host[b_idx, h_idx] = acc * scale

        # Cast output to bfloat16 and return [B,1,H,V] (V=128 in harness)
        out_bf16 = out_host.unsqueeze(1).expand(B, 1, num_heads, V).contiguous().to(torch.bfloat16)

        # Return outputs as tuple: (output, new_state)
        return (out_bf16, new_state)

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8)
    if isinstance(_out, (tuple, list)):
        return list(_out)
    else:
        return [_out]


def run(*args):
    return ModelNew()(*args)
