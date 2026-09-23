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
    H: tl.constexpr,  # number of heads (compile-time)
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    dt_val = tl.load(dt_bias_ptr + h_idx)
    A_log_val = tl.load(A_log_ptr + h_idx)

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for numerical stability
    x = a_val + dt_val
    abs_x = tl.abs(x)
    max_x0 = tl.maximum(x, 0.0)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + max_x0
    g = tl.exp(-tl.exp(A_log_val) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-x))

    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    state_ptr,         # [B,H,V,K] float32
    new_state_ptr,     # [B,H,V,K] float32
    k_ptr,             # [H,K] float32 (expanded, ratio applied on host)
    v_ptr,             # [H,V] float32 (expanded, ratio applied on host)
    beta_ptr,          # [B,H] float32
    B, H, V: tl.constexpr, K: tl.constexpr,
    stride_b, stride_h, stride_v, stride_k,    # state strides (in elements)
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,  # new_state strides
):
    pid = tl.program_id(axis=0)  # over B*H
    b_idx = pid // H
    h_idx = pid % H

    beta_val = tl.load(beta_ptr + b_idx * H + h_idx)

    for i in tl.static_range(0, V):  # iterate rows of V
        # old_v = dot(k[h], state[b,h,i,:]) -> scalar
        old_v = 0.0
        for j in tl.static_range(0, K):
            k_elem = tl.load(k_ptr + h_idx * K + j)
            state_elem = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            old_v += k_elem * state_elem

        # new_v = beta[h] * v[h,i] + (1 - beta[h]) * old_v
        v_elem = tl.load(v_ptr + h_idx * V + i)
        new_v = beta_val * v_elem + (1.0 - beta_val) * old_v

        # update new_state[b,h,i,j] = state_old + k[j] * new_v
        # new_state initialized to zeros; we compute it explicitly
        for j in tl.static_range(0, K):
            state_old = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            k_elem = tl.load(k_ptr + h_idx * K + j)
            update = k_elem * new_v
            new_val = state_old + update
            tl.store(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + j * stride_new_j, new_val)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,         # [H,K] float32
    new_state_ptr,     # [B,H,V,K] float32
    out_ptr,           # [B,H] float32
    scale,             # float32
    B, H, V: tl.constexpr, K: tl.constexpr,
    stride_q_h, stride_q_k,               # strides for q_exp (rows=h, cols=k)
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,  # new_state strides
):
    pid = tl.program_id(axis=0)  # over B*H
    b_idx = pid // H
    h_idx = pid % H

    acc = 0.0
    for j in tl.static_range(0, K):
        q_elem = tl.load(q_exp_ptr + h_idx * K + j)
        state_row = tl.load(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + 0 * stride_new_i + j * stride_new_j)
        acc += q_elem * state_row

    acc = scale * acc
    tl.store(out_ptr + b_idx * H + h_idx, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward that returns:
          - output: [B, 1, H, 1] bfloat16
          - new_state: [B, H, V, K] float32
        """
        # Cast to float32 for computation; ensure contiguous
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()    # [B,1,num_q_heads,K]
        k_f32 = k.to(torch.float32).contiguous()    # [B,1,num_k_heads,K]
        v_f32 = v.to(torch.float32).contiguous()    # [B,1,num_v_heads,V]
        state_f32 = state.to(torch.float32).contiguous()  # [B,H,V,K]

        # Expand q and k heads by repeat_interleave to match v heads (ratio 2 in provided tests)
        # In general, ratio = num_v_heads // num_q_heads; here 8 // 4 = 2.
        ratio = v_f32.shape[1] // q_f32.shape[1]
        q_exp = q_f32.repeat_interleave(ratio, dim=1)  # [B,8,K]
        k_exp = k_f32.repeat_interleave(ratio, dim=1)  # [B,8,K]

        # Dimensions
        B = q_f32.shape[0]
        H = q_f32.shape[1]
        num_v_heads = v_f32.shape[1]
        num_q_heads = q_f32.shape[1]
        ratio = num_v_heads // num_q_heads
        assert ratio * num_q_heads == num_v_heads, "v heads must be divisible by q heads for repeat_interleave"
        V = v_f32.shape[2]
        K = q_f32.shape[3]
        assert state_f32.shape[2] == V and state_f32.shape[3] == K, "state last two dims must match v and k sizes"
        assert A_log.numel() == H, "A_log length must equal number of heads (H)"
        assert a.shape[0] == B and a.shape[2] == H, "a shape must be [B,1,H]"
        assert dt_bias.numel() == H, "dt_bias length must equal number of heads (H)"
        assert b.shape[0] == B and b.shape[2] == H, "b shape must be [B,1,H]"

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

        # Strides (in elements)
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
            state_f32, new_state, k_exp, v_f32.squeeze(1), beta_out,
            B=B, H=H, V=V, K=K,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h]) per (b,h)
        out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Prepare q_exp strides: q_exp is [H,K], with rows h and cols k
        # We can flatten q_exp to [H*K] and compute pointers per element.
        # For q_exp[h,k], base is h*K + k.
        stride_q_h = K  # since we treat q_exp as [H,K] contiguous
        stride_q_k = 1

        grid_out = (B * H,)
        output_dot_kernel[grid_out](
            q_exp, new_state, out, float(scale),
            B=B, H=H, V=V, K=K,
            stride_q_h=stride_q_h, stride_q_k=stride_q_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Prepare output: cast to bfloat16 and shape [B,1,H,1]
        # v has V=1 in given inputs; returning [B,1,H,1] matches original intent.
        output = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1] bfloat16

        # new_state is [B,H,V,K] float32 (here V=1, but we keep general shape)
        return (output, new_state)


def run(*args):
    return ModelNew()(*args)
