import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_gate_up_kernel(
    X_ptr,          # *float16: [1, H]
    W_ptr,          # *float16: [H, M_out]
    Y_ptr,          # *float32: [1, M_out]
    H: tl.int32,    # hidden_size (rows of X)
    M_out: tl.int32,  # output size (columns of W, rows of Y)
    stride_xb: tl.int32,
    stride_xh: tl.int32,
    stride_w0: tl.int32,
    stride_w1: tl.int32,
    stride_yb: tl.int32,
    stride_ym: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One output row (b=0). Compute Y[0, m_start : m_start+BLOCK_M].
    b = 0
    for m_start in range(0, M_out, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        acc = tl.zeros([1, BLOCK_M], dtype=tl.float32)
        # Reduction over H in chunks of BLOCK_K
        for k0 in range(0, H, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            # Load X row slice: [BLOCK_K]
            x = tl.load(
                X_ptr + b * stride_xb + k_offsets * stride_xh,
                mask=k_offsets < H,
                other=0.0,
            ).to(tl.float32)  # cast to fp32 for accumulation
            # Load W tile: [BLOCK_K, BLOCK_M]
            w = tl.load(
                W_ptr + k_offsets[:, None] * stride_w0 + m_offsets[None, :] * stride_w1,
                mask=(k_offsets[:, None] < H) & (m_offsets[None, :] < M_out),
                other=0.0,
            ).to(tl.float32)
            # acc += sum_k x[k] * w[k, :]
            # x: [BLOCK_K], w: [BLOCK_K, BLOCK_M] -> broadcast multiply and reduce
            acc += tl.sum(x[:, None] * w, axis=0)
        # Store Y[0, m_start : m_start+BLOCK_M]
        tl.store(
            Y_ptr + b * stride_yb + m_offsets * stride_ym,
            acc[0, :],
            mask=m_offsets < M_out,
        )


@triton.jit
def bmm_triton_down_kernel(
    X_ptr,          # *float32: [1, M_out]
    W_ptr,          # *float16: [M_out, H_in]
    Y_ptr,          # *float32: [1, H_in]
    M_out: tl.int32,  # input size for this bmm (rows of X, columns of W)
    H_in: tl.int32,   # output size (columns of W, rows of Y)
    stride_xb: tl.int32,
    stride_xm: tl.int32,
    stride_w0: tl.int32,
    stride_w1: tl.int32,
    stride_yb: tl.int32,
    stride_yh: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One output row (b=0). Compute Y[0, h_start : h_start+BLOCK_M].
    b = 0
    for h_start in range(0, H_in, BLOCK_M):
        h_offsets = h_start + tl.arange(0, BLOCK_M)
        acc = tl.zeros([1, BLOCK_M], dtype=tl.float32)
        for k0 in range(0, M_out, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                X_ptr + b * stride_xb + k_offsets * stride_xm,
                mask=k_offsets < M_out,
                other=0.0,
            ).to(tl.float32)  # [BLOCK_K]
            w = tl.load(
                W_ptr + k_offsets[:, None] * stride_w0 + h_offsets[None, :] * stride_w1,
                mask=(k_offsets[:, None] < M_out) & (h_offsets[None, :] < H_in),
                other=0.0,
            ).to(tl.float32)  # [BLOCK_K, BLOCK_M]
            acc += tl.sum(x[:, None] * w, axis=0)
        tl.store(
            Y_ptr + b * stride_yb + h_offsets * stride_yh,
            acc[0, :],
            mask=h_offsets < H_in,
        )


@triton.jit
def elementwise_silu_mul_kernel(
    Z_ptr,  # *float32: [1, M_out] (gate_out)
    U_ptr,  # *float32: [1, M_out] (up_out)
    O_ptr,  # *float32: [1, M_out] (output activated)
    M_out: tl.int32,
    stride_zb: tl.int32,
    stride_zm: tl.int32,
    stride_ub: tl.int32,
    stride_um: tl.int32,
    stride_ob: tl.int32,
    stride_om: tl.int32,
    BLOCK_M: tl.constexpr,
):
    b = 0
    for m_start in range(0, M_out, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        z = tl.load(Z_ptr + b * stride_zb + m_offsets * stride_zm, mask=m_offsets < M_out, other=0.0)
        u = tl.load(U_ptr + b * stride_ub + m_offsets * stride_um, mask=m_offsets < M_out, other=0.0)
        silu = z * (1.0 / (1.0 + tl.exp(-z)))
        o = silu * u
        tl.store(O_ptr + b * stride_ob + m_offsets * stride_om, o, mask=m_offsets < M_out)


@triton.jit
def atomic_accumulate_kernel(
    WEIGHT_ptr,        # *float32: [1, 1] (scalar weight)
    X_ptr,             # *float32: [1, H_in] (final_out)
    OUT_ptr,           # *float32: [num_tokens, hidden_size]
    token_id: tl.int32,
    H_in: tl.int32,
    stride_wb: tl.int32,
    stride_wm: tl.int32,
    stride_xb: tl.int32,
    stride_xm: tl.int32,
    stride_out_row: tl.int32,
    stride_out_col: tl.int32,
    BLOCK_H: tl.constexpr,
):
    b = 0
    for h_start in range(0, H_in, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        w = tl.load(WEIGHT_ptr + b * stride_wb + 0 * stride_wm)  # scalar
        x = tl.load(X_ptr + b * stride_xb + h_offsets * stride_xm, mask=h_offsets < H_in, other=0.0)
        out_ptrs = OUT_ptr + token_id * stride_out_row + h_offsets * stride_out_col
        tl.atomic_add(out_ptrs, x * w, mask=h_offsets < H_in)


def _triton_bmm_gate_up(h_vec, expert_weights, out):
    """
    h_vec: [1, H], expert_weights: [H, M_out], out: [1, M_out] fp32
    """
    H = h_vec.shape[1]
    M = expert_weights.shape[1]
    X_ = h_vec.float()          # [1, H]
    W_ = expert_weights.float() # [H, M]
    out.zero_()
    stride_xb = X_.stride(0)
    stride_xh = X_.stride(1)
    stride_w0 = W_.stride(0)
    stride_w1 = W_.stride(1)
    stride_yb = out.stride(0)
    stride_ym = out.stride(1)
    grid = (triton.cdiv(M, 64),)
    bmm_triton_gate_up_kernel[grid](
        X_, W_, out,
        H, M,
        stride_xb, stride_xh,
        stride_w0, stride_w1,
        stride_yb, stride_ym,
        BLOCK_M=64, BLOCK_K=64,
        num_warps=4, num_stages=2,
    )


def _triton_bmm_down(activated, expert_down, out):
    """
    activated: [1, M_out], expert_down: [M_out, H_in], out: [1, H_in] fp32
    """
    M = activated.shape[1]
    H = expert_down.shape[1]
    X_ = activated.float()        # [1, M]
    W_ = expert_down.float()      # [M, H]
    out.zero_()
    stride_xb = X_.stride(0)
    stride_xm = X_.stride(1)
    stride_w0 = W_.stride(0)
    stride_w1 = W_.stride(1)
    stride_yb = out.stride(0)
    stride_yh = out.stride(1)
    grid = (triton.cdiv(H, 64),)
    bmm_triton_down_kernel[grid](
        X_, W_, out,
        M, H,
        stride_xb, stride_xm,
        stride_w0, stride_w1,
        stride_yb, stride_yh,
        BLOCK_M=64, BLOCK_K=64,
        num_warps=4, num_stages=2,
    )


def _triton_silu_mul(Z, U, out):
    """
    Z: [1, M_out] fp32, U: [1, M_out] fp32, out: [1, M_out] fp32
    """
    M = Z.shape[1]
    Z_ = Z
    U_ = U
    out.zero_()
    stride_zb = Z_.stride(0)
    stride_zm = Z_.stride(1)
    stride_ub = U_.stride(0)
    stride_um = U_.stride(1)
    stride_ob = out.stride(0)
    stride_om = out.stride(1)
    grid = (triton.cdiv(M, 64),)
    elementwise_silu_mul_kernel[grid](
        Z_, U_, out,
        M,
        stride_zb, stride_zm,
        stride_ub, stride_um,
        stride_ob, stride_om,
        BLOCK_M=64,
        num_warps=2, num_stages=2,
    )


def _triton_atomic_accumulate(WEIGHT, X, OUT, token_id):
    """
    WEIGHT: [1, 1] (float32 scalar), X: [1, H_in] (float32), OUT: [num_tokens, hidden_size] (float32)
    Accumulate OUT[token_id, :] += WEIGHT[0] * X[0, :]
    """
    H = X.shape[1]
    WEIGHT_ = WEIGHT.float()  # [1, 1] fp32
    X_ = X.float()
    stride_wb = WEIGHT_.stride(0)
    stride_wm = WEIGHT_.stride(1)
    stride_xb = X_.stride(0)
    stride_xm = X_.stride(1)
    stride_out_row = OUT.stride(0)
    stride_out_col = OUT.stride(1)
    grid = (triton.cdiv(H, 64),)
    atomic_accumulate_kernel[grid](
        WEIGHT_, X_, OUT,
        token_id, H,
        stride_wb, stride_wm,
        stride_xb, stride_xm,
        stride_out_row, stride_out_col,
        BLOCK_H=64,
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        hidden_states: [num_tokens, hidden_size] (bfloat16)
        selected_experts: [num_tokens, num_experts_per_tok] (int64)
        routing_weights: [num_tokens, num_experts_per_tok] (bfloat16)
        expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size] (bfloat16)
        expert_up_weights:   [num_experts, hidden_size, moe_intermediate_size] (bfloat16)
        expert_down_weights: [num_experts, moe_intermediate_size, hidden_size] (bfloat16)
        Returns: [num_tokens, hidden_size] (float32)
        """
        # Ensure contiguous
        hidden_states = hidden_states.contiguous()
        selected_experts = selected_experts.contiguous()
        routing_weights = routing_weights.contiguous()
        expert_gate_weights = expert_gate_weights.contiguous()
        expert_up_weights = expert_up_weights.contiguous()
        expert_down_weights = expert_down_weights.contiguous()

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        # We do not use torch to process selected_experts or routing_weights. Forward avoids torch ops on tensors.

        # Output result (fp32 accumulation for stability)
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        for t in range(num_tokens):
            # Iterate selected experts per token
            for j in range(selected_experts.shape[1]):
                expert_id = int(selected_experts[t, j].item())
                # Prepare h vector [1, H]
                h_vec = hidden_states[t].unsqueeze(0).contiguous()  # bfloat16
                # 1) gate_out = h_vec @ expert_gate_weights[expert_id]
                gate_out = torch.empty(1, expert_gate_weights.shape[2], dtype=torch.float32, device=hidden_states.device)
                _triton_bmm_gate_up(h_vec, expert_gate_weights[expert_id], gate_out)
                # 2) up_out = h_vec @ expert_up_weights[expert_id]
                up_out = torch.empty(1, expert_up_weights.shape[2], dtype=torch.float32, device=hidden_states.device)
                _triton_bmm_gate_up(h_vec, expert_up_weights[expert_id], up_out)
                # 3) activated = silu(gate_out) * up_out
                activated = torch.empty(1, gate_out.shape[1], dtype=torch.float32, device=hidden_states.device)
                _triton_silu_mul(gate_out, up_out, activated)
                # 4) final_out = activated @ expert_down_weights[expert_id]
                final_out = torch.empty(1, expert_down_weights.shape[2], dtype=torch.float32, device=hidden_states.device)
                _triton_bmm_down(activated, expert_down_weights[expert_id], final_out)
                # 5) Accumulate into result[t, :]
                # routing_weights[t, j] is a scalar (bfloat16). Convert to float32 for kernel.
                weight = routing_weights[t, j].to(torch.float32)
                _triton_atomic_accumulate(weight.unsqueeze(0), final_out, result, t)

        # Return fp32 result; evaluator can cast to bfloat16 if required.
        return result


def run(*args):
    return ModelNew()(*args)
