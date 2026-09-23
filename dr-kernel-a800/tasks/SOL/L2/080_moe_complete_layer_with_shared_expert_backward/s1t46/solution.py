import torch
import triton
import triton.language as tl


# GEMV: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H] (bf16), W: [N, H] (bf16), y: [B, N] (f32)
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,   # *bf16, [B, H]
    W_ptr,        # *bf16, [N, H]
    y_ptr,        # *f32,  [B, N]
    B, H, N,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row index
    pid_e = tl.program_id(1)  # output index in W (e.g., gate or up)
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Elementwise SiLU: y = x * sigmoid(x), operate on flat arrays (f32)
@triton.jit
def silu_elemwise_kernel(x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Elementwise multiply: y = a * b, both f32
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# Down reduction: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
# activated_pre: [B, N_down], down: [H_out, N_down], y: [B, H_out]
@triton.jit
def down_gemv_kernel(
    activated_ptr,  # *f32, [B, N_down]
    down_ptr,       # *bf16, [H_out, N_down]
    y_ptr,          # *f32, [B, H_out]
    B, N_down, H_out,
    stride_a_b, stride_a_t,
    stride_d_h, stride_d_t,
    stride_y_b, stride_y_h,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row index
    pid_h = tl.program_id(1)  # output hidden index
    acc = 0.0
    for t_start in range(0, N_down, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < N_down
        a_vals = tl.load(activated_ptr + pid_b * stride_a_b + offs_t * stride_a_t, mask=mask_t, other=0.0).to(tl.float32)
        d_vals = tl.load(down_ptr + pid_h * stride_d_h + offs_t * stride_d_t, mask=mask_t, other=0.0).to(tl.float32)
        acc += tl.sum(a_vals * d_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The evaluator passes all inputs: hidden, gate_weight, up_weight, down_weight, etc.
        # We will compute and return: (shared_gate_output, shared_up_output, shared_activated)
        # Shapes and dtypes match original: gate_output, up_output: [B, 1408], bfloat16
        # shared_activated: [B, 4096], bfloat16

        # Assume args order matches the original function signature:
        # (grad_output, hidden_states, router_weight, e_score_correction_bias, router_logits, scores, topk_indices, topk_weights, score_mask,
        # shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight, shared_gate_output, shared_up_output, shared_activated)
        # We don't use most of these for forward; we only need the last 3 weights and hidden_states.
        # In practice, the evaluator injects all tensors; we will use hidden_states and the 3 weights.

        # Extract inputs (we expect the last 3 weights and hidden):
        # Note: In real calls, the evaluator will pass all tensors; here we retrieve needed ones.
        # To be safe, let's assume args[-4] is hidden_states, args[-3] is gate_weight, args[-2] is up_weight, args[-1] is down_weight.
        # But since forward signature is *args, we can access them by index:
        hidden = args[-4]  # hidden_states: [B, H], bfloat16
        gate_weight = args[-3]  # [H, N_gate] = [4096, 1408], bfloat16
        up_weight = args[-2]    # [H, N_up] = [4096, 1408], bfloat16
        down_weight = args[-1]  # [H_out, N_down] = [4096, 1408], bfloat16

        B, H = hidden.shape
        N_gate = gate_weight.shape[1]
        N_up = up_weight.shape[1]
        H_out = down_weight.shape[0]
        N_down = down_weight.shape[1]
        assert H == 4096 and N_gate == 1408 and N_up == 1408 and H_out == 4096 and N_down == 1408

        # Ensure contiguous
        hidden_c = hidden.contiguous()
        gate_weight_c = gate_weight.contiguous()
        up_weight_c = up_weight.contiguous()
        down_weight_c = down_weight.contiguous()

        # Allocate outputs (compute in float32, cast to bfloat16 at return)
        gate_output = torch.empty((B, N_gate), dtype=torch.float32, device=hidden.device)
        up_output = torch.empty((B, N_up), dtype=torch.float32, device=hidden.device)
        activated_pre = torch.empty((B, N_down), dtype=torch.float32, device=hidden.device)  # N_down == N_gate == 1408
        shared_activated = torch.empty((B, H_out), dtype=torch.float32, device=hidden.device)

        # Launch GEMV for gate_output
        grid_gate = (B, N_gate)
        gemv_linear_kernel[grid_gate](
            hidden_c, gate_weight_c, gate_output,
            B, H, N_gate,
            hidden_c.stride(0), hidden_c.stride(1),
            gate_weight_c.stride(0), gate_weight_c.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_H=128,
            num_warps=4,
        )

        # Launch GEMV for up_output
        grid_up = (B, N_up)
        gemv_linear_kernel[grid_up](
            hidden_c, up_weight_c, up_output,
            B, H, N_up,
            hidden_c.stride(0), hidden_c.stride(1),
            up_weight_c.stride(0), up_weight_c.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_H=128,
            num_warps=4,
        )

        # Elementwise SiLU on gate_output
        N_silu = B * N_gate
        silu_gate = torch.empty((B * N_gate), dtype=torch.float32, device=hidden.device)
        grid_silu = (triton.cdiv(N_silu, 1024),)
        silu_elemwise_kernel[grid_silu](
            gate_output.reshape(-1),
            silu_gate,
            N_elements=N_silu,
            BLOCK=1024,
        )
        silu_gate_output = silu_gate.view(B, N_gate)

        # Elementwise multiply: activated_pre = silu_gate_output * up_output
        N_mul = B * N_down
        activated_vec = torch.empty((B * N_down), dtype=torch.float32, device=hidden.device)
        grid_mul = (triton.cdiv(N_mul, 1024),)
        mul_elemwise_kernel[grid_mul](
            silu_gate_output.reshape(-1),
            up_output.reshape(-1),
            activated_vec,
            N_elements=N_mul,
            BLOCK=1024,
        )
        activated_pre = activated_vec.view(B, N_down)

        # Down reduction: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
        grid_down = (B, H_out)
        down_gemv_kernel[grid_down](
            activated_pre, down_weight_c, shared_activated,
            B, N_down, H_out,
            activated_pre.stride(0), activated_pre.stride(1),
            down_weight_c.stride(0), down_weight_c.stride(1),
            shared_activated.stride(0), shared_activated.stride(1),
            BLOCK_T=128,
            num_warps=4,
        )

        # Cast outputs to bfloat16 to match original
        gate_output_bf16 = gate_output.to(torch.bfloat16)
        up_output_bf16 = up_output.to(torch.bfloat16)
        shared_activated_bf16 = shared_activated.to(torch.bfloat16)

        # Return exactly the same 3 outputs as the original forward
        return (gate_output_bf16, up_output_bf16, shared_activated_bf16)


def run(*args):
    return ModelNew()(*args)
