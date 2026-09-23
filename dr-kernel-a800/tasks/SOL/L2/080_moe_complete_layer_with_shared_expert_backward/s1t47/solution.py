import torch
import triton
import triton.language as tl


# Triton GEMV-like kernel: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H] (bf16/f32), W: [N, H] (bf16/f32), y: [B, N] (f32)
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,    # *bf16/f32, [B, H]
    W_ptr,         # *bf16/f32, [N, H]
    y_ptr,         # *f32,      [B, N]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_e = tl.program_id(1)  # output index (gate or up)
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Triton elementwise SiLU: y = x * sigmoid(x), operates on a flat vector
@triton.jit
def silu_elemwise_kernel(x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_f32))
    y = x_f32 * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Triton elementwise multiply: y = a * b, both inputs flat vectors
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# Triton GEMV-like reduction: y[b, h] = sum_t pre[b, t] * W[h, t]
# pre: [B, N_pre], W: [H_out, N_pre], y: [B, H_out]
@triton.jit
def down_gemv_kernel(
    pre_ptr,       # *f32, [B, N_pre]
    W_ptr,         # *bf16/f32, [H_out, N_pre]
    y_ptr,         # *f32, [B, H_out]
    B: tl.constexpr,
    N_pre: tl.constexpr,
    H_out: tl.constexpr,
    stride_pre_b, stride_pre_n,
    stride_W_h, stride_W_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # output hidden index
    acc = 0.0
    for n_start in range(0, N_pre, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_pre
        pre_vals = tl.load(pre_ptr + pid_b * stride_pre_b + offs_n * stride_pre_n, mask=mask_n, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_h * stride_W_h + offs_n * stride_W_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(pre_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight):
        """
        Returns:
        - shared_gate_output: [B, 1408], bfloat16
        - shared_up_output:   [B, 1408], bfloat16
        - shared_activated:   [B, 4096], bfloat16
        """
        # Shapes (fixed as per original code)
        B = hidden_states.shape[0]
        H = hidden_states.shape[1]                     # 4096
        N_gate = shared_expert_gate_weight.shape[1]   # 1408
        N_up = shared_expert_up_weight.shape[1]       # 1408
        N_down = shared_expert_down_weight.shape[1]   # 1408
        H_out = hidden_states.shape[1]                # 4096 (keep for clarity)

        # Ensure contiguous (metadata-only, no device-side torch ops)
        hidden = hidden_states.contiguous()
        gate_W = shared_expert_gate_weight.contiguous()
        up_W = shared_expert_up_weight.contiguous()
        down_W = shared_expert_down_weight.contiguous()

        # 1) shared_gate_output: [B, 1408], float32 accumulation
        gate_y = torch.empty((B, N_gate), device=hidden.device, dtype=torch.float32)
        grid_gate = (B, N_gate)
        gemv_linear_kernel[grid_gate](
            hidden, gate_W, gate_y,
            B, H, N_gate,
            hidden.stride(0), hidden.stride(1),
            gate_W.stride(0), gate_W.stride(1),
            gate_y.stride(0), gate_y.stride(1),
            BLOCK_H=256,
        )

        # 2) shared_up_output: [B, 1408], float32
        up_y = torch.empty((B, N_up), device=hidden.device, dtype=torch.float32)
        grid_up = (B, N_up)
        gemv_linear_kernel[grid_up](
            hidden, up_W, up_y,
            B, H, N_up,
            hidden.stride(0), hidden.stride(1),
            up_W.stride(0), up_W.stride(1),
            up_y.stride(0), up_y.stride(1),
            BLOCK_H=256,
        )

        # 3) SiLU(gate_y) in Triton, elementwise
        silu_gate_y = torch.empty_like(gate_y, device=hidden.device, dtype=torch.float32)
        N_silu = B * N_gate
        grid_silu = (triton.cdiv(N_silu, 1024),)
        silu_elemwise_kernel[grid_silu](gate_y.view(-1), silu_gate_y.view(-1), N_silu, 1024)

        # 4) activated_pre = silu_gate_y * up_y, elementwise
        activated_pre = torch.empty((B, N_up), device=hidden.device, dtype=torch.float32)
        N_mul = B * N_up
        grid_mul = (triton.cdiv(N_mul, 1024),)
        mul_elemwise_kernel[grid_mul](silu_gate_y.view(-1), up_y.view(-1), activated_pre.view(-1), N_mul, 1024)

        # 5) shared_activated: [B, 4096], float32 accumulation via down_gemv
        activated = torch.empty((B, H_out), device=hidden.device, dtype=torch.float32)
        grid_down = (B, H_out)
        down_gemv_kernel[grid_down](
            activated_pre, down_W, activated,
            B, N_down, H_out,
            activated_pre.stride(0), activated_pre.stride(1),
            down_W.stride(0), down_W.stride(1),
            activated.stride(0), activated.stride(1),
            BLOCK_N=256,
        )

        # Cast outputs to bfloat16 to match original
        gate_out = gate_y.to(torch.bfloat16)
        up_out = up_y.to(torch.bfloat16)
        act_out = activated.to(torch.bfloat16)

        return gate_out, up_out, act_out


def run(*args):
    return ModelNew()(*args)
