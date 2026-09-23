import torch
import triton
import triton.language as tl


# Triton GEMV: y[b, e] = sum_h hidden[b, h] * W[e, h]
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
    pid_e = tl.program_id(1)  # output index in W (t)
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        # Load hidden row segment
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        # Load W row segment
        w_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * w_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Triton elementwise SiLU: y = x * sigmoid(x), operate on flat buffers (f32)
@triton.jit
def silu_elemwise_kernel(x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Triton elementwise multiply: y = a * b on flat buffers (f32)
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# Triton GEMV: y[b, h] = sum_t pre[b, t] * down[h, t]
# pre: [B, N_down], down: [H_out, N_down], y: [B, H_out] (f32)
@triton.jit
def down_gemv_kernel(
    pre_ptr,       # *f32, [B, N_down]
    down_ptr,      # *f32, [H_out, N_down]
    y_ptr,         # *f32, [B, H_out]
    B: tl.constexpr,
    H_out: tl.constexpr,
    N_down: tl.constexpr,
    stride_pre_b, stride_pre_t,
    stride_down_h, stride_down_t,
    stride_y_b, stride_y_h,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # output index in down (h)
    acc = 0.0
    for t_start in range(0, N_down, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < N_down
        pre_vals = tl.load(pre_ptr + pid_b * stride_pre_b + offs_t * stride_pre_t, mask=mask_t, other=0.0).to(tl.float32)
        down_vals = tl.load(down_ptr + pid_h * stride_down_h + offs_t * stride_down_t, mask=mask_t, other=0.0).to(tl.float32)
        acc += tl.sum(pre_vals * down_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,
        shared_expert_up_weight: torch.Tensor,
        shared_expert_down_weight: torch.Tensor,
    ) -> tuple:
        # Ensure inputs are on the same device
        assert hidden_states.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        device = hidden_states.device

        B = hidden_states.shape[0]
        H = hidden_states.shape[1]  # 4096
        N_gate = shared_expert_gate_weight.shape[1]  # 1408
        N_up = shared_expert_up_weight.shape[1]      # 1408
        H_out = shared_expert_down_weight.shape[0]   # 4096
        N_down = shared_expert_down_weight.shape[1]  # 1408

        # 1) Compute gate_output: F.linear(hidden, shared_expert_gate_weight) -> [B, 1408]
        gate_out_f32 = torch.empty((B, N_gate), dtype=torch.float32, device=device)
        grid_gate = (B, N_gate)
        gemv_linear_kernel[grid_gate](
            hidden_states,
            shared_expert_gate_weight,
            gate_out_f32,
            B=B, H=H, N=N_gate,
            stride_h_b=hidden_states.stride(0), stride_h_h=hidden_states.stride(1),
            stride_W_e=shared_expert_gate_weight.stride(0), stride_W_h=shared_expert_gate_weight.stride(1),
            stride_y_b=gate_out_f32.stride(0), stride_y_e=gate_out_f32.stride(1),
            BLOCK_H=128,
        )

        # 2) Compute up_output: F.linear(hidden, shared_expert_up_weight) -> [B, 1408]
        up_out_f32 = torch.empty((B, N_up), dtype=torch.float32, device=device)
        grid_up = (B, N_up)
        gemv_linear_kernel[grid_up](
            hidden_states,
            shared_expert_up_weight,
            up_out_f32,
            B=B, H=H, N=N_up,
            stride_h_b=hidden_states.stride(0), stride_h_h=hidden_states.stride(1),
            stride_W_e=shared_expert_up_weight.stride(0), stride_W_h=shared_expert_up_weight.stride(1),
            stride_y_b=up_out_f32.stride(0), stride_y_e=up_out_f32.stride(1),
            BLOCK_H=128,
        )

        # 3) Elementwise SiLU on gate_output -> [B, 1408]
        silu_gate_f32 = torch.empty_like(gate_out_f32)
        N_gates = gate_out_f32.numel()
        grid_silu = (triton.cdiv(N_gates, 1024),)
        silu_elemwise_kernel[grid_silu](gate_out_f32, silu_gate_f32, N_elements=N_gates, BLOCK=1024)

        # 4) Elementwise multiply: activated_pre = silu_gate_output * up_output -> [B, 1408]
        pre_f32 = torch.empty_like(up_out_f32)
        N_pre = up_out_f32.numel()
        grid_mul = (triton.cdiv(N_pre, 1024),)
        mul_elemwise_kernel[grid_mul](silu_gate_f32, up_out_f32, pre_f32, N_elements=N_pre, BLOCK=1024)

        # 5) Compute shared_activated: F.linear(pre, shared_expert_down_weight) -> [B, 4096]
        act_f32 = torch.empty((B, H_out), dtype=torch.float32, device=device)
        grid_down = (B, H_out)
        down_gemv_kernel[grid_down](
            pre_f32,
            shared_expert_down_weight,
            act_f32,
            B=B, H_out=H_out, N_down=N_down,
            stride_pre_b=pre_f32.stride(0), stride_pre_t=pre_f32.stride(1),
            stride_down_h=shared_expert_down_weight.stride(0), stride_down_t=shared_expert_down_weight.stride(1),
            stride_y_b=act_f32.stride(0), stride_y_h=act_f32.stride(1),
            BLOCK_T=128,
        )

        # Cast outputs to bfloat16 to match original model's dtype for returned tensors
        gate_out = gate_out_f32.to(torch.bfloat16)
        up_out   = up_out_f32.to(torch.bfloat16)
        act_out  = act_f32.to(torch.bfloat16)

        return gate_out, up_out, act_out


def run(*args):
    return ModelNew()(*args)
