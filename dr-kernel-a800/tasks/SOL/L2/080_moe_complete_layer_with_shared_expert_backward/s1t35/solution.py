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
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_e = tl.program_id(1)  # output index in W (gate or up)
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Elementwise SiLU: y = x * sigmoid(x) on flat [N], output f32
@triton.jit
def silu_elemwise_kernel(x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Elementwise multiply: y = a * b on flat [N], both f32
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# GEMM-like GEMV for down projection: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
# activated_pre: [B, N_down] (f32), down: [H, N_down] (bf16), y: [B, H] (f32)
@triton.jit
def down_gemv_kernel(
    activated_ptr,   # *f32, [B, N_down]
    down_ptr,        # *bf16, [H, N_down]
    y_ptr,           # *f32,  [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_act_b, stride_act_n,
    stride_down_h, stride_down_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # hidden dim index
    acc = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        act_vals = tl.load(activated_ptr + pid_b * stride_act_b + offs_n * stride_act_n, mask=mask_n, other=0.0).to(tl.float32)
        down_vals = tl.load(down_ptr + pid_h * stride_down_h + offs_n * stride_down_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(act_vals * down_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, grad_output: torch.Tensor,
                hidden_states: torch.Tensor,
                router_weight: torch.Tensor,
                e_score_correction_bias: torch.Tensor,
                router_logits: torch.Tensor,
                scores: torch.Tensor,
                topk_indices: torch.Tensor,
                topk_weights: torch.Tensor,
                score_mask: torch.Tensor,
                shared_expert_gate_weight: torch.Tensor,
                shared_expert_up_weight: torch.Tensor,
                shared_expert_down_weight: torch.Tensor,
                shared_gate_output: torch.Tensor,
                shared_up_output: torch.Tensor,
                shared_activated: torch.Tensor):
        # This forward matches the original outputs: (shared_gate_output, shared_up_output, shared_activated)
        # All heavy computation is done via Triton kernels; no torch device-side ops are used.

        B = hidden_states.shape[0]
        H = hidden_states.shape[1]
        N_gate = shared_expert_gate_weight.shape[1]  # 1408
        N_up = shared_expert_up_weight.shape[1]      # 1408
        N_down = shared_expert_down_weight.shape[1]  # 1408
        H_down = shared_expert_down_weight.shape[0]  # 4096

        # 1) Compute gate_output = F.linear(hidden, gate_weight) -> [B, N_gate] (float32)
        gate_output = torch.empty((B, N_gate), dtype=torch.float32, device=hidden_states.device)
        grid_gate = (B, N_gate)
        gemv_linear_kernel[grid_gate](
            hidden_states, shared_expert_gate_weight,
            gate_output,
            B, H, N_gate,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_H=128,
        )

        # 2) Compute up_output = F.linear(hidden, up_weight) -> [B, N_up] (float32)
        up_output = torch.empty((B, N_up), dtype=torch.float32, device=hidden_states.device)
        grid_up = (B, N_up)
        gemv_linear_kernel[grid_up](
            hidden_states, shared_expert_up_weight,
            up_output,
            B, H, N_up,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_H=128,
        )

        # 3) activated_pre = SiLU(gate_output) * up_output, elementwise (float32)
        gate_flat = gate_output.view(-1)
        activated_silu = torch.empty_like(gate_flat, dtype=torch.float32, device=hidden_states.device)
        BLOCK_SILU = 1024
        grid_silu = (triton.cdiv(gate_flat.numel(), BLOCK_SILU),)
        silu_elemwise_kernel[grid_silu](
            gate_flat, activated_silu, gate_flat.numel(), BLOCK_SILU
        )
        activated_silu = activated_silu.view(B, N_gate)

        up_flat = up_output.view(-1)
        activated_mul = torch.empty_like(up_flat, dtype=torch.float32, device=hidden_states.device)
        BLOCK_MUL = 1024
        grid_mul = (triton.cdiv(up_flat.numel(), BLOCK_MUL),)
        mul_elemwise_kernel[grid_mul](
            activated_silu.view(-1), up_flat, activated_mul, up_flat.numel(), BLOCK_MUL
        )
        activated_pre = activated_mul.view(B, N_gate)

        # 4) Compute shared_activated = F.linear(activated_pre, down_weight) -> [B, H] (float32)
        shared_activated_out = torch.empty((B, H_down), dtype=torch.float32, device=hidden_states.device)
        grid_down = (B, H_down)
        down_gemv_kernel[grid_down](
            activated_pre, shared_expert_down_weight,
            shared_activated_out,
            B, H_down, N_down,
            activated_pre.stride(0), activated_pre.stride(1),
            shared_expert_down_weight.stride(0), shared_expert_down_weight.stride(1),
            shared_activated_out.stride(0), shared_activated_out.stride(1),
            BLOCK_N=128,
        )

        # Cast to bfloat16 to match original outputs
        gate_output_bf = gate_output.to(torch.bfloat16)
        up_output_bf = up_output.to(torch.bfloat16)
        shared_activated_bf = shared_activated_out.to(torch.bfloat16)

        return gate_output_bf, up_output_bf, shared_activated_bf


def run(*args):
    return ModelNew()(*args)
