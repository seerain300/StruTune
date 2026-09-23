import torch
import triton
import triton.language as tl


# GEMV: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H] (input dtype, we load and cast to f32 inside), W: [N, H] (input dtype, cast to f32), y: [B, N] (f32)
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,   # *bf16 or *f32, [B, H]
    W_ptr,        # *bf16 or *f32, [N, H]
    y_ptr,        # *f32,          [B, N]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_e = tl.program_id(1)  # output index (gate or up expert)
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Triton elementwise SiLU: y = x * sigmoid(x), operate on flat vectors (f32)
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


# Triton elementwise multiply: y = a * b on flat vectors, both f32
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# GEMV for down projection: y[b, h] = sum_t pre[b, t] * down[t, h]
# pre: [B, N], down: [H, N], y: [B, H] (f32)
@triton.jit
def down_gemv_kernel(
    pre_ptr,      # *bf16 or *f32, [B, N]
    down_ptr,     # *bf16 or *f32, [H, N]
    y_ptr,        # *f32,          [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_pre_b, stride_pre_n,
    stride_dn_h, stride_dn_n,
    stride_y_b, stride_y_h,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # hidden index
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            pre_vals = tl.load(pre_ptr + pid_b * stride_pre_b + offs_n * stride_pre_n, mask=mask_n, other=0.0).to(tl.float32)
            down_vals = tl.load(down_ptr + offs_h[:, None] * stride_dn_h + offs_n[None, :] * stride_dn_n,
                                mask=mask_h[:, None] & mask_n[None, :],
                                other=0.0).to(tl.float32)
            acc += tl.sum(pre_vals[None, :] * down_vals, axis=1)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        router_logits: torch.Tensor,
        scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        score_mask: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,  # [H, N_gate] = [4096, 1408], bf16
        shared_expert_up_weight: torch.Tensor,    # [H, N_up]   = [4096, 1408], bf16
        shared_expert_down_weight: torch.Tensor,  # [H, N_down] = [4096, 1408], bf16
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        # We will NOT use torch ops in forward; all computation is via Triton kernels.
        B = hidden_states.shape[0]
        H = hidden_states.shape[1]
        N_gate = shared_expert_gate_weight.shape[1]
        N_up = shared_expert_up_weight.shape[1]
        N_down = shared_expert_down_weight.shape[1]

        # 1) Compute gate_output = F.linear(hidden, gate_weight) -> [B, N_gate], f32
        gate_output = torch.empty((B, N_gate), device=hidden_states.device, dtype=torch.float32)
        grid_gate = (B, N_gate)
        gemv_linear_kernel[grid_gate](
            hidden_states, shared_expert_gate_weight, gate_output,
            B, H, N_gate,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_H=128,
        )

        # 2) Compute up_output = F.linear(hidden, up_weight) -> [B, N_up], f32
        up_output = torch.empty((B, N_up), device=hidden_states.device, dtype=torch.float32)
        grid_up = (B, N_up)
        gemv_linear_kernel[grid_up](
            hidden_states, shared_expert_up_weight, up_output,
            B, H, N_up,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_H=128,
        )

        # 3) Compute activated_pre = SiLU(gate_output) * up_output -> [B, N_up], f32
        activated_pre = torch.empty((B * N_up), device=hidden_states.device, dtype=torch.float32)
        silu_gate = torch.empty((B * N_gate), device=hidden_states.device, dtype=torch.float32)

        # For silu_gate: flatten gate_output
        gate_flat = gate_output.contiguous().view(-1)  # [B*N_gate]
        silu_elemwise_kernel[(B * N_gate,)](
            gate_flat, silu_gate,
            N_elements=B * N_gate,
            BLOCK=256,
        )
        # For mul: flatten up_output and activated_pre
        up_flat = up_output.contiguous().view(-1)  # [B*N_up]
        mul_elemwise_kernel[(B * N_up,)](
            silu_gate, up_flat, activated_pre,
            N_elements=B * N_up,
            BLOCK=256,
        )
        activated_pre = activated_pre.view(B, N_up)

        # 4) Compute shared_activated = F.linear(activated_pre, down_weight) -> [B, H], f32
        shared_activated = torch.empty((B, H), device=hidden_states.device, dtype=torch.float32)
        grid_down = (B, H)
        down_gemv_kernel[grid_down](
            activated_pre, shared_expert_down_weight, shared_activated,
            B, H, N_up,
            activated_pre.stride(0), activated_pre.stride(1),
            shared_expert_down_weight.stride(0), shared_expert_down_weight.stride(1),
            shared_activated.stride(0), shared_activated.stride(1),
            BLOCK_H=256, BLOCK_N=128,
        )

        # Return exactly the three outputs the original forward returns, cast to bf16
        return (
            gate_output.to(torch.bfloat16),
            up_output.to(torch.bfloat16),
            shared_activated.to(torch.bfloat16),
        )


def run(*args):
    return ModelNew()(*args)
