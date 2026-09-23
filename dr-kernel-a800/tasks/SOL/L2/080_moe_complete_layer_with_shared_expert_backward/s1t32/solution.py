import torch
import triton
import triton.language as tl


# GEMV kernel for N == 1408: y[b, e] = sum_h hidden[b, h] * W[e, h]
@triton.jit
def gemv_n1408_kernel(
    hidden_ptr,   # *bf16, [B, H]
    W_ptr,        # *bf16, [1408, H] or [1408, 1408], here H=4096
    y_ptr,        # *f32,  [B, 1408]
    B: tl.constexpr,
    H: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row index
    pid_e = tl.program_id(1)  # output feature index in W (1408)
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# GEMV kernel for N == 4096: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
@triton.jit
def gemv_n4096_kernel(
    inp_ptr,      # *bf16 or *f32, [B, 1408]
    down_ptr,     # *bf16, [4096, 1408]
    y_ptr,        # *f32,  [B, 4096]
    B: tl.constexpr,
    N_in: tl.constexpr,   # input features (1408)
    H_out: tl.constexpr,  # output hidden size (4096)
    stride_inp_b, stride_inp_n,
    stride_down_h, stride_down_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # output hidden index
    acc = 0.0
    for n_start in range(0, N_in, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_in
        inp_vals = tl.load(inp_ptr + pid_b * stride_inp_b + offs_n * stride_inp_n, mask=mask_n, other=0.0).to(tl.float32)
        down_vals = tl.load(down_ptr + pid_h * stride_down_h + offs_n * stride_down_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(inp_vals * down_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


# Elementwise SiLU: y = x * sigmoid(x), with masked load/store
@triton.jit
def silu_elemwise_kernel(
    x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Elementwise multiply: y = a * b (both float32 vectors)
@triton.jit
def mul_elemwise_kernel(
    a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Accept the same 19 tensors as the original run(...)
        (grad_output, hidden_states, router_weight, e_score_correction_bias,
         router_logits, scores, topk_indices, topk_weights, score_mask,
         shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
         shared_gate_output, shared_up_output, shared_activated) = args

        # We only need hidden_states and the three weights; all others are not used in outputs.
        B = hidden_states.shape[0]
        H = hidden_states.shape[1]  # 4096
        N_gate = shared_expert_gate_weight.shape[1]  # 1408
        N_up = shared_expert_up_weight.shape[1]      # 1408
        N_down = shared_expert_down_weight.shape[1]  # 1408 (unused directly, used via activated_pre)
        H_out = 4096  # output hidden size

        # 1) Compute shared_gate_output = F.linear(hidden, shared_expert_gate_weight)
        gate_out_f32 = torch.empty((B, N_gate), dtype=torch.float32, device=hidden_states.device)
        grid_gate = (B, N_gate)
        gemv_n1408_kernel[grid_gate](
            hidden_states, shared_expert_gate_weight, gate_out_f32,
            B, H,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            gate_out_f32.stride(0), gate_out_f32.stride(1),
            BLOCK_H=256,
            num_warps=4,
            num_stages=2,
        )

        # 2) Compute shared_up_output = F.linear(hidden, shared_expert_up_weight)
        up_out_f32 = torch.empty((B, N_up), dtype=torch.float32, device=hidden_states.device)
        grid_up = (B, N_up)
        gemv_n1408_kernel[grid_up](
            hidden_states, shared_expert_up_weight, up_out_f32,
            B, H,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            up_out_f32.stride(0), up_out_f32.stride(1),
            BLOCK_H=256,
            num_warps=4,
            num_stages=2,
        )

        # 3) Compute activated_pre = SiLU(gate_out) * up_out (elementwise)
        silu_gate_f32 = torch.empty((B, N_gate), dtype=torch.float32, device=hidden_states.device)
        N_elements_silu = B * N_gate
        silu_elemwise_kernel[(N_elements_silu + 1023) // 1024,](
            gate_out_f32, silu_gate_f32, N_elements_silu, 1024,
            num_warps=2, num_stages=2,
        )

        activated_pre_f32 = torch.empty((B, N_up), dtype=torch.float32, device=hidden_states.device)
        mul_elemwise_kernel[(B * N_up + 1023) // 1024,](
            silu_gate_f32, up_out_f32, activated_pre_f32, B * N_up, 1024,
            num_warps=2, num_stages=2,
        )

        # 4) Compute shared_activated = F.linear(activated_pre, shared_expert_down_weight)
        # activated_pre_f32: [B, 1408]; down_weight: [4096, 1408]
        shared_activated_f32 = torch.empty((B, H_out), dtype=torch.float32, device=hidden_states.device)
        grid_down = (B, H_out)
        gemv_n4096_kernel[grid_down](
            activated_pre_f32, shared_expert_down_weight, shared_activated_f32,
            B, N_up, H_out,
            activated_pre_f32.stride(0), activated_pre_f32.stride(1),
            shared_expert_down_weight.stride(0), shared_expert_down_weight.stride(1),
            shared_activated_f32.stride(0), shared_activated_f32.stride(1),
            BLOCK_N=256,
            num_warps=4,
            num_stages=2,
        )

        # Return exactly the three outputs, cast to bfloat16 to match original dtypes
        return (
            gate_out_f32.to(torch.bfloat16),
            up_out_f32.to(torch.bfloat16),
            shared_activated_f32.to(torch.bfloat16),
        )


def run(*args):
    return ModelNew()(*args)
