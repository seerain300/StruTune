import torch
import triton
import triton.language as tl


# Triton GEMV: y[b, e] = sum_h hidden[b, h] * W[e, h]
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
    BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)  # batch row
    pid_e = tl.program_id(1)  # expert index
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


# Triton elementwise multiply: y = a * b, operate on flat vectors (f32)
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# Triton GEMV down projection:
# y[b, h] = sum_t pre_down[b, t] * down[h, t]
# pre_down: [B, K] (f32), down: [H, K] (bf16), y: [B, H] (f32)
@triton.jit
def down_gemv_kernel(
    pre_down_ptr,  # *f32, [B, K]
    down_ptr,      # *bf16, [H, K]
    y_ptr,         # *f32, [B, H]
    B: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    stride_pd_b, stride_pd_k,
    stride_dn_h, stride_dn_k,
    stride_y_b, stride_y_h,
    BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # output hidden index
    acc = 0.0
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        pd_vals = tl.load(pre_down_ptr + pid_b * stride_pd_b + offs_k * stride_pd_k, mask=mask_k, other=0.0)
        dn_vals = tl.load(down_ptr + pid_h * stride_dn_h + offs_k * stride_dn_k, mask=mask_k, other=0.0).to(tl.float32)
        acc += tl.sum(pd_vals * dn_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,
        shared_expert_up_weight: torch.Tensor,
        shared_expert_down_weight: torch.Tensor,
    ):
        """
        Accepts the same 7 inputs as the original Model.forward and returns:
        - shared_gate_output: [B, 1408] (bf16)
        - shared_up_output:   [B, 1408] (bf16)
        - shared_activated:   [B, 4096] (bf16)
        No torch device-side ops in forward.
        """
        # Ensure contiguous for predictable strides
        hidden = hidden_states.contiguous()          # [B, H]
        gate = shared_expert_gate_weight.contiguous() # [N_gate=1408, H=4096]
        up = shared_expert_up_weight.contiguous()    # [N_up=1408, H=4096]
        down = shared_expert_down_weight.contiguous()# [H_out=4096, K=1408]

        B, H = hidden.shape
        N_gate = gate.shape[0]
        N_up = up.shape[0]
        assert N_gate == N_up, "gate and up must have same number of experts"
        K = down.shape[1]  # 1408

        # 1) Compute shared_gate_output via GEMV: [B, N_gate] in f32
        gate_output_f32 = torch.empty((B, N_gate), dtype=torch.float32, device=hidden.device)
        grid_gate = (B, N_gate)
        gemv_linear_kernel[grid_gate](
            hidden, gate, gate_output_f32,
            B=B, H=H, N=N_gate,
            stride_h_b=hidden.stride(0), stride_h_h=hidden.stride(1),
            stride_W_e=gate.stride(0), stride_W_h=gate.stride(1),
            stride_y_b=gate_output_f32.stride(0), stride_y_e=gate_output_f32.stride(1),
            BLOCK_H=256,
        )

        # 2) Compute shared_up_output via GEMV: [B, N_up] in f32
        up_output_f32 = torch.empty((B, N_up), dtype=torch.float32, device=hidden.device)
        grid_up = (B, N_up)
        gemv_linear_kernel[grid_up](
            hidden, up, up_output_f32,
            B=B, H=H, N=N_up,
            stride_h_b=hidden.stride(0), stride_h_h=hidden.stride(1),
            stride_W_e=up.stride(0), stride_W_h=up.stride(1),
            stride_y_b=up_output_f32.stride(0), stride_y_e=up_output_f32.stride(1),
            BLOCK_H=256,
        )

        # 3) Compute silu(gate_output) in f32 via Triton elementwise kernel
        silu_gate = torch.empty((B, N_gate), dtype=torch.float32, device=hidden.device)
        silu_elemwise_kernel[(triton.cdiv(B * N_gate, 1024),)](
            gate_output_f32.view(-1), silu_gate.view(-1),
            N_elements=B * N_gate, BLOCK=1024
        )

        # 4) Compute pre_down = silu_gate * up_output in f32 via Triton elementwise kernel
        pre_down = torch.empty((B, N_gate), dtype=torch.float32, device=hidden.device)
        mul_elemwise_kernel[(triton.cdiv(B * N_gate, 1024),)](
            silu_gate.view(-1), up_output_f32.view(-1), pre_down.view(-1),
            N_elements=B * N_gate, BLOCK=1024
        )

        # 5) Compute activated = down(pre_down) via GEMV: [B, H] in f32
        activated_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
        grid_down = (B, H)
        down_gemv_kernel[grid_down](
            pre_down, down, activated_f32,
            B=B, K=K, H=H,
            stride_pd_b=pre_down.stride(0), stride_pd_k=pre_down.stride(1),
            stride_dn_h=down.stride(0), stride_dn_k=down.stride(1),
            stride_y_b=activated_f32.stride(0), stride_y_h=activated_f32.stride(1),
            BLOCK_K=128,
        )

        # Cast to bfloat16 to match original
        shared_gate_output = gate_output_f32.to(torch.bfloat16)   # [B, 1408]
        shared_up_output   = up_output_f32.to(torch.bfloat16)     # [B, 1408]
        shared_activated   = activated_f32.to(torch.bfloat16)     # [B, 4096]

        return shared_gate_output, shared_up_output, shared_activated


def run(*args):
    return ModelNew()(*args)
