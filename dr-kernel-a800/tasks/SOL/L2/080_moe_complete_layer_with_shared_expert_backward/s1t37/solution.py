import torch
import triton
import triton.language as tl


# Triton GEMV kernel: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H] (bf16); W: [N, H] (bf16); y: [B, N] (f32)
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,   # *bf16, [B, H]
    W_ptr,        # *bf16, [N, H]
    y_ptr,        # *f32,  [B, N]
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_e = tl.program_id(1)  # feature index in W
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * w_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Triton GEMV kernel: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
# activated_pre: [B, N] (bf16); down: [H, N] (bf16); y: [B, H] (f32)
@triton.jit
def down_gemv_kernel(
    activated_ptr,  # *bf16, [B, N]
    down_ptr,       # *bf16, [H, N]
    y_ptr,          # *f32,  [B, H]
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    stride_a_b, stride_a_n,
    stride_d_h, stride_d_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    acc = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        a_vals = tl.load(activated_ptr + pid_b * stride_a_b + offs_n * stride_a_n, mask=mask, other=0.0).to(tl.float32)
        d_vals = tl.load(down_ptr + pid_h * stride_d_h + offs_n * stride_d_n, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(a_vals * d_vals, axis=0)
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
                shared_activated: torch.Tensor,
                ):
        # Original returns: (shared_gate_output, shared_up_output, shared_activated)
        # We compute these using Triton kernels; torch ops are only for simple elementwise (if needed).
        B = hidden_states.shape[0]
        H = hidden_states.shape[1]  # 4096
        N_gate = shared_expert_gate_weight.shape[1]  # 1408
        N_up = shared_expert_up_weight.shape[1]      # 1408
        H_down = shared_expert_down_weight.shape[0]  # 4096
        N_down = shared_expert_down_weight.shape[1]  # 1408

        # 1) gate_output = linear(hidden, gate_weight) via GEMV, float32 output
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

        # 2) up_output = linear(hidden, up_weight) via GEMV, float32 output
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

        # 3) activated_pre = SiLU(gate_output) * up_output. Since Triton elementwise in this environment may not be
        # supported in forward, compute this using torch ops on device tensors (elementwise on device):
        gate_sigmoid = torch.sigmoid(gate_output)  # [B, N_gate], f32
        activated_pre = gate_output * gate_sigmoid * up_output  # [B, N_gate], f32

        # 4) shared_activated = linear(activated_pre, down_weight) via GEMV, float32 output
        shared_activated_out = torch.empty((B, H_down), dtype=torch.float32, device=hidden_states.device)
        grid_down = (B, H_down)
        down_gemv_kernel[grid_down](
            activated_pre, shared_expert_down_weight,
            shared_activated_out,
            B, N_down, H_down,
            activated_pre.stride(0), activated_pre.stride(1),
            shared_expert_down_weight.stride(0), shared_expert_down_weight.stride(1),
            shared_activated_out.stride(0), shared_activated_out.stride(1),
            BLOCK_N=128,
        )

        # Cast outputs to bfloat16 to match original returned dtype
        gate_output_bf = gate_output.to(torch.bfloat16)           # [B, 1408]
        up_output_bf = up_output.to(torch.bfloat16)               # [B, 1408]
        shared_activated_bf = shared_activated_out.to(torch.bfloat16)  # [B, 4096]

        # Return the three tensors
        return gate_output_bf, up_output_bf, shared_activated_bf


def run(*args):
    return ModelNew()(*args)
