import torch
import triton
import triton.language as tl


# GEMV kernel: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H] (bf16), W: [N, H] (bf16), y: [B, N] (bf16)
@triton.jit
def gemv_linear_bf16_kernel(
    hidden_ptr,   # *bf16, [B, H]
    W_ptr,        # *bf16, [N, H]
    y_ptr,        # *bf16, [B, N]
    B, H, N,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_e = tl.program_id(1)  # output index in W (gate or up)
    acc = tl.zeros((), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    # Store as bfloat16
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc.to(tl.bfloat16))


# Elementwise SiLU and multiply: pre = silu(gate) * up
# gate: [B, N] (bf16), up: [B, N] (bf16), output pre_flat: [B*N] (f32)
@triton.jit
def silu_mul_elemwise_kernel(gate_ptr, up_ptr, out_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)  # gate is bf16, cast to f32 for stable math
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)    # up is bf16, cast to f32
    sig = 1.0 / (1.0 + tl.exp(-g))
    y = g * sig * u
    tl.store(out_ptr + offs, y, mask=mask)


# Down GEMV: y[b, h] = sum_t pre[b, t] * down[h, t]
# pre_flat: [B*N] (f32), down: [H, N] (bf16), y: [B, H] (f32)
@triton.jit
def down_gemv_kernel(
    pre_ptr,      # *f32,  [B, N] flattened (we pass pre_flat of length B*N)
    down_ptr,     # *bf16, [H, N]
    y_ptr,        # *f32,  [B, H]
    B, H, N,
    stride_down_h, stride_down_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # hidden index
    acc = tl.zeros((), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        # Map linear index to (b, n): idx = b*N + n
        b_idx = pid_b
        n_idx = offs_n
        pre_vals = tl.load(pre_ptr + b_idx * N + n_idx, mask=mask_n, other=0.0)  # pre_ptr is f32
        down_vals = tl.load(down_ptr + pid_h * stride_down_h + n_idx * stride_down_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(pre_vals * down_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self,
        grad_output: torch.Tensor,
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
        """
        Triton-only forward. Computes:
        - shared_gate_output = F.linear(hidden, shared_expert_gate_weight)  [B, 1408], bf16
        - shared_up_output    = F.linear(hidden, shared_expert_up_weight)   [B, 1408], bf16
        - shared_activated    = F.linear(silu(shared_gate_output) * shared_up_output, shared_expert_down_weight)  [B, 4096], bf16

        Returns (shared_gate_output, shared_up_output, shared_activated).
        """
        # Ensure inputs are on CUDA
        assert hidden_states.is_cuda, "hidden_states must be on CUDA for Triton kernels."
        assert shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda and shared_expert_down_weight.is_cuda, "All weight tensors must be on CUDA."

        B = hidden_states.shape[0]
        H = hidden_states.shape[1]
        N_gate = shared_expert_gate_weight.shape[1]  # 1408
        N_up = shared_expert_up_weight.shape[1]      # 1408
        H_out = shared_expert_down_weight.shape[0]   # 4096

        # 1) Gate GEMV: gate_output[b, t] = sum_h hidden[b, h] * gate_weight[t, h]
        gate_output = torch.empty((B, N_gate), dtype=torch.bfloat16, device=hidden_states.device)
        grid_gate = (B, N_gate)
        gemv_linear_bf16_kernel[grid_gate](
            hidden_states, shared_expert_gate_weight, gate_output,
            B, H, N_gate,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_H=256, num_warps=4
        )

        # 2) Up GEMV: up_output[b, t] = sum_h hidden[b, h] * up_weight[t, h]
        up_output = torch.empty((B, N_up), dtype=torch.bfloat16, device=hidden_states.device)
        grid_up = (B, N_up)
        gemv_linear_bf16_kernel[grid_up](
            hidden_states, shared_expert_up_weight, up_output,
            B, H, N_up,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_H=256, num_warps=4
        )

        # 3) Elementwise SiLU and multiply into flat vector (f32)
        N_elements = B * N_gate  # N_gate == N_up == 1408
        pre_flat = torch.empty(N_elements, dtype=torch.float32, device=hidden_states.device)
        grid_elem = (triton.cdiv(N_elements, 1024),)
        silu_mul_elemwise_kernel[grid_elem](
            gate_output.contiguous().view(-1), up_output.contiguous().view(-1), pre_flat,
            N_elements, BLOCK=1024
        )

        # 4) Down GEMV: activated[b, h] = sum_t pre[b, t] * down[h, t]
        activated = torch.empty((B, H_out), dtype=torch.float32, device=hidden_states.device)
        grid_down = (B, H_out)
        down_gemv_kernel[grid_down](
            pre_flat, shared_expert_down_weight,
            activated,
            B, H_out, N_gate,
            shared_expert_down_weight.stride(0), shared_expert_down_weight.stride(1),
            activated.stride(0), activated.stride(1),
            BLOCK_N=256, num_warps=4
        )
        # Cast to bfloat16 to match original output dtype
        activated = activated.to(torch.bfloat16)

        return gate_output, up_output, activated


def run(*args):
    return ModelNew()(*args)
