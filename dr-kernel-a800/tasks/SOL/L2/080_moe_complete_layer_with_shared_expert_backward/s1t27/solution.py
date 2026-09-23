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


# Triton GEMV: y[b, h] = sum_t pre_vec[b, t] * down_vec[t, h]
# pre_vec: flattened vector of length B * N_down (f32), down_vec: [H, N_down] (bf16), y: [B, H] (f32)
@triton.jit
def down_gemv_kernel(
    pre_vec_ptr,   # *f32, flattened [B * N_down]
    down_ptr,      # *bf16, [H, N_down]
    y_ptr,         # *f32,  [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    N_down: tl.constexpr,
    stride_y_b, stride_y_h,
    stride_dn_h, stride_dn_d,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,  # iterate over N_down
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # hidden index
    acc = 0.0
    for t_start in range(0, N_down, BLOCK_K):
        offs_t = t_start + tl.arange(0, BLOCK_K)
        mask_t = offs_t < N_down
        # idx = b * N_down + t
        idx = pid_b * N_down + offs_t
        pre_vals = tl.load(pre_vec_ptr + idx, mask=mask_t, other=0.0)  # f32
        down_vals = tl.load(down_ptr + pid_h * stride_dn_h + offs_t * stride_dn_d, mask=mask_t, other=0.0).to(tl.float32)
        acc += tl.sum(pre_vals * down_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed sizes as in the original code
        self.H = 4096         # hidden_size
        self.N_gate = 1408    # gate expert weight rows
        self.N_up = 1408      # up expert weight rows
        self.N_down = 1408    # down expert weight rows (intermediate size)

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
        shared_expert_up_weight: torch.Tensor,    # [H, N_up] = [4096, 1408], bf16
        shared_expert_down_weight: torch.Tensor,  # [H, N_down] = [4096, 1408], bf16
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        """
        Triton-only forward: returns (shared_gate_output, shared_up_output, shared_activated).
        Shapes:
          - shared_gate_output: [B, 1408] (bf16)
          - shared_up_output:   [B, 1408] (bf16)
          - shared_activated:   [B, 4096] (bf16)
        """
        # Ensure inputs are contiguous
        hidden = hidden_states.contiguous()  # [B, H], bf16
        B = hidden.shape[0]
        H = self.H

        # 1) Compute gate = F.linear(hidden, shared_expert_gate_weight) => [B, N_gate], f32
        gate_out = torch.empty((B, self.N_gate), dtype=torch.float32, device=hidden.device)
        grid_gate = (B, self.N_gate)
        gemv_linear_kernel[grid_gate](
            hidden, shared_expert_gate_weight, gate_out,
            B, H, self.N_gate,
            hidden.stride(0), hidden.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_H=128,
            num_warps=4,
        )

        # 2) Compute up = F.linear(hidden, shared_expert_up_weight) => [B, N_up], f32
        up_out = torch.empty((B, self.N_up), dtype=torch.float32, device=hidden.device)
        grid_up = (B, self.N_up)
        gemv_linear_kernel[grid_up](
            hidden, shared_expert_up_weight, up_out,
            B, H, self.N_up,
            hidden.stride(0), hidden.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_H=128,
            num_warps=4,
        )

        # 3) SiLU(gate) elementwise, produce [B * N_gate] f32
        silu_gate = torch.empty((B * self.N_gate,), dtype=torch.float32, device=hidden.device)
        grid_silu = ((B * self.N_gate + 1023) // 1024,)
        silu_elemwise_kernel[grid_silu](
            gate_out.reshape(-1), silu_gate, B * self.N_gate, BLOCK=1024,
            num_warps=4,
        )

        # 4) elementwise multiply: activated_pre = SiLU(gate) * up_out => [B * N_up] f32
        mul_silu_up = torch.empty((B * self.N_up,), dtype=torch.float32, device=hidden.device)
        grid_mul = ((B * self.N_up + 1023) // 1024,)
        mul_elemwise_kernel[grid_mul](
            up_out.reshape(-1), silu_gate[:B * self.N_up], mul_silu_up, B * self.N_up, BLOCK=1024,
            num_warps=4,
        )

        # 5) Down projection: y_activated[b, h] = sum_t mul_silu_up[b, t] * down[h, t]
        y_activated = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
        grid_down = (B, H)
        down_gemv_kernel[grid_down](
            mul_silu_up, shared_expert_down_weight, y_activated,
            B, H, self.N_down,
            y_activated.stride(0), y_activated.stride(1),
            shared_expert_down_weight.stride(0), shared_expert_down_weight.stride(1),
            BLOCK_H=256, BLOCK_K=128,
            num_warps=8,
        )

        # Cast to bf16 to match original output dtypes
        gate_out_bf = gate_out.to(torch.bfloat16)   # [B, 1408]
        up_out_bf = up_out.to(torch.bfloat16)       # [B, 1408]
        activated_bf = y_activated.to(torch.bfloat16)  # [B, 4096]

        return gate_out_bf, up_out_bf, activated_bf


def run(*args):
    return ModelNew()(*args)
