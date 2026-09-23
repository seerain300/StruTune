import torch
import triton
import triton.language as tl


# Triton GEMV kernel: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H], row-major, bf16; W: [N, H], bf16; y: [B, N], f32
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,    # *bf16, [B, H]
    W_ptr,         # *bf16, [N, H]
    y_ptr,         # *f32,  [B, N]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_e, stride_W_h,
    stride_y_b, stride_y_e,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row index
    pid_e = tl.program_id(1)  # output index in W (e = expert index / N)
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        # Load hidden row slice
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        # Load weight row slice
        w_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * w_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Triton elementwise SiLU: y = x * sigmoid(x), on flat buffers, f32 -> f32
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


# Triton elementwise multiply: y = a * b on flat buffers, f32 -> f32
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# Triton GEMV kernel for down projection: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
@triton.jit
def down_gemv_kernel(
    activated_ptr,  # *f32, [B, N] (pre-activated buffer)
    down_ptr,       # *bf16, [H, N] (row-major)
    y_ptr,          # *f32, [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_act_b, stride_act_t,
    stride_down_h, stride_down_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # output hidden index
    acc = 0.0
    for t_start in range(0, N, BLOCK_N):
        offs_t = t_start + tl.arange(0, BLOCK_N)
        mask_t = offs_t < N
        act_vals = tl.load(activated_ptr + pid_b * stride_act_b + offs_t * stride_act_t, mask=mask_t, other=0.0).to(tl.float32)
        down_vals = tl.load(down_ptr + pid_h * stride_down_h + offs_t * stride_down_n, mask=mask_t, other=0.0).to(tl.float32)
        acc += tl.sum(act_vals * down_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


def _launch_gemv(hidden: torch.Tensor, W: torch.Tensor, out: torch.Tensor):
    # hidden: [B, H], W: [N, H], out: [B, N], float32
    B, H = hidden.shape
    N = W.shape[0]
    grid = (B, N)
    # Choose tile size
    BLOCK_H = 128
    gemv_linear_kernel[grid](
        hidden, W, out,
        B, H, N,
        hidden.stride(0), hidden.stride(1),
        W.stride(0), W.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=2,
    )


def _launch_silu_mul(gate_out: torch.Tensor, up_out: torch.Tensor, activated_buf: torch.Tensor):
    # gate_out, up_out: [B, N], float32, device tensors
    B, N = gate_out.shape
    total = B * N
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    silu_elemwise_kernel[grid](gate_out, activated_buf, total, BLOCK, num_warps=4, num_stages=2)
    mul_elemwise_kernel[grid](activated_buf, up_out, activated_buf, total, BLOCK, num_warps=4, num_stages=2)


def _launch_down_gemv(activated_buf: torch.Tensor, down_weight: torch.Tensor, out: torch.Tensor):
    # activated_buf: [B, N], f32; down_weight: [H, N], bf16; out: [B, H], f32
    B, N = activated_buf.shape
    H = out.shape[1]
    grid = (B, H)
    BLOCK_N = 128
    down_gemv_kernel[grid](
        activated_buf, down_weight, out,
        B, H, N,
        activated_buf.stride(0), activated_buf.stride(1),
        down_weight.stride(0), down_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args correspond to: grad_output, hidden_states, router_weight, e_score_correction_bias, router_logits, scores, topk_indices, topk_weights, score_mask,
        #                    shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight,
        #                    shared_gate_output, shared_up_output, shared_activated
        # We only need hidden_states and shared_expert_* weights to compute outputs.
        grad_output, hidden_states, router_weight, e_score_correction_bias, router_logits, scores, topk_indices, topk_weights, score_mask, \
        shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight, \
        shared_gate_output, shared_up_output, shared_activated = args

        # Shapes (constants assumed from get_inputs)
        B = hidden_states.shape[0]
        H = hidden_states.shape[1]  # 4096
        N_gate = shared_expert_gate_weight.shape[0]  # 1408
        N_up = shared_expert_up_weight.shape[0]      # 1408
        N_down = shared_expert_down_weight.shape[1]  # 1408
        H_down = shared_expert_down_weight.shape[0]  # 4096

        # 1) Compute gate_output = linear(hidden, gate_weight) -> [B, N_gate], float32
        gate_output = torch.empty((B, N_gate), dtype=torch.float32, device=hidden_states.device)
        _launch_gemv(hidden_states, shared_expert_gate_weight, gate_output)

        # 2) Compute up_output = linear(hidden, up_weight) -> [B, N_up], float32
        up_output = torch.empty((B, N_up), dtype=torch.float32, device=hidden_states.device)
        _launch_gemv(hidden_states, shared_expert_up_weight, up_output)

        # 3) activated_pre = SiLU(gate_output) * up_output (elementwise, float32)
        activated_buf = torch.empty((B, N_gate), dtype=torch.float32, device=hidden_states.device)
        _launch_silu_mul(gate_output, up_output, activated_buf)

        # 4) shared_activated = linear(activated_pre, down_weight) -> [B, H_down], float32
        shared_activated_out = torch.empty((B, H_down), dtype=torch.float32, device=hidden_states.device)
        _launch_down_gemv(activated_buf, shared_expert_down_weight, shared_activated_out)

        # Cast to bfloat16 to match original outputs
        gate_out_bf = gate_output.to(torch.bfloat16)      # [B, 1408]
        up_out_bf = up_output.to(torch.bfloat16)          # [B, 1408]
        activated_bf = shared_activated_out.to(torch.bfloat16)  # [B, 4096]

        # Return exactly the three outputs
        return gate_out_bf, up_out_bf, activated_bf


def run(*args):
    return ModelNew()(*args)
