import torch
import triton
import triton.language as tl


# GEMV: y[b, e] = sum_h hidden[b, h] * W[e, h]
# hidden: [B, H] (pointer, read-only), W: [N, H] (pointer, read-only), y: [B, N] (pointer, write)
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,   # *f32/bf16, [B, H]
    W_ptr,        # *f32/bf16, [N, H]
    y_ptr,        # *f32,      [B, N]
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # hidden size (4096)
    N: tl.constexpr,  # output size (e.g., 1408)
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
        # Load hidden row slice and W row slice; cast to f32 for accumulation
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        w_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * w_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Elementwise SiLU: y = x * sigmoid(x), operate on a flat vector, output f32
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


# Elementwise multiply: y = a * b on flat vectors, both f32, output f32
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# GEMV for down projection: y[b, h] = sum_t activated_pre[b, t] * down[h, t]
# activated_pre: [B, N_down], down: [H, N_down], y: [B, H]
@triton.jit
def down_gemv_kernel(
    activated_ptr,  # *f32, [B, N_down]
    down_ptr,       # *f32/bf16, [H, N_down]
    y_ptr,          # *f32, [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    N_down: tl.constexpr,
    stride_act_b, stride_act_n,
    stride_down_h, stride_down_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # hidden output index
    acc = 0.0
    for n_start in range(0, N_down, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_down
        a_vals = tl.load(activated_ptr + pid_b * stride_act_b + offs_n * stride_act_n, mask=mask_n, other=0.0).to(tl.float32)
        d_vals = tl.load(down_ptr + pid_h * stride_down_h + offs_n * stride_down_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(a_vals * d_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Inputs provided by evaluator: hidden_states, gate_weight, up_weight, down_weight
        # We will launch Triton kernels for the heavy computation and return the three outputs.
        # The original forward returns:
        # - shared_gate_output: [B, 1408], bfloat16
        # - shared_up_output: [B, 1408], bfloat16
        # - shared_activated: [B, 4096], bfloat16
        # Here, we will compute exactly these outputs using Triton kernels.

        # Extract inputs
        hidden_states = args[0]  # [B, H], typically bfloat16
        gate_weight = args[3]    # [H, N_gate] = [4096, 1408], typically bfloat16
        up_weight = args[5]      # [H, N_up] = [4096, 1408], typically bfloat16
        down_weight = args[7]    # [H, N_down] = [4096, 1408], typically bfloat16

        B = hidden_states.shape[0]
        H = hidden_states.shape[1]
        N_gate = gate_weight.shape[1]  # 1408
        N_up = up_weight.shape[1]      # 1408
        N_down = down_weight.shape[1]  # 1408

        # Ensure contiguous tensors for correct strides
        hidden_c = hidden_states.contiguous()
        gate_c = gate_weight.contiguous()
        up_c = up_weight.contiguous()
        down_c = down_weight.contiguous()

        # 1) Compute gate_output: [B, N_gate], f32 via GEMV
        gate_out = torch.empty((B, N_gate), dtype=torch.float32, device=hidden_states.device)
        grid_gate = (B, N_gate)
        gemv_linear_kernel[grid_gate](
            hidden_c, gate_c, gate_out,
            B, H, N_gate,
            hidden_c.stride(0), hidden_c.stride(1),
            gate_c.stride(0), gate_c.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_H=256,
        )

        # 2) Compute up_output: [B, N_up], f32 via GEMV
        up_out = torch.empty((B, N_up), dtype=torch.float32, device=hidden_states.device)
        grid_up = (B, N_up)
        gemv_linear_kernel[grid_up](
            hidden_c, up_c, up_out,
            B, H, N_up,
            hidden_c.stride(0), hidden_c.stride(1),
            up_c.stride(0), up_c.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_H=256,
        )

        # 3) SiLU on gate_output: [B, N_gate], f32 via elementwise kernel
        silu_gate = torch.empty_like(gate_out)  # [B, N_gate], f32
        N_elements = gate_out.numel()
        BLOCK_SILU = 2048
        grid_silu = ((N_elements + BLOCK_SILU - 1) // BLOCK_SILU,)
        silu_elemwise_kernel[grid_silu](
            gate_out, silu_gate, N_elements, BLOCK_SILU
        )

        # 4) Multiply: activated_pre = silu_gate * up_out -> [B, N_up], f32 via elementwise kernel
        activated_pre = torch.empty((B, N_up), dtype=torch.float32, device=hidden_states.device)
        N_mul = up_out.numel()
        grid_mul = ((N_mul + BLOCK_SILU - 1) // BLOCK_SILU,)
        mul_elemwise_kernel[grid_mul](
            silu_gate, up_out, activated_pre, N_mul, BLOCK_SILU
        )

        # 5) Compute shared_activated: [B, H], f32 via down GEMV (note: this matches original path)
        shared_activated = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)
        grid_down = (B, H)
        down_gemv_kernel[grid_down](
            activated_pre, down_c, shared_activated,
            B, H, N_down,
            activated_pre.stride(0), activated_pre.stride(1),
            down_c.stride(0), down_c.stride(1),
            shared_activated.stride(0), shared_activated.stride(1),
            BLOCK_N=256,
        )

        # Cast outputs to bfloat16 to match original
        shared_gate_output = gate_out.to(torch.bfloat16)       # [B, 1408]
        shared_up_output = up_out.to(torch.bfloat16)           # [B, 1408]
        shared_activated = shared_activated.to(torch.bfloat16) # [B, 4096]

        return shared_gate_output, shared_up_output, shared_activated


def run(*args):
    return ModelNew()(*args)
