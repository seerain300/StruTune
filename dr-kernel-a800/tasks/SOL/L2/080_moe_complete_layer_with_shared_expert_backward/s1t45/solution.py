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
        # Load hidden row as bf16, cast to f32 for accumulation
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        # Load W row segment as bf16, cast to f32
        W_vals = tl.load(W_ptr + pid_e * stride_W_e + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        # Fused multiply-add and reduction
        acc += tl.sum(h_vals * W_vals, axis=0)
    # Store as f32
    tl.store(y_ptr + pid_b * stride_y_b + pid_e * stride_y_e, acc)


# Triton elementwise SiLU: y = x * sigmoid(x), operate on flat vectors (f32)
# Inputs: x_ptr [B*N], y_ptr [B*N]
@triton.jit
def silu_elemwise_kernel(x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Triton elementwise multiply: y = a * b on flat vectors, both f32
# Inputs: a_ptr [B*N], b_ptr [B*N], y_ptr [B*N]
@triton.jit
def mul_elemwise_kernel(a_ptr, b_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a * b
    tl.store(y_ptr + offs, y, mask=mask)


# GEMV-like reduction over N dimension: y[b, h] = sum_t A[b, t] * W[h, t]
# A: [B, N] (bf16), W: [H, N] (bf16), y: [B, H] (f32)
@triton.jit
def down_gemv_kernel(
    A_ptr,        # *bf16, [B, N]
    W_ptr,        # *bf16, [H, N]
    y_ptr,        # *f32,  [B, H]
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    stride_A_b, stride_A_n,
    stride_W_h, stride_W_n,
    stride_y_b, stride_y_h,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_h = tl.program_id(1)  # hidden index
    acc = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        A_vals = tl.load(A_ptr + pid_b * stride_A_b + offs_n * stride_A_n, mask=mask_n, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_h * stride_W_h + offs_n * stride_W_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(A_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_h * stride_y_h, acc)


def _gemv_launch(hidden, W, out):
    """
    Launch gemv_linear_kernel producing out[B, N] in float32.
    hidden: [B, H] (bf16), W: [N, H] (bf16), out: [B, N] (f32) preallocated.
    """
    B, H = hidden.shape
    N = W.shape[0]
    grid = (B, N)
    BLOCK_H = 128
    gemv_linear_kernel[grid](
        hidden, W, out,
        B, H, N,
        hidden.stride(0), hidden.stride(1),
        W.stride(0), W.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_H=BLOCK_H,
        num_warps=4,
    )


def _silu_elemwise(x):
    """
    x: [B*N] (device, float32)
    returns y: [B*N] (device, float32)
    """
    N_elements = x.numel()
    y = torch.empty_like(x, dtype=torch.float32, device=x.device)
    BLOCK = 1024
    grid = (triton.cdiv(N_elements, BLOCK),)
    silu_elemwise_kernel[grid](x, y, N_elements, BLOCK=BLOCK, num_warps=4)
    return y


def _mul_elemwise(a, b):
    """
    Elementwise a*b on flat vectors (float32), returns float32.
    a, b: device tensors of same shape, 1D flattened view
    """
    N_elements = a.numel()
    y = torch.empty_like(a, dtype=torch.float32, device=a.device)
    BLOCK = 1024
    grid = (triton.cdiv(N_elements, BLOCK),)
    mul_elemwise_kernel[grid](a, b, y, N_elements, BLOCK=BLOCK, num_warps=4)
    return y


def _down_gemv(A, W, out):
    """
    A: [B, N] (bf16), W: [H, N] (bf16), out: [B, H] (f32) preallocated
    Computes out[b, h] = sum_t A[b, t] * W[h, t].
    """
    B, N = A.shape
    H = W.shape[0]
    grid = (B, H)
    BLOCK_N = 128
    down_gemv_kernel[grid](
        A, W, out,
        B, H, N,
        A.stride(0), A.stride(1),
        W.stride(0), W.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight):
        """
        hidden_states: [B, H], bfloat16, device tensor
        shared_expert_gate_weight: [H, N_gate], bfloat16, device (N_gate=1408)
        shared_expert_up_weight:   [H, N_up],   bfloat16, device (N_up=1408)
        shared_expert_down_weight: [H, N_down], bfloat16, device (N_down=1408)
        Returns:
        - gate_output: [B, 1408], bfloat16
        - up_output:   [B, 1408], bfloat16
        - shared_activated: [B, 4096], bfloat16
        """
        B, H = hidden_states.shape
        N_GATE = 1408
        N_UP = 1408
        N_DOWN = 1408

        # 1) GEMV for gate and up (float32 outputs)
        gate_out_f32 = torch.empty((B, N_GATE), dtype=torch.float32, device=hidden_states.device)
        up_out_f32 = torch.empty((B, N_UP), dtype=torch.float32, device=hidden_states.device)
        _gemv_launch(hidden_states, shared_expert_gate_weight, gate_out_f32)
        _gemv_launch(hidden_states, shared_expert_up_weight, up_out_f32)

        # 2) Elementwise SiLU on gate_out_f32 (float32)
        silu_gate = _silu_elemwise(gate_out_f32.view(-1)).view(B, N_GATE)  # float32

        # 3) Elementwise multiply: activated_pre = silu_gate * up_out_f32 (float32)
        activated_pre = _mul_elemwise(silu_gate.view(-1), up_out_f32.view(-1))
        activated_pre = activated_pre.view(B, N_UP)  # float32

        # 4) Down reduction: shared_activated[b, h] = sum_t activated_pre[b, t] * down_weight[h, t]
        shared_activated_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)
        _down_gemv(activated_pre, shared_expert_down_weight, shared_activated_f32)

        # Return in bfloat16 as original model
        gate_out = gate_out_f32.to(torch.bfloat16)
        up_out = up_out_f32.to(torch.bfloat16)
        act_out = shared_activated_f32.to(torch.bfloat16)

        return gate_out, up_out, act_out


def run(*args):
    return ModelNew()(*args)
