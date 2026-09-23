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


# GEMM: C[M, N] = A[M, K] @ B[K, N]
# We'll use this to compute shared_activated: A=[B, N], B=[N, H], C=[B, H]
@triton.jit
def matmul_kernel(
    A_ptr,  # *bf16, [M, K]
    B_ptr,  # *bf16, [K, N]
    C_ptr,  # *f32,  [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_A_m, stride_A_k,
    stride_B_k, stride_B_n,
    stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_A_m + offs_k[None, :] * stride_A_k,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_B_k + offs_n[None, :] * stride_B_n,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b)
    tl.store(
        C_ptr + offs_m[:, None] * stride_C_m + offs_n[None, :] * stride_C_n,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


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
        shared_expert_up_weight: torch.Tensor,    # [H, N_up]    = [4096, 1408], bf16
        shared_expert_down_weight: torch.Tensor,  # [H, N_down]  = [4096, 1408], bf16
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        # This function performs all forward math using Triton kernels.
        # It returns: (shared_gate_output, shared_up_output, shared_activated)
        # shared_gate_output: [B, N_gate] bf16
        # shared_up_output:   [B, N_up]   bf16
        # shared_activated:   [B, H]      bf16
        device = hidden_states.device
        B = hidden_states.shape[0]
        H = hidden_states.shape[1]
        N_gate = shared_expert_gate_weight.shape[1]
        N_up = shared_expert_up_weight.shape[1]
        N_down = shared_expert_down_weight.shape[1]

        # Ensure inputs are contiguous for Triton
        hidden = hidden_states.contiguous()                      # [B, H], bf16
        gate   = shared_expert_gate_weight.contiguous()         # [H, N_gate], bf16
        up     = shared_expert_up_weight.contiguous()           # [H, N_up], bf16
        down   = shared_expert_down_weight.contiguous()         # [H, N_down], bf16

        # Allocate outputs (float32 for numerical stability, will cast to bf16 at end)
        y_gate = torch.empty((B, N_gate), dtype=torch.float32, device=device)  # [B, N_gate]
        y_up   = torch.empty((B, N_up),   dtype=torch.float32, device=device)  # [B, N_up]

        # Launch GEMV for gate and up: 2D grids (B, N_gate) and (B, N_up)
        gemv_linear_kernel[(B, N_gate)](
            hidden, gate, y_gate,
            B, H, N_gate,
            hidden.stride(0), hidden.stride(1),
            gate.stride(0), gate.stride(1),
            y_gate.stride(0), y_gate.stride(1),
            128,
            num_warps=4,
        )
        gemv_linear_kernel[(B, N_up)](
            hidden, up, y_up,
            B, H, N_up,
            hidden.stride(0), hidden.stride(1),
            up.stride(0), up.stride(1),
            y_up.stride(0), y_up.stride(1),
            128,
            num_warps=4,
        )

        # SiLU on gate in float32: silu_gate = y_gate * sigmoid(y_gate)
        silu_gate = torch.empty((B, N_gate), dtype=torch.float32, device=device)  # [B, N_gate], f32
        N_elements_silu = B * N_gate
        BLOCK_SILU = 1024
        silu_elemwise_kernel[(triton.cdiv(N_elements_silu, BLOCK_SILU),)](
            y_gate, silu_gate, N_elements_silu, BLOCK_SILU,
            num_warps=4,
        )

        # Multiply: activated_pre = silu_gate * y_up
        activated_pre = torch.empty((B, N_up), dtype=torch.float32, device=device)  # [B, N_up], f32
        N_elements_mul = B * N_up
        mul_elemwise_kernel[(triton.cdiv(N_elements_mul, BLOCK_SILU),)](
            silu_gate, y_up, activated_pre, N_elements_mul, BLOCK_SILU,
            num_warps=4,
        )

        # Down projection: shared_activated[b, h] = sum_t activated_pre[b, t] * down[h, t]
        # A: [B, N_down] = activated_pre, B: [N_down, H] = down^T, C: [B, H]
        A = activated_pre.contiguous()                         # [B, N_down], f32
        down_T = down.transpose(0, 1).contiguous()            # [N_down, H], bf16
        shared_activated_f32 = torch.empty((B, H), dtype=torch.float32, device=device)  # [B, H], f32

        # Launch matmul: (B, N_down) @ (N_down, H) -> (B, H)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        matmul_kernel[(triton.cdiv(B, BLOCK_M), triton.cdiv(H, BLOCK_N))](  # grid over B and H tiles
            A, down_T, shared_activated_f32,
            B, H, N_down,
            A.stride(0), A.stride(1),
            down_T.stride(0), down_T.stride(1),
            shared_activated_f32.stride(0), shared_activated_f32.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4,
        )

        # Return outputs matching original: (shared_gate_output, shared_up_output, shared_activated)
        # Cast to bfloat16 to match original dtype.
        shared_gate_output = y_gate.to(torch.bfloat16)        # [B, N_gate], bf16
        shared_up_output   = y_up.to(torch.bfloat16)          # [B, N_up],   bf16
        shared_activated   = shared_activated_f32.to(torch.bfloat16)  # [B, H], bf16

        return shared_gate_output, shared_up_output, shared_activated


def run(*args):
    return ModelNew()(*args)
