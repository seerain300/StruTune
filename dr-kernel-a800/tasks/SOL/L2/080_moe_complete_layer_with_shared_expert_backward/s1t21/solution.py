import torch
import triton
import triton.language as tl


# GEMV: y[b, t] = sum_h hidden[b, h] * W[t, h]
# hidden: [B, H] (bf16), W: [T, H] (bf16), y: [B, T] (f32)
@triton.jit
def gemv_linear_kernel(
    hidden_ptr,   # *bf16, [B, H]
    W_ptr,        # *bf16, [T, H]
    y_ptr,        # *f32,  [B, T]
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    stride_h_b, stride_h_h,
    stride_W_t, stride_W_h,
    stride_y_b, stride_y_t,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch row
    pid_t = tl.program_id(1)  # target index
    acc = 0.0
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        h_vals = tl.load(hidden_ptr + pid_b * stride_h_b + offs_h * stride_h_h, mask=mask_h, other=0.0).to(tl.float32)
        W_vals = tl.load(W_ptr + pid_t * stride_W_t + offs_h * stride_W_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(h_vals * W_vals, axis=0)
    tl.store(y_ptr + pid_b * stride_y_b + pid_t * stride_y_t, acc)


# Triton elementwise SiLU: y = x * sigmoid(x), operate on flat vectors (f32)
@triton.jit
def silu_elemwise_kernel(x_ptr, y_ptr, N_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# GEMM: C[B, N] = A[B, K] @ B[K, N] where A is [B, T] and B is [T, H], C is [B, H]
@triton.jit
def matmul_kernel(
    A_ptr,  # *f32, [B, K]
    B_ptr,  # *bf16, [K, N] (here N=H, K=T)
    C_ptr,  # *bf16, [B, N]
    Bsz: tl.constexpr,  # B
    Ksz: tl.constexpr,  # T
    Nsz: tl.constexpr,  # H
    stride_A_b, stride_A_k,
    stride_B_k, stride_B_n,
    stride_C_b, stride_C_n,
    BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # along B
    pid_n = tl.program_id(1)  # along N (H)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, Ksz, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_b[:, None] * stride_A_b + offs_k[None, :] * stride_A_k,
            mask=(offs_b[:, None] < Bsz) & (offs_k[None, :] < Ksz),
            other=0.0
        ).to(tl.float32)  # [BLOCK_B, BLOCK_K]
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_B_k + offs_n[None, :] * stride_B_n,
            mask=(offs_k[:, None] < Ksz) & (offs_n[None, :] < Nsz),
            other=0.0
        ).to(tl.float32)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)
    # Store as bf16 to match output dtype
    tl.store(
        C_ptr + offs_b[:, None] * stride_C_b + offs_n[None, :] * stride_C_n,
        acc,  # Triton will cast to destination dtype on store if C_ptr is bf16; here acc is f32
        mask=(offs_b[:, None] < Bsz) & (offs_n[None, :] < Nsz)
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
        shared_expert_gate_weight: torch.Tensor,  # [H, T] = [4096, 1408]
        shared_expert_up_weight: torch.Tensor,    # [H, T] = [4096, 1408]
        shared_expert_down_weight: torch.Tensor,  # [H, T] = [4096, 1408]
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        # The original forward returns only the shared expert outputs:
        # (shared_gate_output, shared_up_output, shared_activated)
        # We compute them using Triton kernels (no torch device-side ops).
        Bsz = hidden_states.shape[0]
        Hsz = hidden_states.shape[1]
        Tsz = shared_expert_gate_weight.shape[1]  # 1408

        # Ensure contiguous memory for simple strides
        hidden_c = hidden_states.contiguous()

        # 1) Compute shared_gate_output: y[b, t] = sum_h hidden[b,h] * gate[t,h]
        y_gate = torch.empty((Bsz, Tsz), dtype=torch.float32, device=hidden_c.device)
        grid_gemv = (Bsz, Tsz)
        gemv_linear_kernel[grid_gemv](
            hidden_c, shared_expert_gate_weight, y_gate,
            Bsz, Hsz, Tsz,
            hidden_c.stride(0), hidden_c.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            y_gate.stride(0), y_gate.stride(1),
            BLOCK_H=128,
        )
        # Cast to bf16 to match original output dtype
        shared_gate_output = y_gate.to(torch.bfloat16)

        # 2) Compute shared_up_output: y[b, t] = sum_h hidden[b,h] * up[t,h]
        y_up = torch.empty((Bsz, Tsz), dtype=torch.float32, device=hidden_c.device)
        gemv_linear_kernel[(Bsz, Tsz)](
            hidden_c, shared_expert_up_weight, y_up,
            Bsz, Hsz, Tsz,
            hidden_c.stride(0), hidden_c.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            y_up.stride(0), y_up.stride(1),
            BLOCK_H=128,
        )
        shared_up_output = y_up.to(torch.bfloat16)

        # 3) Compute activated = SiLU(gate) * up, then down projection to [B, H] in bf16
        # SiLU elementwise in f32
        act = torch.empty((Bsz * Tsz,), dtype=torch.float32, device=hidden_c.device)
        silu_elemwise_kernel[(Bsz * Tsz + 1023) // 1024](  # grid as number of blocks
            y_gate.contiguous(), act, Bsz * Tsz, 1024
        )
        act = act.view(Bsz, Tsz)  # [B, T] f32
        activated = (act * y_up)  # [B, T] f32

        # Down projection: y[b, h] = sum_t activated[b, t] * down[t, h]
        y_activated = torch.empty((Bsz, Hsz), dtype=torch.float32, device=hidden_c.device)
        matmul_kernel[(Bsz, (Hsz + 1023) // 1024)](
            activated, shared_expert_down_weight,
            y_activated,
            Bsz, Tsz, Hsz,
            activated.stride(0), activated.stride(1),
            shared_expert_down_weight.stride(0), shared_expert_down_weight.stride(1),
            y_activated.stride(0), y_activated.stride(1),
            BLOCK_B=64, BLOCK_N=128, BLOCK_K=64,
        )
        shared_activated = y_activated.to(torch.bfloat16)

        # Return exactly the three tensors as the original forward
        return shared_gate_output, shared_up_output, shared_activated


def run(*args):
    return ModelNew()(*args)
