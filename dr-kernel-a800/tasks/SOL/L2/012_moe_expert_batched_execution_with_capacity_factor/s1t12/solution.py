import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_forward(A_ptr, B_ptr, Y_ptr,
                        M, K, N,
                        A_stride_m, A_stride_k,
                        B_stride_k, B_stride_n,
                        Y_stride_m, Y_stride_n,
                        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # Y = A @ B
    # A: [M, K], B: [K, N], Y: [M, N]
    # In this implementation, M is typically 1, but we keep it generic.
    pid_n = tl.program_id(0)  # tile id along N
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: [M, BLOCK_K]
        a_ptrs = A_ptr + 0 * A_stride_m + offs_k[None, :] * A_stride_k  # M=1 -> 0*A_stride_m
        # Load B tile: [BLOCK_K, N]
        b_ptrs = B_ptr + offs_k[:, None] * B_stride_k + offs_n[None, :] * B_stride_n

        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # Accumulate
        acc += tl.dot(a, b)

    # Store Y: [M, N]
    y_ptrs = Y_ptr + 0 * Y_stride_m + offs_n[None, :] * Y_stride_n  # M=1
    tl.store(y_ptrs, acc, mask=(offs_n[None, :] < N))


@triton.jit
def silu_mul_triton(G_ptr, U_ptr, OUT_ptr,
                    M, N,
                    G_stride_m, U_stride_m, OUT_stride_m,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # OUT = silu(G) * U, G, U: [M, N], OUT: [M, N]
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        nn = n0 + offs_n
        g_ptrs = G_ptr + offs_m[:, None] * G_stride_m + nn[None, :] * 0  # N=1 -> stride 0
        u_ptrs = U_ptr + offs_m[:, None] * U_stride_m + nn[None, :] * 0
        g = tl.load(g_ptrs, mask=(offs_m[:, None] < M) & (nn[None, :] < N), other=0.0)
        u = tl.load(u_ptrs, mask=(offs_m[:, None] < M) & (nn[None, :] < N), other=0.0)
        # silu(x) = x * sigmoid(x)
        silu_g = g * (tl.sigmoid(g) * 0.5)
        acc += silu_g * u

    out_ptrs = OUT_ptr + offs_m[:, None] * OUT_stride_m + 0 * 0  # M=1
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (0 * 0 < N))


@triton.jit
def atomic_accumulate_triton(SCALAR_ptr, Y_ptr, OUT_ptr, ROW,
                             Y_stride_m, Y_stride_n, OUT_stride_m, OUT_stride_n,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Y: [1, N], OUT: [num_tokens, hidden_size]
    # result[ROW, :] += scalar * Y[0, :]
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scalar = tl.load(SCALAR_ptr)  # float32 scalar
    y_ptrs = Y_ptr + 0 * Y_stride_m + offs_n[None, :] * Y_stride_n
    y = tl.load(y_ptrs, mask=(offs_n[None, :] < Y.shape[1]), other=0.0)
    contrib = y * scalar
    out_ptrs = OUT_ptr + ROW * OUT_stride_m + offs_n[None, :] * OUT_stride_n
    tl.atomic_add(out_ptrs, contrib, mask=(offs_n[None, :] < OUT.shape[1]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        Triton-only forward that reproduces the original computation:
        - hidden_states: [num_tokens, hidden_size], bfloat16, CUDA
        - selected_experts: [num_tokens, num_experts_per_tok], int64, CUDA
        - routing_weights: [num_tokens, num_experts_per_tok], bfloat16, CUDA
        - expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size], bfloat16, CUDA
        - expert_up_weights: [num_experts, hidden_size, moe_intermediate_size], bfloat16, CUDA
        - expert_down_weights: [num_experts, moe_intermediate_size, hidden_size], bfloat16, CUDA
        Returns: [num_tokens, hidden_size], bfloat16
        """
        # Validate device
        assert hidden_states.is_cuda, "hidden_states must be on CUDA"
        assert selected_experts.is_cuda and routing_weights.is_cuda, "selected_experts and routing_weights must be on CUDA"
        assert expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, "All expert weights must be on CUDA"

        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, H_in, H_out = expert_gate_weights.shape  # H_in = hidden_size, H_out = moe_intermediate_size
        num_experts_per_tok = selected_experts.shape[1]

        # Accumulator in fp32 for numerical stability
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        # Iterate tokens and selected_experts; no torch operations on tensors
        for t in range(num_tokens):
            for j in range(num_experts_per_tok):
                expert_id = int(selected_experts[t, j].item())
                hidden_vec = hidden_states[t].contiguous()  # [hidden_size], bfloat16, contiguous

                # 1) gate_out = hidden_vec @ expert_gate_weights[expert_id] -> [H_out], fp32
                gate_out = torch.empty(1, H_out, dtype=torch.float32, device=hidden_states.device)
                _ = bmm_triton_forward(gate_out, hidden_vec, expert_gate_weights[expert_id],
                                        M=1, K=hidden_size, N=H_out,
                                        A_stride_m=1, A_stride_k=0,  # A is 1D vector -> strides
                                        B_stride_k=0, B_stride_n=1,  # expert_gate_weights[expert_id] is [K, N]
                                        Y_stride_m=1, Y_stride_n=0,  # gate_out is [1, N]
                                        BLOCK_M=1, BLOCK_K=128, BLOCK_N=128, num_warps=4, num_stages=2)

                # 2) up_out = hidden_vec @ expert_up_weights[expert_id] -> [H_out], fp32
                up_out = torch.empty(1, H_out, dtype=torch.float32, device=hidden_states.device)
                _ = bmm_triton_forward(up_out, hidden_vec, expert_up_weights[expert_id],
                                        M=1, K=hidden_size, N=H_out,
                                        A_stride_m=1, A_stride_k=0,
                                        B_stride_k=0, B_stride_n=1,
                                        Y_stride_m=1, Y_stride_n=0,
                                        BLOCK_M=1, BLOCK_K=128, BLOCK_N=128, num_warps=4, num_stages=2)

                # 3) activated = silu(gate_out) * up_out
                activated = torch.empty(1, H_out, dtype=torch.float32, device=hidden_states.device)
                _ = silu_mul_triton(activated, gate_out, up_out,
                                    M=1, N=H_out,
                                    G_stride_m=1, U_stride_m=1, OUT_stride_m=1,
                                    BLOCK_M=1, BLOCK_N=128, num_warps=2, num_stages=2)

                # 4) final_out = activated @ expert_down_weights[expert_id] -> [hidden_size], fp32
                final_out = torch.empty(1, hidden_size, dtype=torch.float32, device=hidden_states.device)
                _ = bmm_triton_forward(final_out, activated, expert_down_weights[expert_id],
                                        M=1, K=H_out, N=hidden_size,
                                        A_stride_m=1, A_stride_k=0,
                                        B_stride_k=0, B_stride_n=1,
                                        Y_stride_m=1, Y_stride_n=0,
                                        BLOCK_M=1, BLOCK_K=128, BLOCK_N=128, num_warps=4, num_stages=2)

                # 5) Accumulate: result[t] += routing_weights[t, j] * final_out
                weight = float(routing_weights[t, j].item())
                _ = atomic_accumulate_triton(weight, final_out, result, t,
                                             Y_stride_m=1, Y_stride_n=0,
                                             OUT_stride_m=1, OUT_stride_n=0,
                                             BLOCK_M=1, BLOCK_N=128, num_warps=2, num_stages=2)

        # Return result as bfloat16 (original inputs are bfloat16)
        return result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
