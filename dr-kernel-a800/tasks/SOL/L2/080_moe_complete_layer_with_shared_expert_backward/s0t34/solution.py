import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_2d_kernel(out_ptr, a_ptr, b_ptr,
                      M, N, K,
                      a_stride_m, a_stride_k,
                      b_stride_k, b_stride_n,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D launch: each program handles a tile of size (BLOCK_M, BLOCK_N) of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Pointers to A and B tiles
        a_ptrs = a_ptr + (offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k)
        b_ptrs = b_ptr + (offs_k[:, None] * b_stride_k + offs_n[None, :] * b_stride_n)

        # Masks for tails
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles (as float32), masked
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results
    out_ptrs = out_ptr + (offs_m[:, None] * N + offs_n[None, :])
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _silu_1d_kernel(out_ptr, in_ptr, count,
                    BLOCK: tl.constexpr):
    # 1D elementwise SiLU: out = x * sigmoid(x)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x))
    out = x * s
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _mul_1d_kernel(out_ptr, a_ptr, b_ptr, count,
                   BLOCK: tl.constexpr):
    # 1D elementwise multiply: out = a * b
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = a * b
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args correspond to: grad_output, hidden_states, router_weight, e_score_correction_bias, ...
        # We only need hidden_states, shared_expert_gate_weight, shared_expert_up_weight to compute
        # shared_activated = SiLU(gate_output) * up_output
        hidden_states = args[1]  # [M, K], dtype=bfloat16 or float32, device inferred
        shared_expert_gate_weight = args[6]  # [H, K], K=hidden_size, H=intermediate_size
        shared_expert_up_weight = args[7]   # [H, K]

        # Ensure all inputs are on the same device and contiguous
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        H = shared_expert_gate_weight.shape[0]

        # Gate output: gate_output = hidden @ gate_weight.T -> [M, H]
        gate_weight_T = shared_expert_gate_weight.transpose(0, 1).contiguous()  # [K, H]
        gate_output = torch.empty((M, H), dtype=torch.float32, device=device)
        grid_gate = (triton.cdiv(M, 64), triton.cdiv(H, 64))
        _matmul_2d_kernel[grid_gate](
            gate_output, hidden_states.to(torch.float32), gate_weight_T.to(torch.float32),
            M, H, K,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight_T.stride(0), gate_weight_T.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

        # Up output: up_output = hidden @ up_weight.T -> [M, H]
        up_weight_T = shared_expert_up_weight.transpose(0, 1).contiguous()  # [K, H]
        up_output = torch.empty((M, H), dtype=torch.float32, device=device)
        grid_up = (triton.cdiv(M, 64), triton.cdiv(H, 64))
        _matmul_2d_kernel[grid_up](
            up_output, hidden_states.to(torch.float32), up_weight_T.to(torch.float32),
            M, H, K,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight_T.stride(0), up_weight_T.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

        # SiLU(gate_output) in float32
        silu_out = torch.empty((M * H,), dtype=torch.float32, device=device)
        grid_silu = (triton.cdiv(M * H, 1024),)
        _silu_1d_kernel[grid_silu](silu_out, gate_output.view(-1), M * H, BLOCK=1024)

        # Multiply silu_out * up_output (both [M, H] flattened)
        final_flat = torch.empty((M * H,), dtype=torch.float32, device=device)
        grid_mul = (triton.cdiv(M * H, 1024),)
        _mul_1d_kernel[grid_mul](final_flat, silu_out, up_output.view(-1), M * H, BLOCK=1024)

        # Reshape and cast to bfloat16 for return
        final = final_flat.view(M, H).to(torch.bfloat16)
        return final


def run(*args):
    return ModelNew()(*args)
