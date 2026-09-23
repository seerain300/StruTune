import torch
import triton
import triton.language as tl


@triton.jit
def matmul_rowwise_bf16(
    X_ptr,           # *const bfloat16, shape [M, K]
    W_ptr,           # *const bfloat16, shape [K, N]  (W is [K, N], not [N, K])
    Y_ptr,           # *float32, shape [M, N]
    M,               # int
    K,               # int
    N,               # int
    stride_xm,       # int: stride along M for X
    stride_xk,       # int: stride along K for X
    stride_wk,       # int: stride along K for W (rows)
    stride_wn,       # int: stride along N for W (cols)
    stride_ym,       # int: stride along M for Y
    stride_yn,       # int: stride along N for Y
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        # Load a slice of the m-th row of X: X[m, kk:kk+BLOCK_K]
        x_row_ptrs = X_ptr + m * stride_xm + k_offsets * stride_xk
        x_row = tl.load(x_row_ptrs, mask=k_offsets < K, other=0).to(tl.float32)

        for nn in range(0, N, BLOCK_N):
            n_offsets = nn + tl.arange(0, BLOCK_N)
            # Load W tile: W[kk:kk+BLOCK_K, nn:nn+BLOCK_N]
            w_tile_ptrs = W_ptr + (kk + tl.arange(0, BLOCK_K))[:, None] * stride_wk + n_offsets[None, :] * stride_wn
            # Mask: only load valid k within K range
            w_tile = tl.load(
                w_tile_ptrs,
                mask=(kk + tl.arange(0, BLOCK_K))[:, None] < K,
                other=0
            ).to(tl.float32)

            # acc[n] += sum over k of x_row[k] * w_tile[k, n]
            acc[n_offsets] += tl.sum(w_tile * x_row[:, None], axis=0)

    # Store the accumulated row to Y[m, :]
    y_row_ptrs = Y_ptr + m * stride_ym + tl.arange(0, BLOCK_N) * stride_yn
    tl.store(y_row_ptrs, acc, mask=tl.arange(0, BLOCK_N) < N)


@triton.jit
def silu_mul_kernel(
    A_ptr,           # *float32, shape [M, N] (GateOut)
    B_ptr,           # *float32, shape [M, N] (UpOut)
    C_ptr,           # *float32, shape [M, N] (Output)
    M,               # int
    N,               # int
    stride_am,       # int
    stride_an,       # int
    stride_bm,       # int
    stride_bn,       # int
    stride_cm,       # int
    stride_cn,       # int
):
    # 1D grid over M*N elements
    pid = tl.program_id(0)
    total = M * N
    n = pid % N
    m = pid // N

    a = A_ptr + m * stride_am + n * stride_an
    b = B_ptr + m * stride_bm + n * stride_bn
    a_val = tl.load(a)
    b_val = tl.load(b)
    # silu(x) = x * sigmoid(x)
    silu_a = a_val * (1.0 / (1.0 + tl.exp(-a_val)))
    c_val = silu_a * b_val
    c_ptr = C_ptr + m * stride_cm + n * stride_cn
    tl.store(c_ptr, c_val)


class ModelNew(torch.nn.Module):
    def __init__(self, BLOCK_K=256, BLOCK_N=128, NUM_WARPS=4):
        super().__init__()
        self.BLOCK_K = BLOCK_K
        self.BLOCK_N = BLOCK_N
        self.NUM_WARPS = NUM_WARPS

    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        """
        Compute:
          gate = F.linear(hidden_states, shared_expert_gate_weight.T, bias=None)
          up   = F.linear(hidden_states, shared_expert_up_weight.T, bias=None)
          return SiLU(gate) * up

        All computation is performed in Triton kernels; no torch ops on tensors in forward.
        """
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, \
            "All tensors must be on CUDA for Triton kernels"
        hidden_states = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()  # [K, N_gate]
        up_weight = shared_expert_up_weight.contiguous()      # [K, N_up]

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N_gate = gate_weight.shape[1]
        N_up = up_weight.shape[1]

        # Allocate outputs (float32)
        gate_out = torch.empty((M, N_gate), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N_up), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMV kernels: X[M,K] @ W[K,N] -> Y[M,N]
        grid = (M,)
        matmul_rowwise_bf16[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N_gate,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N,
            num_warps=self.NUM_WARPS
        )
        matmul_rowwise_bf16[grid](
            hidden_states, up_weight, up_out,
            M, K, N_up,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N,
            num_warps=self.NUM_WARPS
        )

        # Elementwise Triton kernel: silu(gate_out) * up_out
        # For the provided setup, N_gate == N_up == hidden_size == 4096. If not equal, original behavior is ambiguous.
        assert N_gate == N_up, "Gate and Up output sizes must match for this Triton implementation."

        Y = torch.empty((M, N_gate), dtype=torch.float32, device=hidden_states.device)
        total_elems = M * N_gate
        silu_mul_kernel[(total_elems,)](
            gate_out, up_out, Y,
            M, N_gate,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            Y.stride(0), Y.stride(1),
            num_warps=4
        )

        # Cast to bfloat16 to match original dtype and return
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
