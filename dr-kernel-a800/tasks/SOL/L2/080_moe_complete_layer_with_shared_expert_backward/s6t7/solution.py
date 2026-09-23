import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bf16_bf16_to_f32_kernel(
    X_ptr,           # *const bfloat16, shape [M, K]
    W_ptr,           # *const bfloat16, shape [K, N]  (note: W is [K, N], we index as W[k, n])
    Y_ptr,           # *float32, shape [M, N]
    M,               # int: rows of X
    K,               # int: cols of X / rows of W
    N,               # int: cols of W
    stride_xm,       # int: stride for X along M
    stride_xk,       # int: stride for X along K
    stride_wk,       # int: stride for W along K
    stride_wn,       # int: stride for W along N
    stride_ym,       # int: stride for Y along M
    stride_yn,       # int: stride for Y along N
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # compute row/col offsets for this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # loop over K dimension in tiles
    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)

        # load X tile: shape [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        x_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # load W tile: shape [BLOCK_K, BLOCK_N], W is [K, N], we access W[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # accumulate
        # x_tile: [BM, BK] bfloat16, w_tile: [BK, BN] bfloat16
        # acc += tl.dot(x_tile, w_tile) -> [BM, BN] float32
        acc += tl.dot(x_tile.to(tl.float32), w_tile.to(tl.float32))

    # store results
    y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,     # *const float32, shape [M, N]
    UpOut_ptr,       # *const float32, shape [M, N]
    Y_ptr,           # *float32, shape [M, N]
    M,               # int
    N,               # int
    stride_gm,       # int
    stride_gn,       # int
    stride_um,       # int
    stride_un,       # int
    stride_ym,       # int
    stride_yn,       # int
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets[:, None] < M
    n_mask = n_offsets[None, :] < N
    mask = m_mask & n_mask

    g_ptrs = GateOut_ptr + m_offsets[:, None] * stride_gm + n_offsets[None, :] * stride_gn
    u_ptrs = UpOut_ptr + m_offsets[:, None] * stride_um + n_offsets[None, :] * stride_un
    y_ptrs = Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn

    gate = tl.load(g_ptrs, mask=mask, other=0.0)
    up = tl.load(u_ptrs, mask=mask, other=0.0)

    # silu(x) = x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    y = gate * sig * up

    tl.store(y_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # tile sizes chosen for K,N around 4K/1.4K; can be tuned
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = 128

    def forward(self, hidden_states: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor):
        """
        Compute:
          gate_out = hidden_states @ gate_weight.T   # [M, N]
          up_out   = hidden_states @ up_weight.T     # [M, N]
          y        = silu(gate_out) * up_out
        Return y in bfloat16.
        All computation is done via Triton kernels; no torch ops on tensors.
        """
        assert hidden_states.is_cuda and gate_weight.is_cuda and up_weight.is_cuda, "Triton kernels require CUDA tensors"
        assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
        assert gate_weight.dtype == torch.bfloat16 and up_weight.dtype == torch.bfloat16, "Weights must be bfloat16"

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N_gate = gate_weight.shape[1]  # N
        N_up = up_weight.shape[1]      # N

        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        # Outputs as float32 (accumulate/store in float32)
        gate_out = torch.empty((M, N_gate), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N_up), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMM Triton kernels
        grid = (triton.cdiv(M, self.BLOCK_M), triton.cdiv(N_gate, self.BLOCK_N))
        matmul_bf16_bf16_to_f32_kernel[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N_gate,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=4, num_stages=3,
        )

        grid_up = (triton.cdiv(M, self.BLOCK_M), triton.cdiv(N_up, self.BLOCK_N))
        matmul_bf16_bf16_to_f32_kernel[grid_up](
            hidden_states, up_weight, up_out,
            M, K, N_up,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Elementwise activation and multiply in Triton
        # We compute silu(gate_out) * up_out and store to float32
        Y = torch.empty((M, N_gate), dtype=torch.float32, device=hidden_states.device)
        element_grid = (triton.cdiv(M, self.BLOCK_M), triton.cdiv(N_gate, self.BLOCK_N))
        silu_mul_kernel[element_grid](
            gate_out, up_out, Y,
            M, N_gate,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            Y.stride(0), Y.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Return in bfloat16 to match original example's dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
