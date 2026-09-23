import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,            # *bfloat16, shape [M, K]
    W_ptr,            # *bfloat16, shape [K, N]
    Y_ptr,            # *float32,  shape [M, N]
    M: tl.constexpr,  # int
    K: tl.constexpr,  # int
    N: tl.constexpr,  # int
    stride_xm,        # int
    stride_xk,        # int
    stride_wk,        # int
    stride_wn,        # int
    stride_ym,        # int
    stride_yn,        # int
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row m
    m = tl.program_id(0)
    # Initialize accumulator for the entire N dimension
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        # Vector of N indices for this program
        n_offsets = tl.arange(0, BLOCK_N)
        n_valid = n_offsets < N

        # Partial accumulation over this K-chunk
        partial = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Loop over kk within chunk
        for kk in range(0, BLOCK_K):
            k = k_start + kk
            k_valid = k < K
            # Load X[m, k] as scalar (bfloat16), promote to float32
            x_val = tl.load(X_ptr + m * stride_xm + k * stride_xk, mask=k_valid, other=0.0)
            x_val = x_val.to(tl.float32)

            # Load W[k, n_offsets] vector (bfloat16), mask invalid n
            w_ptrs = W_ptr + k * stride_wk + n_offsets * stride_wn
            w_vec = tl.load(w_ptrs, mask=n_valid, other=0.0)
            w_vec = w_vec.to(tl.float32)

            # FMA accumulate: partial += x_val * w_vec
            partial += x_val * w_vec

        # Accumulate partial into acc for valid n
        acc += tl.where(n_valid, partial, 0.0)

    # Store acc to Y[m, :]
    y_ptrs = Y_ptr + m * stride_ym + n_offsets * stride_yn
    tl.store(y_ptrs, acc, mask=n_valid)


@triton.jit
def silu_mul_kernel(
    A_ptr,            # *float32, GateOut, shape [M, N]
    B_ptr,            # *float32, UpOut,   shape [M, N]
    C_ptr,            # *float32, Output,  shape [M, N]
    M: tl.constexpr,  # int
    N: tl.constexpr,  # int
    stride_am,        # int
    stride_an,        # int
    stride_bm,        # int
    stride_bn,        # int
    stride_cm,        # int
    stride_cn,        # int
):
    # 2D grid: (M, N)
    m = tl.program_id(0)
    n = tl.program_id(1)
    # Bounds check
    m_valid = m < M
    n_valid = n < N

    # Compute pointers
    a_ptr = A_ptr + m * stride_am + n * stride_an
    b_ptr = B_ptr + m * stride_bm + n * stride_bn
    c_ptr = C_ptr + m * stride_cm + n * stride_cn

    # Load
    a = tl.load(a_ptr, mask=m_valid & n_valid, other=0.0)  # GateOut[m, n]
    b = tl.load(b_ptr, mask=m_valid & n_valid, other=0.0)  # UpOut[m, n]

    # y = a * sigmoid(a) * b
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-a))
    y = a * sig * b

    # Store
    tl.store(c_ptr, y, mask=m_valid & n_valid)


class ModelNew(torch.nn.Module):
    def __init__(self, K: int = 4096, N: int = 1408):
        super().__init__()
        self.K = K
        self.N = N
        # Tunable tile sizes; chosen for robustness
        self.BLOCK_K = 256
        self.BLOCK_N = 128

    def forward(self, hidden_states: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor):
        # hidden_states: [M, K] bfloat16
        # gate_weight, up_weight: [K, N] bfloat16
        assert hidden_states.dtype == torch.bfloat16
        assert gate_weight.dtype == torch.bfloat16 and up_weight.dtype == torch.bfloat16
        assert hidden_states.is_cuda and gate_weight.is_cuda and up_weight.is_cuda, "Tensors must be on CUDA"
        assert hidden_states.dim() == 2 and gate_weight.dim() == 2 and up_weight.dim() == 2
        M, K = hidden_states.shape
        K_w, N = gate_weight.shape
        assert K == K_w and N == self.N, "Input dimensions do not match expected K or N"

        # Ensure contiguous
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        # Compute gate_out and up_out using Triton GEMV row-wise kernels (float32 outputs)
        gate_out = torch.empty((M, self.N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, self.N), dtype=torch.float32, device=hidden_states.device)

        grid_gate = (M,)
        linear_rowwise_bf16_to_f32[grid_gate](
            hidden_states, gate_weight, gate_out,
            M, self.K, self.N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N,
            num_warps=4, num_stages=2
        )

        grid_up = (M,)
        linear_rowwise_bf16_to_f32[grid_up](
            hidden_states, up_weight, up_out,
            M, self.K, self.N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Compute shared_activated = gate_out * SiLU(gate_out) * up_out
        activated = torch.empty((M, self.N), dtype=torch.float32, device=hidden_states.device)
        grid = (M, self.N)
        silu_mul_kernel[grid](
            gate_out, up_out, activated,
            M, self.N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return bfloat16 tensor (final cast)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
