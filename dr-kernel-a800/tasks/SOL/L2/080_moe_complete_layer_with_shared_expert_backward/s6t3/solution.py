import torch
import triton
import triton.language as tl


@triton.jit
def matmul_rowwise_bf16(
    X_ptr,           # *const bfloat16, shape [M, K]
    W_ptr,           # *const bfloat16, shape [K, N]  (we pass weights as [K, N])
    Y_ptr,           # *float32, shape [M, N]
    M,               # int
    K,               # int
    N,               # int
    stride_xm,       # int (stride for X along M)
    stride_xk,       # int (stride for X along K)
    stride_wk,       # int (stride for W along K)
    stride_wn,       # int (stride for W along N)
    stride_ym,       # int (stride for Y along M)
    stride_yn,       # int (stride for Y along N)
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    # Accumulator over N tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over K in tiles
    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load X row slice: X[m, kk:kk+BLOCK_K]
        x_ptrs = X_ptr + m * stride_xm + k_offsets * stride_xk
        x_vals = tl.load(x_ptrs, mask=k_offsets < K, other=0.0)
        x_vals_f32 = x_vals.to(tl.float32)  # [BLOCK_K]

        # Accumulate acc += sum_{i in tile} x_vals[i] * W[kk+BLOCK_K, :]
        # W is [K, N]; we iterate over i in BLOCK_K, j in BLOCK_N
        for i in range(BLOCK_K):
            k_idx = kk + i
            if k_idx >= K:
                break
            n_offsets = tl.arange(0, BLOCK_N)
            w_ptrs = W_ptr + k_idx * stride_wk + n_offsets * stride_wn
            w_vals = tl.load(w_ptrs, mask=n_offsets < N, other=0.0)
            w_vals_f32 = w_vals.to(tl.float32)  # [BLOCK_N]
            acc += x_vals_f32[i] * w_vals_f32

    # Store the result row
    y_ptrs = Y_ptr + m * stride_ym + tl.arange(0, BLOCK_N) * stride_yn
    tl.store(y_ptrs, acc, mask=tl.arange(0, BLOCK_N) < N)


@triton.jit
def silu_mul_kernel(
    A_ptr,            # *const float32, GateOut[M, N]
    B_ptr,            # *const float32, UpOut[M, N]
    C_ptr,            # *float32, output [M, N] = A * sigmoid(A) * B
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
):
    total = M * N
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)  # 1D launch, 1024 elements per program
    mask = offsets < total

    # Compute 2D indices
    m = offsets // N
    n = offsets % N

    # Pointers for A and B
    a_ptrs = A_ptr + m * stride_am + n * stride_an
    b_ptrs = B_ptr + m * stride_bm + n * stride_bn

    a = tl.load(a_ptrs, mask=mask, other=0.0)  # gate_out
    b = tl.load(b_ptrs, mask=mask, other=0.0)  # up_out

    # silu(a) = a * sigmoid(a) = a / (1 + exp(-a))
    s = 1.0 / (1.0 + tl.exp(-a))
    y = a * s * b

    c_ptrs = C_ptr + m * stride_cm + n * stride_cn
    tl.store(c_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, BLOCK_K=256, BLOCK_N=128):
        super().__init__()
        self.BLOCK_K = BLOCK_K
        self.BLOCK_N = BLOCK_N

    def forward(self, *args):
        # Expect: hidden_states, gate_weight, up_weight
        # hidden_states: [M, hidden_size] bfloat16
        # gate_weight: [hidden_size, N_gate] bfloat16 (we use W^T in kernel)
        # up_weight: [hidden_size, N_up] bfloat16 (we use W^T in kernel)
        assert len(args) == 3, "forward expects (hidden_states, gate_weight, up_weight)"
        hidden_states, gate_weight, up_weight = args

        # Ensure on CUDA and contiguous
        device = hidden_states.device
        assert device.type == "cuda", "ModelNew requires CUDA device"
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        # Gate output dimension
        N_gate = gate_weight.shape[1]
        # Up output dimension
        N_up = up_weight.shape[1]

        # Allocate outputs for gate and up (float32 for accumulation)
        gate_out = torch.empty((M, N_gate), dtype=torch.float32, device=device)
        up_out = torch.empty((M, N_up), dtype=torch.float32, device=device)

        # Launch GEMV for gate_out: X = hidden_states [M,K], W = gate_weight [K,N_gate]
        grid_gate = (M,)
        matmul_rowwise_bf16[grid_gate](
            hidden_states, gate_weight, gate_out,
            M, K, N_gate,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),  # W is [K, N_gate]
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N,
        )

        # Launch GEMV for up_out: X = hidden_states [M,K], W = up_weight [K,N_up]
        grid_up = (M,)
        matmul_rowwise_bf16[grid_up](
            hidden_states, up_weight, up_out,
            M, K, N_up,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),  # W is [K, N_up]
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N,
        )

        # Elementwise activation and multiply: activated = silu(gate_out) * up_out
        # Ensure shapes match (in provided setup N_gate == N_up == hidden_size)
        assert N_gate == N_up, "Gate and Up output dimensions must match for activation"
        activated = torch.empty((M, N_gate), dtype=torch.float32, device=device)

        # 1D launch: one program processes 1024 elements
        total = M * N_gate
        element_grid = (triton.cdiv(total, 1024),)
        silu_mul_kernel[element_grid](
            gate_out, up_out, activated,
            M, N_gate,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
        )

        # Cast to bfloat16 to match original dtype
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
