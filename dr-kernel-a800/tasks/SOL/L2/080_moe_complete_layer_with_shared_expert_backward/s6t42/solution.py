import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,          # *bfloat16, shape [M, K]
    W_ptr,          # *bfloat16, shape [K, N]
    Y_ptr,          # *float32,  shape [M, N]
    M, K, N,        # int scalars
    stride_xm, stride_xk,  # strides for X
    stride_wk, stride_wn,  # strides for W
    stride_ym, stride_yn,  # strides for Y
    BLOCK_K: tl.constexpr,  # tile over K
    BLOCK_N: tl.constexpr,  # tile over N
):
    # Each program handles one row m
    m = tl.program_id(0)

    n_start = 0
    while n_start < N:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        k_start = 0
        while k_start < K:
            # Vector of size BLOCK_K over K dimension
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            # Load x[m, k_start:k_start+BLOCK_K] with mask
            x_ptrs = X_ptr + m * stride_xm + k_offsets * stride_xk
            x_mask = k_offsets < K
            x_vec = tl.load(x_ptrs, mask=x_mask, other=0.0)
            # Cast to float32 for accumulation
            x_vec = x_vec.to(tl.float32)

            # Matrix of size (BLOCK_K, BLOCK_N) over W[k, n]
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
            w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
            w_mat = tl.load(w_ptrs, mask=w_mask, other=0.0)
            w_mat = w_mat.to(tl.float32)

            # Accumulate: acc += sum over k of x_vec[k] * w_mat[k, :]
            # Implement as loop over BLOCK_K for robustness
            for kk in range(BLOCK_K):
                # Mask for valid k
                valid_k = k_start + kk < K
                x_val = 0.0
                if valid_k:
                    x_val = x_vec[kk]
                acc += x_val * w_mat[kk, :]

            k_start += BLOCK_K

        # Store acc into Y[m, n_start:n_start+BLOCK_N]
        y_ptrs = Y_ptr + m * stride_ym + (n_start + tl.arange(0, BLOCK_N)) * stride_yn
        y_mask = (n_start + tl.arange(0, BLOCK_N)) < N
        tl.store(y_ptrs, acc, mask=y_mask)

        n_start += BLOCK_N


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,     # *float32, shape [M, N]
    UpOut_ptr,       # *float32, shape [M, N]
    Y_ptr,           # *float32, shape [M, N]
    M, N,            # int scalars
    stride_gm, stride_gn,  # strides for GateOut
    stride_um, stride_un,  # strides for UpOut
    stride_ym, stride_yn,  # strides for Y
    BLOCK_MN: tl.constexpr,  # tile over total elements
):
    # 1D grid
    pid = tl.program_id(0)
    total = M * N
    offsets = pid * BLOCK_MN + tl.arange(0, BLOCK_MN)
    mask = offsets < total

    # Compute (m, n) from linear offsets
    m = offsets // N
    n = offsets % N

    # Load GateOut and UpOut with masks
    g_ptrs = GateOut_ptr + m * stride_gm + n * stride_gn
    u_ptrs = UpOut_ptr + m * stride_um + n * stride_un
    gate = tl.load(g_ptrs, mask=mask, other=0.0)
    up = tl.load(u_ptrs, mask=mask, other=0.0)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-gate))
    y = gate * sig * up

    y_ptrs = Y_ptr + m * stride_ym + n * stride_yn
    tl.store(y_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, shared_expert_gate_weight, shared_expert_up_weight):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda, "hidden_states must be on CUDA device"
        assert shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, "Weights must be on CUDA device"

        # Make sure inputs are contiguous
        hidden_states = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()
        up_weight = shared_expert_up_weight.contiguous()

        M, K = hidden_states.shape
        K2 = gate_weight.shape[0]
        N = up_weight.shape[1]
        assert K2 == K, f"hidden_states second dim {K} must match gate weight rows {K2}"
        assert up_weight.shape[0] == K, "up_weight first dim must match hidden_size K"

        # Allocate outputs for gate_out and up_out in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch row-wise linear kernels: one program per row
        grid_lin = (M,)
        BLOCK_K = 64
        BLOCK_N = 128
        linear_rowwise_bf16_to_f32[grid_lin](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        linear_rowwise_bf16_to_f32[grid_lin](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Compute shared_activated = SiLU(gate_out) * up_out using Triton elementwise kernel
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid_silu = (triton.cdiv(M * N, 1024),)
        silu_mul_kernel[grid_silu](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            BLOCK_MN=1024,
            num_warps=4, num_stages=1
        )

        # Return bfloat16 tensor
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
