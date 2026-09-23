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
    # Loop over N in tiles
    n_start = 0
    while n_start < N:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        # Loop over K in tiles and accumulate
        k_start = 0
        while k_start < K:
            # Build indices for this tile
            bk_idx = k_start + tl.arange(0, BLOCK_K)  # shape [BLOCK_K]
            bn_idx = n_start + tl.arange(0, BLOCK_N)  # shape [BLOCK_N]
            # Load X row slice (bfloat16), shape [BLOCK_K]
            x_mask = bk_idx < K
            x_vec = tl.load(X_ptr + m * stride_xm + bk_idx * stride_xk, mask=x_mask, other=0.0)  # bfloat16
            x_vec_f32 = x_vec.to(tl.float32)  # cast to float32

            # Load W rows slice, shape [BLOCK_K, BLOCK_N]
            w_ptrs = W_ptr + bk_idx[:, None] * stride_wk + bn_idx[None, :] * stride_wn  # [BLOCK_K, BLOCK_N]
            w_mask = (bk_idx[:, None] < K) & (bn_idx[None, :] < N)
            w_mat = tl.load(w_ptrs, mask=w_mask, other=0.0)  # bfloat16
            w_mat_f32 = w_mat.to(tl.float32)  # cast to float32

            # Accumulate: acc += sum over bk of x_vec[bk] * w_mat[bk, :]
            # x_vec_f32: [BLOCK_K], w_mat_f32: [BLOCK_K, BLOCK_N]
            for kk in range(BLOCK_K):
                contrib = x_vec_f32[kk] * w_mat_f32[kk, :]  # [BLOCK_N]
                acc += contrib

            k_start += BLOCK_K

        # Store acc to Y[m, n_start:n_start+BLOCK_N]
        y_ptrs = Y_ptr + m * stride_ym + (n_start + tl.arange(0, BLOCK_N)) * stride_yn
        y_mask = (n_start + tl.arange(0, BLOCK_N)) < N
        tl.store(y_ptrs, acc, mask=y_mask)
        n_start += BLOCK_N


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,    # *float32, shape [M, N]
    UpOut_ptr,      # *float32, shape [M, N]
    Activated_ptr,  # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_am, stride_an,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    # Load GateOut and UpOut elements
    gate = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    up = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
    # Compute SiLU(gate) = gate * sigmoid(gate)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    y = gate * sigmoid_gate * up
    tl.store(Activated_ptr + m * stride_am + n * stride_an, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                shared_expert_gate_weight: torch.Tensor,
                shared_expert_up_weight: torch.Tensor):
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda, "hidden_states must be CUDA tensor"
        assert shared_expert_gate_weight.is_cuda and shared_expert_up_weight.is_cuda, \
            "Gate and Up weights must be CUDA tensors"

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = shared_expert_gate_weight.shape[1]  # since W is [K, N]

        # Make inputs contiguous
        X = hidden_states.contiguous()
        gate_weight = shared_expert_gate_weight.contiguous()
        up_weight = shared_expert_up_weight.contiguous()

        # Output buffers for linear ops
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMV kernels: one row per program
        grid = (M,)
        linear_rowwise_bf16_to_f32[grid](
            X, gate_weight, gate_out,
            M, K, N,
            X.stride(0), X.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=128, BLOCK_N=128,
            num_warps=4, num_stages=2
        )
        linear_rowwise_bf16_to_f32[grid](
            X, up_weight, up_out,
            M, K, N,
            X.stride(0), X.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=128, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # Elementwise activated = SiLU(gate_out) * up_out
        activated_f32 = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid2 = (M, N)
        silu_mul_kernel[grid2](
            gate_out, up_out, activated_f32,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated_f32.stride(0), activated_f32.stride(1),
            num_warps=4, num_stages=1
        )

        # Cast to bfloat16 and return
        return activated_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
