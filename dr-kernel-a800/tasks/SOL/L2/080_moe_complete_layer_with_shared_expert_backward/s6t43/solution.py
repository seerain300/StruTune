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
    # Loop over N dimension in tiles of BLOCK_N
    n_start = 0
    while n_start < N:
        # Accumulator for this row-tile
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        # Iterate over K dimension in tiles of BLOCK_K
        k_start = 0
        while k_start < K:
            # Load a slice of the input row X[m, k_start:k_start+BLOCK_K]
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            # Masks for valid k indices
            mask_k = k_offsets < K
            # Load X row slice; pointer arithmetic: X_ptr + m*stride_xm + k_offsets*stride_xk
            x_vec = tl.load(X_ptr + m * stride_xm + k_offsets * stride_xk, mask=mask_k, other=0.0)
            # Cast to float32 for accumulation
            x_vec_f32 = x_vec.to(tl.float32)
            # Load a corresponding slice of W: W[k_offsets, n_start:n_start+BLOCK_N]
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N
            # Construct 2D pointer for W: shape (BLOCK_K, BLOCK_N)
            w_ptr = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
            mask_w = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
            w_mat = tl.load(w_ptr, mask=mask_w, other=0.0)
            w_mat_f32 = w_mat.to(tl.float32)
            # Accumulate outer product over BLOCK_K
            # acc[n] += sum_{bk} x_vec_f32[bk] * w_mat_f32[bk, n]
            for bk in range(BLOCK_K):
                # scalar x and vector w
                x_scalar = x_vec_f32[bk]
                w_vec = w_mat_f32[bk, :]
                acc += x_scalar * w_vec
            k_start += BLOCK_K
        # Store accumulated result into Y[m, n_start:n_start+BLOCK_N]
        y_ptr = Y_ptr + m * stride_ym + n_offsets * stride_yn
        store_mask = mask_n
        tl.store(y_ptr, acc, mask=store_mask)
        n_start += BLOCK_N


@triton.jit
def silu_mul_f32(
    GateOut_ptr,    # *float32, shape [M, N]
    UpOut_ptr,      # *float32, shape [M, N]
    Y_ptr,          # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
):
    # 2D grid over rows and cols
    m = tl.program_id(0)
    n = tl.program_id(1)
    # Bounds check
    if m >= M or n >= N:
        return
    gate_val = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    up_val = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
    # silu(x) = x * sigmoid(x)
    sigma = 1.0 / (1.0 + tl.exp(-gate_val))
    y = gate_val * sigma * up_val
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: hidden_states, shared_expert_gate_weight, shared_expert_up_weight
        # hidden_states: [M, K] bfloat16
        # gate_weight, up_weight: [K, N] bfloat16
        # We use Triton to compute gate_out = hidden_states @ gate_weight.T and up_out = hidden_states @ up_weight.T, then y = gate_out * sigmoid(gate_out) * up_out
        # Cast to bfloat16 and ensure contiguous
        hidden_states = args[0].to(torch.bfloat16).contiguous()
        gate_weight = args[1].to(torch.bfloat16).contiguous()
        up_weight = args[2].to(torch.bfloat16).contiguous()
        M, K = hidden_states.shape
        K_w, N = gate_weight.shape
        assert K_w == K, "gate_weight.shape[0] must equal hidden_states.shape[1]"
        assert up_weight.shape[0] == K, "up_weight.shape[0] must equal hidden_states.shape[1]"

        # Allocate outputs in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch linear kernels (row-wise)
        # Choose BLOCK_N and BLOCK_K; N=1408, K=4096
        BLOCK_N = 128
        BLOCK_K = 64
        grid_gate = (M,)
        linear_rowwise_bf16_to_f32[grid_gate](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )
        grid_up = (M,)
        linear_rowwise_bf16_to_f32[grid_up](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Elementwise: y = gate_out * sigmoid(gate_out) * up_out
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid2 = (M, N)
        silu_mul_f32[grid2](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=2
        )

        # Return as bfloat16
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
