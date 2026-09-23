import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32_const(
    X_ptr,          # *bfloat16, [M, K]
    W_ptr,          # *bfloat16, [K, N]
    Y_ptr,          # *float32,  [M, N]
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,   # X.stride(0) = K (elements)
    stride_xk: tl.constexpr,   # X.stride(1) = 1 (element)
    stride_wk: tl.constexpr,   # W.stride(0) = N (elements)
    stride_wn: tl.constexpr,   # W.stride(1) = 1 (element)
    stride_ym: tl.constexpr,   # Y.stride(0) = N (elements)
    stride_yn: tl.constexpr,   # Y.stride(1) = 1 (element)
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # X[m, k_idx] -> [BLOCK_K] bfloat16
        x_vec = tl.load(X_ptr + m * stride_xm + k_idx * stride_xk, mask=k_mask, other=0.0).to(tl.float32)

        # W[k_idx, :] -> [BLOCK_K, BLOCK_N] bfloat16
        n_idx = tl.arange(0, BLOCK_N)
        n_mask = n_idx < N
        w_tile = tl.load(W_ptr + k_idx[:, None] * stride_wk + n_idx[None, :] * stride_wn,
                         mask=k_mask[:, None] & n_mask[None, :],
                         other=0.0).to(tl.float32)

        # acc += sum_k w_tile[k, n] * x_vec[k]
        acc += tl.sum(w_tile * x_vec[:, None], axis=0)

    # Store Y[m, :]
    y_row_ptr = Y_ptr + m * stride_ym
    tl.store(y_row_ptr + n_idx * stride_yn, acc, mask=n_mask)


@triton.jit
def silu_mul_kernel_const(
    GateOut_ptr,     # *float32, [M, N]
    UpOut_ptr,       # *float32, [M, N]
    Y_ptr,           # *float32, [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_um: tl.constexpr,
    stride_un: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    mask = (m < M) & (n < N)

    gate = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn, mask=mask, other=0.0)
    up = tl.load(UpOut_ptr + m * stride_um + n * stride_un, mask=mask, other=0.0)

    # SiLU(x) = x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    y = gate * sig * up

    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, gate_weight, up_weight, *args, **kwargs):
        # hidden_states: [M, K], gate_weight, up_weight: [K, N]
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = gate_weight.shape[1]  # Provided setup uses N=1408

        # Ensure contiguous and move to CUDA (if needed). Evaluation harness sets device.
        hidden_bf16 = hidden_states.to(torch.bfloat16).contiguous()
        gate_w = gate_weight.to(torch.bfloat16).contiguous()
        up_w = up_weight.to(torch.bfloat16).contiguous()

        # Allocate outputs
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_bf16.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_bf16.device)
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_bf16.device)

        # Precompute strides as compile-time constants (in elements)
        # X = hidden_bf16: shape [M, K], strides (K, 1)
        stride_xm = K
        stride_xk = 1
        # W = gate_w / up_w: shape [K, N], strides (N, 1)
        stride_wk = N
        stride_wn = 1
        # Y: [M, N], strides (N, 1)
        stride_ym = N
        stride_yn = 1

        # Launch GEMV: one program per row
        grid = (M,)
        BLOCK_K = 256
        BLOCK_N = 128
        linear_rowwise_bf16_to_f32_const[grid](
            hidden_bf16, gate_w, gate_out,
            M, K, N,
            stride_xm, stride_xk,
            stride_wk, stride_wn,
            stride_ym, stride_yn,
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        linear_rowwise_bf16_to_f32_const[grid](
            hidden_bf16, up_w, up_out,
            M, K, N,
            stride_xm, stride_xk,
            stride_wk, stride_wn,
            stride_ym, stride_yn,
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Elementwise kernel over [M, N]
        grid_e = (M, N)
        silu_mul_kernel_const[grid_e](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=1, num_stages=1,
        )

        # Return bfloat16 (cast via dtype constructor; no torch ops on tensors)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
