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
):
    # Each program handles one row m
    m = tl.program_id(0)
    # Iterate over N dimension in tiles for better vectorization
    n_start = 0
    while n_start < N:
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)  # accumulator for current N tile
        # Iterate over K dimension
        k_start = 0
        while k_start < K:
            # Load X[m, k_start:k_start+BLOCK_K] as bfloat16
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            x = tl.load(
                X_ptr + m * stride_xm + k_offsets * stride_xk,
                mask=k_offsets < K,
                other=0.0
            )  # [BLOCK_K], bfloat16
            # Load W[k_start:k_start+BLOCK_K, n_start:n_start+BLOCK_N] as bfloat16
            n_offsets = n_start + tl.arange(0, BLOCK_K)  # reuse BLOCK_K for vectorization
            w = tl.load(
                W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn,
                mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
                other=0.0
            )  # [BLOCK_K, BLOCK_N], bfloat16
            # Accumulate: acc += sum_k w[k, n] * x[k]
            acc += tl.sum(w.to(tl.float32) * x[:, None].to(tl.float32), axis=0)
            k_start += BLOCK_K
        # Store acc into Y[m, n_start:n_start+BLOCK_N]
        store_mask = (n_start + tl.arange(0, BLOCK_K)) < N
        tl.store(Y_ptr + m * stride_ym + (n_start + tl.arange(0, BLOCK_K)) * stride_yn, acc, mask=store_mask)
        n_start += BLOCK_K


@triton.jit
def silu_mul_kernel(
    Gate_ptr,        # *float32,  shape [M, N]
    Up_ptr,          # *float32,  shape [M, N]
    Out_ptr,         # *float32,  shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
):
    # 1D grid over M*N; compute element (m, n)
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    if m < M and n < N:
        gate = tl.load(Gate_ptr + m * stride_gm + n * stride_gn)
        up = tl.load(Up_ptr + m * stride_um + n * stride_un)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-gate))
        out = gate * sig * up
        tl.store(Out_ptr + m * stride_om + n * stride_on, out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                shared_expert_gate_weight: torch.Tensor,
                shared_expert_up_weight: torch.Tensor,
                *args, **kwargs):
        """
        Compute shared_activated = SiLU(linear(hidden_states, gate_weight.T)) * linear(hidden_states, up_weight.T)
        - hidden_states: [M, K], bfloat16
        - shared_expert_gate_weight: [K, N], bfloat16
        - shared_expert_up_weight: [K, N], bfloat16
        Returns: [M, N], bfloat16
        """
        # Ensure contiguity
        X = hidden_states.contiguous()  # [M, K], bfloat16
        gate_weight = shared_expert_gate_weight.contiguous()  # [K, N], bfloat16
        up_weight = shared_expert_up_weight.contiguous()      # [K, N], bfloat16

        M, K = X.shape
        K_w, N = gate_weight.shape
        assert K == K_w, f"hidden_states K={K} must match gate_weight K={K_w}"

        # Allocate outputs as float32 for accumulation
        gate_out = torch.empty((M, N), dtype=torch.float32, device=X.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=X.device)

        # Launch row-wise GEMV kernels
        # Use BLOCK_K=256 (works well for K=4096); iterate over N in chunks of 256
        BLOCK_K = 256
        grid = (M,)
        linear_rowwise_bf16_to_f32[grid](
            X, gate_weight, gate_out,
            M, K, N,
            X.stride(0), X.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        linear_rowwise_bf16_to_f32[grid](
            X, up_weight, up_out,
            M, K, N,
            X.stride(0), X.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Elementwise activation: y = gate_out * sigmoid(gate_out) * up_out
        activated = torch.empty((M, N), dtype=torch.float32, device=X.device)
        total_elems = M * N
        silu_mul_kernel[(total_elems,)](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return as bfloat16 (no torch ops on tensors here)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
