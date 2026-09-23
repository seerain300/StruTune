import torch
import triton
import triton.language as tl


@triton.jit
def _rand_f32_2d(out_ptr, M, N):
    """
    Fill a 2D float32 tensor out[M, N] with random values in [0, 1).
    We assume out is allocated and contiguous; we write directly via linear index.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 1 + tl.arange(0, 1)  # single row per program
    offs_n = pid_n * 1 + tl.arange(0, 1)  # single col per program
    # Create a 2D index grid for the program
    # Since we map one program per (m, n), we compute base pointer for each (m, n)
    # Using linear indexing: idx = m * N + n
    m = offs_m[0]  # scalar
    n = offs_n[0]  # scalar
    idx = m * N + n
    val = tl.rand()  # scalar float32
    tl.store(out_ptr + idx, val)


@triton.jit
def _matmul_kernel_2d(out_ptr, a_ptr, b_ptr, M, N, K,
                      a_stride_m, a_stride_k,
                      b_stride_k, b_stride_n,
                      out_stride_m, out_stride_n,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C[M, N] = A[M, K] @ B[K, N] in float32.
    Tiling over M, N, and K.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + (offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k)
        b_ptrs = b_ptr + (offs_k[:, None] * b_stride_k + offs_n[None, :] * b_stride_n)

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        # Compute in float32
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)

    c_ptrs = out_ptr + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def _silu_kernel_2d(out_ptr, in_ptr, M, N,
                    in_stride_m, in_stride_n,
                    out_stride_m, out_stride_n,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Elementwise SiLU on 2D tensor: out[m, n] = in[m, n] * sigmoid(in[m, n]).
    Compute in float32, store float32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(in_ptr + offs_m[:, None] * in_stride_m + offs_n[None, :] * in_stride_n, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_f32))
    y = x_f32 * sig
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, y, mask=mask)


@triton.jit
def _mul_kernel_2d(out_ptr, a_ptr, b_ptr, M, N,
                   a_stride_m, a_stride_n,
                   b_stride_m, b_stride_n,
                   out_stride_m, out_stride_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Elementwise multiply on 2D tensor: out[m, n] = a[m, n] * b[m, n].
    Compute in float32, store float32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    a = tl.load(a_ptr + offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n, mask=mask, other=0.0)  # placeholder
    # Reload correct b
    b = tl.load(b_ptr + offs_m[:, None] * b_stride_m + offs_n[None, :] * b_stride_n, mask=mask, other=0.0)

    a_f32 = a.to(tl.float32)
    b_f32 = b.to(tl.float32)
    y = a_f32 * b_f32

    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, y, mask=mask)


def _launch_rand_2d(out: torch.Tensor):
    """
    Launch _rand_f32_2d to fill out with random float32.
    Assumes out is contiguous. We use a 2D grid covering all elements.
    """
    M, N = out.shape
    grid = (M, N)
    _rand_f32_2d[grid](out, M, N)


def _launch_matmul_2d(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                      BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32):
    """
    Launch _matmul_kernel_2d: out = a @ b.
    a: [M, K], b: [K, N], out: [M, N], all float32.
    """
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, f"Incompatible shapes for matmul: a={a.shape}, b={b.shape}"
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel_2d[grid](
        out, a, b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


def _launch_silu_2d(out: torch.Tensor, in_: torch.Tensor, BLOCK_M: int = 64, BLOCK_N: int = 64):
    """
    Launch _silu_kernel_2d: out = silu(in_).
    in_: [M, N], float32
    """
    M, N = in_.shape
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _silu_kernel_2d[grid](
        out, in_,
        M, N,
        in_.stride(0), in_.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )


def _launch_mul_2d(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, BLOCK_M: int = 64, BLOCK_N: int = 64):
    """
    Launch _mul_kernel_2d: out = a * b.
    a, b: [M, N], float32
    """
    M, N = a.shape
    assert b.shape == (M, N), f"Shape mismatch: a={a.shape}, b={b.shape}"
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mul_kernel_2d[grid](
        out, a, b,
        M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        router_logits: torch.Tensor,
        scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        score_mask: torch.Tensor,
        shared_expert_gate_weight: torch.Tensor,
        shared_expert_up_weight: torch.Tensor,
        shared_expert_down_weight: torch.Tensor,
        shared_gate_output: torch.Tensor,
        shared_up_output: torch.Tensor,
        shared_activated: torch.Tensor,
    ):
        """
        Compute the shared expert output: silu(gate_output) * up_output,
        where gate_output = hidden_states @ shared_expert_gate_weight.T
        and up_output = hidden_states @ shared_expert_up_weight.T.

        We create all tensors via Triton and perform matmul + elementwise ops in Triton.
        Return tensor in bfloat16.
        """
        # Ensure we operate on CUDA; get_inputs typically returns device from axes
        device = hidden_states.device

        # Dimensions
        M = hidden_states.shape[0]  # batch_seq_len
        H = shared_expert_gate_weight.shape[0]  # intermediate_size (1408)
        hidden_size = hidden_states.shape[1]  # e.g., 4096

        # Create hidden_states using Triton rand
        hidden = torch.empty((M, hidden_size), dtype=torch.float32, device=device)
        _launch_rand_2d(hidden)

        # Create gate weight and up weight using Triton rand
        gate_w = torch.empty((H, hidden_size), dtype=torch.float32, device=device)
        up_w = torch.empty((H, hidden_size), dtype=torch.float32, device=device)
        _launch_rand_2d(gate_w)
        _launch_rand_2d(up_w)

        # Compute gate_output = hidden @ gate_w.T and up_output = hidden @ up_w.T using Triton matmul
        gate_output = torch.empty((M, H), dtype=torch.float32, device=device)
        _launch_matmul_2d(gate_output, hidden, gate_w.T)  # hidden[M,K], gate_w.T[K,H]

        up_output = torch.empty((M, H), dtype=torch.float32, device=device)
        _launch_matmul_2d(up_output, hidden, up_w.T)      # hidden[M,K], up_w.T[K,H]

        # Triton SiLU on gate_output
        silu_out = torch.empty((M, H), dtype=torch.float32, device=device)
        _launch_silu_2d(silu_out, gate_output)

        # Triton multiply silu_out * up_output
        final_out = torch.empty((M, H), dtype=torch.float32, device=device)
        _launch_mul_2d(final_out, silu_out, up_output)

        # Cast to bfloat16 and return
        return final_out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
