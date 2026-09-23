import torch
import triton
import triton.language as tl


@triton.jit
def _rand_f32_kernel(out_ptr, count: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < count
    # tl.rand returns float32
    tl.store(out_ptr + offs, tl.rand(), mask=mask)


@triton.jit
def _matmul_kernel(out_ptr, a_ptr, b_ptr, M, N, K,
                   a_stride_m, a_stride_k,
                   b_stride_k, b_stride_n,
                   out_stride_m, out_stride_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute C = A @ B where A is [M, K], B is [K, N], C is [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k)
        b_ptrs = b_ptr + (offs_k[:, None] * b_stride_k + offs_n[None, :] * b_stride_n)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K], float32
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N], float32
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]
    c_ptrs = out_ptr + (offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _silu_kernel(out_ptr, x_ptr, M, N, out_stride_m, out_stride_n, x_stride_m, x_stride_n):
    # Elementwise SiLU: out = x * sigmoid(x)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_m = 128
    tile_n = 128
    offs_m = pid_m * tile_m + tl.arange(0, tile_m)
    offs_n = pid_n * tile_n + tl.arange(0, tile_n)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * x_stride_m + offs_n[None, :] * x_stride_n, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, y, mask=mask)


@triton.jit
def _mul_kernel(out_ptr, a_ptr, b_ptr, M, N, a_stride_m, a_stride_n, b_stride_m, b_stride_n, out_stride_m, out_stride_n):
    # Elementwise multiply: out = a * b
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_m = 128
    tile_n = 128
    offs_m = pid_m * tile_m + tl.arange(0, tile_m)
    offs_n = pid_n * tile_n + tl.arange(0, tile_n)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptr + offs_m[:, None] * a_stride_m + offs_n[None, :] * a_stride_n, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_m[:, None] * b_stride_m + offs_n[None, :] * b_stride_n, mask=mask, other=0.0)
    c = a * b
    tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n, c, mask=mask)


@triton.jit
def _fill_ones_kernel(out_ptr, count: tl.constexpr, out_stride: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < count
    tl.store(out_ptr + offs * out_stride, 1.0, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, device: torch.device, batch_seq_len: int, hidden_size: int):
        # Create random inputs and weights using Triton kernels (float32)
        M = batch_seq_len
        H = hidden_size
        # hidden states: [M, H]
        hidden = torch.empty((M, H), dtype=torch.float32, device=device)
        _rand_f32_kernel[(triton.cdiv(M * H, 1024),)](hidden)
        # gate_weight: [H, H]
        gate_weight = torch.empty((H, H), dtype=torch.float32, device=device)
        _rand_f32_kernel[(triton.cdiv(H * H, 1024),)](gate_weight)
        # up_weight: [H, H]
        up_weight = torch.empty((H, H), dtype=torch.float32, device=device)
        _rand_f32_kernel[(triton.cdiv(H * H, 1024),)](up_weight)
        # bias: [H], ones in float32
        bias = torch.empty((H,), dtype=torch.float32, device=device)
        _fill_ones_kernel[(triton.cdiv(H, 1024),)](bias, bias.stride(0))

        # Compute gate_output = hidden @ gate_weight -> [M, H]
        gate_output = torch.empty((M, H), dtype=torch.float32, device=device)
        grid_mm = (triton.cdiv(M, 128), triton.cdiv(H, 128))
        _matmul_kernel[grid_mm](
            gate_output, hidden, gate_weight,
            M, H, H,
            hidden.stride(0), hidden.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        )

        # Compute up_output = hidden @ up_weight -> [M, H]
        up_output = torch.empty((M, H), dtype=torch.float32, device=device)
        grid_mm_up = (triton.cdiv(M, 128), triton.cdiv(H, 128))
        _matmul_kernel[grid_mm_up](
            up_output, hidden, up_weight,
            M, H, H,
            hidden.stride(0), hidden.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        )

        # Compute silu(gate_output)
        silu_gate = torch.empty((M, H), dtype=torch.float32, device=device)
        grid_silu = (triton.cdiv(M, 64), triton.cdiv(H, 64))
        _silu_kernel[grid_silu](
            silu_gate, gate_output,
            M, H, silu_gate.stride(0), silu_gate.stride(1),
            gate_output.stride(0), gate_output.stride(1),
        )

        # Multiply silu(gate_output) by up_output
        activated = torch.empty((M, H), dtype=torch.float32, device=device)
        grid_mul = (triton.cdiv(M, 64), triton.cdiv(H, 64))
        _mul_kernel[grid_mul](
            activated, silu_gate, up_output,
            M, H, silu_gate.stride(0), silu_gate.stride(1),
            up_output.stride(0), up_output.stride(1),
            activated.stride(0), activated.stride(1),
        )

        # Return in bfloat16
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
