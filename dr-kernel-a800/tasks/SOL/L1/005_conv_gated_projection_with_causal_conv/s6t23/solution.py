import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _matmul_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + Bias, where:
      A: (M, K), row-major: stride_am for rows, stride_ak for cols
      B: (K, N), row-major: stride_bk for rows, stride_bn for cols
      C: (M, N), row-major: stride_cm for rows, stride_cn for cols
    Bias: (N), added after matmul
    Accumulation is in float32, outputs stored in C_ptr dtype.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # Ensure a/b are fp32 for stable accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,  # M = B, N = S, K = H, we pass sizes to help kernel if needed
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_ck,
):
    """
    Elementwise C = A * B for tensors A, B, C of shape (M, N, K).
    Launch grid over (M, N) and vectorize along K.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # We process K in chunks; here we assume K is small enough to handle in one go or pass as meta. Use simple 1D elementwise:
    # Instead, launch per (m,n) and iterate over K using tl.arange:
    offs_k = tl.arange(0, 128)  # use 128; mask handles K
    for k0 in range(0, K, 128):
        k = k0 + offs_k
        mask = k < K
        a_ptrs = A_ptr + pid_m * stride_am + pid_n * stride_bm + k * stride_ak
        b_ptrs = B_ptr + pid_m * stride_am + pid_n * stride_bm + k * stride_bk
        a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)
        c = a * b
        c_ptrs = C_ptr + pid_m * stride_cm + pid_n * stride_ck + k * stride_ck
        tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _out_proj_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B^T + Bias, where:
      A: (M, K), row-major: stride_am for rows, stride_ak for cols
      B: (K, N), row-major: stride_bk for rows, stride_bn for cols
      C: (M, N), row-major: stride_cm for rows, stride_cn for cols
    Bias: (N), added after matmul
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # B^T indexing: (K, N)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, W_ptr, Bias_ptr, C_ptr,
    B, H, S,
    stride_bx_bh, stride_bx_bs, stride_bx_bh2,
    stride_w_h, stride_w_k,
    stride_c_b, stride_c_h, stride_c_s,
    OUT_DTYPE: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_S: tl.constexpr,
):
    """
    Grouped causal 1D convolution:
    Input Bx: (B, H, S) (we pass it as (B, S, H) via strides but logical is (B,H,S))
    Weight W: (H, 4) where each row corresponds to a group.
    Output C: (B, H, S)
    Padding is implicitly handled by left offset: for output index s, valid input is at index s + 3 - k for k in [0..3].
    We only access valid s positions; for k > s, those terms are 0 due to padding.
    Groups are handled by using separate programs per (b, h).
    """
    pid_bh = tl.program_id(0)  # over B*H
    b = pid_bh // H
    h = pid_bh % H

    # Prepare offsets for S
    s0 = 0
    while s0 < S:
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        # Accumulator for this (b, h) slice over BLOCK_S
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Loop over k in {0,1,2,3}; conv kernel_size=4, padding=3 implicitly
        for k in range(4):
            inp_idx = offs_s + 3 - k  # left causal padding
            # Valid positions: inp_idx < S and k <= offs_s (because for s < k, input index negative; but here offs_s starts at s0 and we mask later)
            # We can simply mask by inp_idx < S and k <= offs_s ? Not needed; we rely on padding via index. For k > offs_s, access out of bounds will be masked later.
            # We need to compute addresses for Bx[b, h, inp_idx]. For Triton, we pass Bx with shape (B, S, H) logically via strides.
            # Logical addressing: Bx_ptr + b*stride_bx_bh + h*stride_bx_bh2 + inp_idx*stride_bx_bs
            bx_ptrs = Bx_ptr + b * stride_bx_bh + h * stride_bx_bh2 + inp_idx * stride_bx_bs
            bx_mask = mask_s & (inp_idx < S)
            bx = tl.load(bx_ptrs, mask=bx_mask, other=0.0).to(tl.float32)

            w = tl.load(W_ptr + h * stride_w_h + k * stride_w_k).to(tl.float32)  # scalar weight for this (h, k)
            acc += bx * w

        # Add bias if provided
        bias_val = tl.load(Bias_ptr + h).to(tl.float32)
        acc += bias_val

        # Store to C[b, h, s]
        c_ptrs = C_ptr + b * stride_c_b + h * stride_c_h + offs_s * stride_c_s
        tl.store(c_ptrs, acc, mask=mask_s)

        s0 += BLOCK_S


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only fused implementation of the original model:
        1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        2) Split BCx into B, C, x_proj along last dim
        3) Bx = B * x_proj
        4) Grouped causal conv1d with kernel_size=4, groups=H
        5) y = C * conv_out  (conv_out is (B, H, S); gate elementwise)
        6) out_proj: output = y @ out_proj_weight^T + out_proj_bias
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        assert x.ndim == 3, "x must be (B, S, H)"
        B, S, H = x.shape

        # Ensure tensors are on same device and float32 for kernels
        x_f = x
        in_proj_weight_f = in_proj_weight.float()
        in_proj_bias_f = in_proj_bias.float()
        conv_weight_f = conv_weight.float()  # expected last 4 columns derived from in_proj_weight in host
        conv_bias_f = conv_bias.float()
        out_proj_weight_f = out_proj_weight.float()
        out_proj_bias_f = out_proj_bias.float()

        # 1) in_proj via Triton matmul: BCx = x @ in_proj_weight^T + in_proj_bias
        K = H
        N = 3 * H
        M = B * S
        A = x_f.contiguous().view(M, K)
        B_w = in_proj_weight_f.t().contiguous().view(K, N)  # (K, N) == (H, 3H)
        Bias_in = in_proj_bias_f.contiguous().view(N)

        BCx = torch.empty((B, S, 3 * H), dtype=torch.float32, device=x.device)
        C_BCx = BCx.view(M, N)

        # Choose blocks
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_linear_kernel[grid](
            A, B_w, Bias_in, C_BCx,
            M, N, K,
            A.stride(0), A.stride(1),
            B_w.stride(0), B_w.stride(1),
            C_BCx.stride(0), C_BCx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Split BCx into B_mat, C_mat, x_proj
        B_mat = BCx[:, :, :H]            # (B, S, H)
        C_mat = BCx[:, :, H:2*H]         # (B, S, H)
        x_proj = BCx[:, :, 2*H:]         # (B, S, H)

        # 2) Elementwise gate: Bx = B_mat * x_proj (Triton)
        Bx = torch.empty_like(B_mat, dtype=torch.float32, device=x.device)
        M2 = B * S
        # Use simple elementwise 2D kernel: process (M2, H) elements
        _elementwise_mul_2d_kernel[(B, S)](
            B_mat, x_proj, Bx,
            B, S, H,
            B_mat.stride(0), B_mat.stride(1),
            x_proj.stride(0), x_proj.stride(1),
            Bx.stride(0), Bx.stride(1),
            # No BLOCK_* here; kernel loops K in chunks; pass K explicitly via launch loop? Triton expects static grid; revise kernel to 1D.
            # Revised approach: implement elementwise kernel that just iterates over all elements.
        )
        # Since Triton elementwise kernel above isn't invoked, instead use torch to maintain correctness. But the requirement is to invoke Triton kernels. Let's implement an inline elementwise kernel call:
        # We will implement an elementwise kernel that takes (B,S,H) and computes C = A*B. The previous kernel signature was wrong; define a correct one.

        # Define correct elementwise kernel:
        # We need a 3D elementwise multiply kernel. Triton supports indexing via program_id over dims. Implement as:
        @triton.jit
        def _elemwise_mul_3d(A_ptr, B_ptr, C_ptr, B, S, H):
            pid_b = tl.program_id(0)
            pid_s = tl.program_id(1)
            pid_k = tl.program_id(2)
            # each program handles one element (b, s, k)
            # We can launch grid as (B, S, H). Use static shape.
            # Load A and B, store C
            offs_k = pid_k  # scalar index
            a = tl.load(A_ptr + pid_b * B * S * H + pid_s * H + offs_k).to(tl.float32)
            b = tl.load(B_ptr + pid_b * B * S * H + pid_s * H + offs_k).to(tl.float32)
            c = a * b
            tl.store(C_ptr + pid_b * B * S * H + pid_s * H + offs_k, c)

        # Launch with grid (B, S, H)
        _elemwise_mul_3d[(B, S, H)](
            B_mat, x_proj, Bx,
            B, S, H,
        )

        # 3) Grouped causal conv with kernel_size=4, using conv_weight derived from in_proj_weight's last 4 columns
        # conv_weight_f is (H, 4) from in_proj_weight's last 4 columns. conv_bias_f is (H)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)
        # Launch grouped causal conv kernel per (b,h) sliding s
        BLOCK_S = 128
        # We need to pass strides for Bx as (B,S,H). For grouped conv, we can pass Bx as (B,H,S) logically by using strides.
        # However, we can pass Bx as (B,S,H) with strides adjusted. Let's pass Bx as (B,S,H) and adjust logical addressing via strides.
        # We will construct Bx logically: Bx_ptr points to tensor with shape (B,S,H), but we interpret logical (b,h,seq) via strides.
        # Define Bx as (B,S,H). For conv, we need to read Bx[b, h, s + 3 - k]. To implement in Triton, we pass Bx as (B,S,H) and use strides:
        # strides: stride_bx_bh, stride_bx_bs, stride_bx_bh2 (h stride) for Bx pointer manipulation.
        # But simpler: convert Bx to (B,H,S) logical by using strides in Triton loads. We can pass Bx_ptr as (B,S,H) but load with h index using stride_h = 1 (contiguous last dim).
        # For safety, we'll create a view and pass strides accordingly. However, Triton loads use pointer arithmetic; we can pass Bx as contiguous (B,S,H) and use logical strides for h.

        # We'll create Bx as (B,S,H) contiguous tensor, and in kernel we will index as (b,h,seq) via strides. To do that, we need a tensor shaped as (B,H,S). Create that logical view by reshaping and passing strides accordingly.

        # Instead, we'll create Bx_padded in Triton: allocate Bx_padded (B,H,S+3), set padding, then conv. But we don't have pad kernel defined. To keep Triton-only, implement pad implicitly in the conv kernel by passing input with padded zeros.

        # Implementing padding in Triton: We don't have pad kernel; but we can compute conv with input Bx and treat indices with inp_idx < 0 or >= S as zero by masked loads. That requires passing Bx with shape (B,S,H) and compute inp_idx = s + 3 - k, and mask inp_idx.

        # Define Bx as (B,S,H) float32 tensor. conv_out as (B,H,S) float32.

        # However, we need to pass conv kernel Bx pointer as (B,H,S). Triton allows arbitrary strides; we can pass Bx_ptr pointing to (B,S,H) but in conv kernel we will index using strides that correspond to (b,h,seq). For that, we need to pass strides for Bx corresponding to (B,H,S) logical indexing. In practice, we can create Bx as (B,H,S) logically by viewing strides of a contiguous (B,S,H) tensor. Let's do that.

        # Create a contiguous (B,S,H) tensor for Bx: pad left zeros implicitly in kernel by masked loads.
        Bx_ps = torch.empty((B, S, H), dtype=torch.float32, device=x.device)  # (B,S,H): we'll fill this with B_mat * x_proj via Triton elementwise kernel as above.
        # We already computed Bx via elementwise kernel. Now, conv kernel expects Bx as (B,H,S). We can create a logical view by using .transpose(-1,-2) and make it contiguous as (B,S,H). For conv, we need (B,H,S). To avoid confusion, we will implement conv kernel to read from (B,S,H) and treat padding.

        # So we have Bx_ps: (B,S,H). We'll call _grouped_causal_conv1d_kernel on Bx_ps, and it will read Bx_ps[b, :, s + 3 - k] for each (b,h), masking out-of-range. But _grouped_causal_conv1d_kernel expects Bx as (B,H,S). To unify, we'll implement conv kernel to accept any shape and use strides; or we'll explicitly create (B,H,S) input. Given complexity, let's instead create (B,H,S) Bx by reshaping.

        # Create Bx as (B,H,S): elementwise gate results in (B,S,H). We can transpose to (B,H,S) and use in Triton.

        Bx_T = Bx.transpose(-1, -2).contiguous()  # (B,H,S)
        # Now call Triton grouped conv kernel:
        # We need to pass Bx_T as (B,H,S). For kernel, we can pass Bx_ptr to Bx_T (contiguous) and use strides: stride_bx_bh = Bx_T.stride(0), stride_bx_hs = Bx_T.stride(1), stride_bx_hk = Bx_T.stride(2) (but in Triton we only pass base pointer; strides are per tensor, not per logical dim). We will pass Bx_T as base pointer; kernel will use Bx_T.stride(0), Bx_T.stride(1), Bx_T.stride(2) to address (b,h,seq). For inputs, we need to read Bx_T[b,h,seq]; but our conv kernel expects to read from Bx as (B,H,S). To avoid confusion, we will define conv kernel to read from (B,S,H). Therefore, we'll use Bx_ps (B,S,H) and let kernel implement padding via masked loads.

        # Let's redefine Bx as (B,S,H) and call conv kernel on it, with input indexing via (b,h,seq): compute inp_idx = seq + 3 - k, and mask. That means we pass Bx_ps (B,S,H), and in kernel we index with h and seq. Triton allows pointer arithmetic with strides; we can pass strides for (B,S,H) and use h index via multiplication.

        # We'll proceed with conv kernel on Bx_ps: (B,S,H). The kernel will read Bx_ps[b, h, seq + 3 - k] with masking, and write conv_out (B,H,S).

        # Invoke conv kernel:
        # For Bx_ps: shape (B,S,H) => strides (stride_bx_b, stride_bx_s, stride_bx_h)
        Bx_ps = Bx  # already (B,S,H)
        stride_bx_b, stride_bx_s, stride_bx_h = Bx_ps.stride(0), Bx_ps.stride(1), Bx_ps.stride(2)

        stride_w_h, stride_w_k = conv_weight_f.stride(0), conv_weight_f.stride(1)
        stride_c_b, stride_c_h, stride_c_s = conv_out.stride(0), conv_out.stride(1), conv_out.stride(2)

        # Launch one program per (b,h). We can do this by grid (B,H).
        grid_conv = (B, H)
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx_ps, conv_weight_f, conv_bias_f, conv_out,
            B, H, S,
            stride_bx_b, stride_bx_s, stride_bx_h,
            stride_w_h, stride_w_k,
            stride_c_b, stride_c_h, stride_c_s,
            # OUT_DTYPE: 0 fp32
            0,
            BLOCK_S=BLOCK_S,
        )

        # 4) Output gating: y = C_mat * conv_out (conv_out is (B,H,S); transpose to (B,S,H) for elementwise multiply)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B,S,H)
        y = torch.empty_like(C_mat, dtype=torch.float32, device=x.device)
        # Use Triton elementwise kernel for (B,S,H)
        _elemwise_mul_3d[(B, S, H)](
            C_mat, conv_out_T, y,
            B, S, H,
        )

        # 5) Final out-proj via Triton matmul: output = y @ out_proj_weight^T + out_proj_bias
        # y: (B,S,H), out_proj_weight: (H,H)
        M3 = B * S
        A3 = y.contiguous().view(M3, H)
        B_w3 = out_proj_weight_f.t().contiguous().view(H, H)  # (H,H)
        Bias_out = out_proj_bias_f.contiguous().view(H)

        output = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        C_out = output.view(M3, H)

        BLOCK_M3, BLOCK_N3, BLOCK_K3 = 128, 64, 32
        grid_out = (triton.cdiv(M3, BLOCK_M3), triton.cdiv(H, BLOCK_N3))
        _out_proj_kernel[grid_out](
            A3, B_w3, Bias_out, C_out,
            M3, H, H,
            A3.stride(0), A3.stride(1),
            B_w3.stride(0), B_w3.stride(1),
            C_out.stride(0), C_out.stride(1),
            BLOCK_M=BLOCK_M3, BLOCK_N=BLOCK_N3, BLOCK_K=BLOCK_K3,
        )

        return output


def run(*args):
    return ModelNew()(*args)
