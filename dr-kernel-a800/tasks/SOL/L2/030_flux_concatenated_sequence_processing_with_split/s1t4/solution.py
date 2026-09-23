import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    encoder_ptr, hidden_ptr, cat_ptr,
    B, T, P, K,
    enc_stride_b, enc_stride_t, enc_stride_k,
    hid_stride_b, hid_stride_p, hid_stride_k,
    cat_stride_b, cat_stride_t, cat_stride_k,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr
):
    # program ids: batch and sequence tile
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    total_L = T + P

    # sequence indices this program handles
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < total_L

    # Determine which rows come from encoder vs hidden
    mask_encoder = offs_l < T  # boolean per element
    mask_hidden = ~mask_encoder

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load encoder rows where mask_encoder is True
        enc_ptrs = (
            encoder_ptr
            + pid_b * enc_stride_b
            + offs_l[:, None] * enc_stride_t
            + offs_k[None, :] * enc_stride_k
        )
        mask_e = mask_encoder[:, None] & mask_k[None, :]
        enc_vals = tl.load(enc_ptrs, mask=mask_e, other=0.0)

        # Store into cat[b, l, k] for encoder rows
        cat_enc_ptrs = (
            cat_ptr
            + pid_b * cat_stride_b
            + offs_l[:, None] * cat_stride_t
            + offs_k[None, :] * cat_stride_k
        )
        tl.store(cat_enc_ptrs, enc_vals, mask=mask_e)

        # Load hidden rows where mask_hidden is True (l = offs_l - T)
        hid_ptrs = (
            hidden_ptr
            + pid_b * hid_stride_b
            + (offs_l[:, None] - T) * hid_stride_p
            + offs_k[None, :] * hid_stride_k
        )
        mask_h = mask_hidden[:, None] & mask_k[None, :]
        hid_vals = tl.load(hid_ptrs, mask=mask_h, other=0.0)

        # Store into cat[b, l, k] for hidden rows
        cat_hid_ptrs = (
            cat_ptr
            + pid_b * cat_stride_b
            + offs_l[:, None] * cat_stride_t
            + offs_k[None, :] * cat_stride_k
        )
        tl.store(cat_hid_ptrs, hid_vals, mask=mask_h)


@triton.jit
def _matmul_flat_kernel(
    A_ptr, W_ptr, C_ptr,
    M, K,
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # program ids for tiles in output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of W / C

    mask_m = offs_m < M
    mask_n = offs_n < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * A_stride_m) + (offs_k[None, :] * A_stride_k)
        A_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W[offs_k, offs_n] -> [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + (offs_k[:, None] * W_stride_k) + (offs_n[None, :] * W_stride_n)
        W_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(W_ptrs, mask=W_mask, other=0.0)

        acc += tl.dot(a, w)

    # Store C[offs_m, offs_n] = acc
    C_ptrs = C_ptr + (offs_m[:, None] * C_stride_m) + (offs_n[None, :] * C_stride_n)
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_t, C_stride_k,
    out_stride_b, out_stride_t, out_stride_k,
    BLOCK_B: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    mask_b = offs_b < B
    mask_t = offs_t < T

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        C_ptrs = (
            C_ptr
            + offs_b[:, None] * C_stride_b
            + offs_t[None, :] * C_stride_t
            + offs_k[:, None] * C_stride_k
        )
        mask = mask_b[:, None] & mask_t[None, :] & mask_k[:, None]
        vals = tl.load(C_ptrs, mask=mask, other=0.0)

        out_ptrs = (
            out_ptr
            + offs_b[:, None] * out_stride_b
            + offs_t[None, :] * out_stride_t
            + offs_k[:, None] * out_stride_k
        )
        tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_t, C_stride_k,
    out_stride_b, out_stride_p, out_stride_k,
    BLOCK_B: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    mask_b = offs_b < B
    mask_p = offs_p < P

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        C_ptrs = (
            C_ptr
            + offs_b[:, None] * C_stride_b
            + (offs_p[None, :] + T) * C_stride_t
            + offs_k[:, None] * C_stride_k
        )
        mask = mask_b[:, None] & mask_p[None, :] & mask_k[:, None]
        vals = tl.load(C_ptrs, mask=mask, other=0.0)

        out_ptrs = (
            out_ptr
            + offs_b[:, None] * out_stride_b
            + offs_p[None, :] * out_stride_p
            + offs_k[:, None] * out_stride_k
        )
        tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Concatenate (encoder, image) via Triton kernel
          - Linear projection via Triton matmul kernel
          - Split into encoder and image outputs via Triton kernels
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "ModelNew requires CUDA tensors for Triton execution."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32, \
            "This Triton implementation expects float32 tensors."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, K)
        assert process_weight.shape == (K, K)

        # 1) Triton concatenation: Acat = [encoder, hidden] -> [B, T+P, K]
        total_L = T + P
        Acat = torch.empty((B, total_L, K), device=hidden_states.device, dtype=torch.float32)

        # Choose block sizes; 64 is a good default across many GPUs
        BLOCK_L = 64
        BLOCK_K = 64

        grid_cat = (B, triton.cdiv(total_L, BLOCK_L))
        _concatenate_kernel[grid_cat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Triton GEMM: C = Acat @ process_weight.T, flatten to [M, K] -> [B*(T+P), K]
        M = B * total_L
        Acat_flat = Acat.reshape(M, K).contiguous()
        W_t = process_weight.transpose(0, 1).contiguous()  # [K, K]

        C_flat = torch.empty((M, K), device=hidden_states.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_flat_kernel[grid_matmul](
            Acat_flat, W_t, C_flat,
            M, K,
            Acat_flat.stride(0), Acat_flat.stride(1),
            W_t.stride(0), W_t.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Reshape and Triton split into encoder and hidden streams
        C = C_flat.view(B, total_L, K)

        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        BLOCK_B = 64
        BLOCK_T = 64
        BLOCK_K = 64

        grid_e = (triton.cdiv(B, BLOCK_B), triton.cdiv(T, BLOCK_T))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_B=BLOCK_B, BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        BLOCK_P = 64
        grid_i = (triton.cdiv(B, BLOCK_B), triton.cdiv(P, BLOCK_P))
        _split_hidden_kernel[grid_i](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_B=BLOCK_B, BLOCK_P=BLOCK_P, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
