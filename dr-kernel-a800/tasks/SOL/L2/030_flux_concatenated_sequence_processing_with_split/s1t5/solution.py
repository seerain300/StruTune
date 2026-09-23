import torch
import triton
import triton.language as tl


@triton.jit
def _concat_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, P, K,
    encoder_stride_b, encoder_stride_l, encoder_stride_k,
    hidden_stride_b, hidden_stride_l, hidden_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_B: tl.constexpr, BLOCK_L: tl.constexpr
):
    """
    Concatenate encoder_hidden_states [B, T, K] and hidden_states [B, P, K] along L into out [B, T+P, K].
    Each program handles a tile [BLOCK_B x BLOCK_L] of the output.
    """
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    b_ids = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    l_ids = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)

    # Masks for boundaries
    b_mask = b_ids < B
    l_mask = l_ids < (T + P)

    # Compute source pointers:
    # For each l < T: encoder[b, l, :]
    # For each l >= T: hidden[b, l-T, :]
    # We’ll create two masks: is_encoder and is_hidden
    is_encoder = l_ids[:, None] < T
    is_hidden = (l_ids[:, None] >= T) & (l_ids[:, None] < (T + P))

    # Compute base offsets for out
    out_offsets = (b_ids[:, None] * out_stride_b) + (l_ids[None, :] * out_stride_l) + (tl.arange(0, K)[None, :] * out_stride_k)
    out_mask = (b_mask[:, None]) & (l_mask[None, :])

    # Compute encoder offsets when l < T
    encoder_offsets = (b_ids[:, None] * encoder_stride_b) + (l_ids[None, :] * encoder_stride_l) + (tl.arange(0, K)[None, :] * encoder_stride_k)
    encoder_mask = (b_mask[:, None]) & (is_encoder[None, :])

    # Compute hidden offsets when l >= T
    hidden_offsets = (b_ids[:, None] * hidden_stride_b) + ((l_ids[None, :] - T) * hidden_stride_l) + (tl.arange(0, K)[None, :] * hidden_stride_k)
    hidden_mask = (b_mask[:, None]) & (is_hidden[None, :])

    # Load values where applicable
    # Note: tl.load supports masked loads; other=0.0 ensures safe loads.
    val = tl.zeros((BLOCK_B, BLOCK_L, K), dtype=tl.float32)
    # If any mask is true, load; otherwise keep zeros. We need per-element masking on (B, L).
    # Triton’s tl.load will ignore masked elements.
    val += tl.load(encoder_ptr + encoder_offsets, mask=encoder_mask, other=0.0)
    val += tl.load(hidden_ptr + hidden_offsets, mask=hidden_mask, other=0.0)

    # Store into out
    tl.store(out_ptr + out_offsets, val, mask=out_mask)


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K], M = B * (T+P)
    W_ptr,  # [K, K]
    C_ptr,  # [M, K]
    M, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute C = A @ W where A is [M, K], W is [K, K], output C is [M, K].
    Each program computes a tile [BLOCK_M, BLOCK_N].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k

        # Load A sub-block: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K + k_ids[None, :])
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W sub-block: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + (k_ids[:, None] * K + offs_n[None, :])
        w_mask = (k_ids[:, None] < K) & (offs_n[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(a, w)

    c_ptrs = C_ptr + (offs_m[:, None] * K + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_l, C_stride_k,
    out_stride_b, out_stride_k,
    BLOCK_B: tl.constexpr, BLOCK_T: tl.constexpr
):
    """
    Copy C[:, :T, :] -> out [B, T, K]
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    b_ids = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    t_ids = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    b_mask = b_ids < B
    t_mask = t_ids < T

    # Compute pointers for source C
    C_offsets = (b_ids[:, None] * C_stride_b) + (t_ids[None, :] * C_stride_l) + (tl.arange(0, K)[None, :] * C_stride_k)
    C_mask = (b_mask[:, None]) & (t_mask[None, :])

    # Compute pointers for destination out (assume contiguous [B, T, K])
    out_offsets = (b_ids[:, None] * out_stride_b) + (t_ids[None, :] * out_stride_k) + (tl.arange(0, K)[None, :] * 1)  # K=hidden_dim, stride_k=1 for contiguous
    # We need out_stride_k for general, but out is [B, T, K], typically contiguous => out_stride_k == 1
    # To be safe, use b_ids*out_stride_b + t_ids*0 + k*1
    # However, Triton tensors use element strides. Since out is [B, T, K], contiguous => strides (K^2, K, 1).
    # We should pass out_stride_k explicitly from tensor.
    # In our code, out is torch.empty((B, T, K), device=..., dtype=torch.float32), so out_stride_k is 1.
    # We pass it as argument.

    # Load from C
    vals = tl.load(C_ptr + C_offsets, mask=C_mask, other=0.0)

    # Store to out
    out_offsets = (b_ids[:, None] * out_stride_b) + (t_ids[None, :] * out_stride_k) + (tl.arange(0, K)[None, :] * out_stride_k)
    out_mask = (b_mask[:, None]) & (t_mask[None, :])
    tl.store(out_ptr + out_offsets, vals, mask=out_mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_l, C_stride_k,
    out_stride_b, out_stride_k,
    BLOCK_B: tl.constexpr, BLOCK_P: tl.constexpr
):
    """
    Copy C[:, T:, :] -> out [B, P, K]
    """
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    b_ids = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    p_ids = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    b_mask = b_ids < B
    p_mask = p_ids < P

    # Compute source offsets for C: l starts at T
    C_offsets = (b_ids[:, None] * C_stride_b) + ((T + p_ids[None, :]) * C_stride_l) + (tl.arange(0, K)[None, :] * C_stride_k)
    C_mask = (b_mask[:, None]) & (p_mask[None, :])

    # Compute destination offsets for out: [B, P, K], contiguous => strides (K^2, K, 1)
    out_offsets = (b_ids[:, None] * out_stride_b) + (p_ids[None, :] * out_stride_k) + (tl.arange(0, K)[None, :] * out_stride_k)
    # Note: We must pass out_stride_k for out_ptr. When creating out as torch.empty((B, P, K), dtype=torch.float32), it is contiguous: stride(2)=1.
    # We'll pass it as argument.

    vals = tl.load(C_ptr + C_offsets, mask=C_mask, other=0.0)
    tl.store(out_ptr + out_offsets, vals, mask=(b_mask[:, None]) & (p_mask[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Concatenate encoder and hidden along sequence dim in Triton
          - Linear projection via Triton matmul
          - Split into encoder and hidden outputs in Triton
        """
        # Ensure float32 for stability and Triton support
        device = hidden_states.device
        hidden_states = hidden_states.to(torch.float32)
        encoder_hidden_states = encoder_hidden_states.to(torch.float32)
        process_weight = process_weight.to(torch.float32)

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # 1) Triton concatenation: Acat [B, T+P, K] without torch.cat
        total_L = T + P
        Acat = torch.empty((B, total_L, K), device=device, dtype=torch.float32)

        # Strides
        eb, et, ek = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        hb, ht, hk = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        ob, ol, ok = Acat.stride(0), Acat.stride(1), Acat.stride(2)

        BLOCK_Bc = 32
        BLOCK_Lc = 128
        grid_c = (triton.cdiv(B, BLOCK_Bc), triton.cdiv(total_L, BLOCK_Lc))
        _concat_kernel[grid_c](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            eb, et, ek,
            hb, ht, hk,
            ob, ol, ok,
            BLOCK_B=BLOCK_Bc, BLOCK_L=BLOCK_Lc,
            num_warps=4, num_stages=2
        )

        # 2) Triton GEMM: C = Acat @ process_weight.T
        M = B * total_L
        Wt = process_weight.contiguous()  # [K, K]
        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_mm = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_mm](
            Acat.reshape(M, K),
            Wt,
            C_flat,
            M, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape to [B, T+P, K]
        C = C_flat.view(B, total_L, K)

        # 3) Triton split: encoder [B, T, K] and hidden [B, P, K]
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        # Strides for C
        Cb, Cl, Ck = C.stride(0), C.stride(1), C.stride(2)

        # Encoder split kernel
        BLOCK_Bs = 32
        BLOCK_Ts = 128
        grid_e = (triton.cdiv(B, BLOCK_Bs), triton.cdiv(T, BLOCK_Ts))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K,
            Cb, Cl, Ck,
            processed_encoder.stride(0), processed_encoder.stride(2),
            BLOCK_B=BLOCK_Bs, BLOCK_T=BLOCK_Ts,
            num_warps=4, num_stages=2
        )

        # Hidden split kernel
        BLOCK_Bh = 32
        BLOCK_Ph = 128
        grid_h = (triton.cdiv(B, BLOCK_Bh), triton.cdiv(P, BLOCK_Ph))
        _split_hidden_kernel[grid_h](
            C, processed_hidden,
            B, T, P, K,
            Cb, Cl, Ck,
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_B=BLOCK_Bh, BLOCK_P=BLOCK_Ph,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
