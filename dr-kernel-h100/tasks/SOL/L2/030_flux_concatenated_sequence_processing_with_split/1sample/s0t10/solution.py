import torch
import triton
import triton.language as tl

@triton.jit
def _concatenator_kernel(
    enc_ptr,          # *const float, shape [B, T, H]
    hid_ptr,          # *const float, shape [B, I, H]
    out_ptr,          # *float,       shape [B, T+I, H]
    B, T, I, H,       # int32 scalars
):
    b = tl.program_id(0)
    # Guard against overlaunch
    if b >= B:
        return

    # Each program handles one batch; iterate over l in [0, T+I)
    total_L = T + I
    l = 0
    while l < total_L:
        # Determine source: encoder or hidden
        is_encoder = l < T
        k = l  # column index within H
        # Load from encoder if is_encoder else from hidden
        # For encoder: load from enc_ptr[b, l, k]
        # For hidden:  load from hid_ptr[b, l - T, k]
        enc_addr = b * (T * H) + l * H + k
        hid_addr = b * (I * H) + (l - T) * H + k
        value = 0.0
        # Use masks for safety (though enc/hid bounds are guaranteed by l and k)
        mask = (b < B) & (l < total_L) & (k < H)
        value = tl.load(enc_ptr + enc_addr, mask=mask & is_encoder, other=0.0)
        # If not encoder, switch to hidden
        value = tl.load(hid_ptr + hid_addr, mask=mask & (~is_encoder), other=0.0)
        out_addr = b * ((T + I) * H) + l * H + k
        tl.store(out_ptr + out_addr, value, mask=mask)
        l += 1

@triton.jit
def _gemma_triton_kernel(  # matmul: C[M, K] = A[M, K] @ W[K, K]
    A_ptr,  # *const float, shape [M, K], contiguous
    W_ptr,  # *const float, shape [K, K], contiguous (note: W is provided as [H,H], K==H)
    C_ptr,  # *float,       shape [M, K], contiguous
    M,      # int32, total rows = B*(T+I)
    K,      # int32, hidden_dim
    BLOCK_M: tl.constexpr,  # tile size in M
    BLOCK_N: tl.constexpr,  # tile size in N (here == K)
    BLOCK_K: tl.constexpr,  # reduction tile size in K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        off_k = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + off_m[:, None] * K + off_k[None, :]
        a = tl.load(a_ptrs, mask=(off_m[:, None] < M) & (off_k[None, :] < K), other=0.0)
        a = a.to(tl.float32)

        # W tile (we want W^T in the computation; here W is [K,K], we load as W[k, n])
        # Form W_block: [BLOCK_K, BLOCK_N] = W[off_k, off_n]
        w_ptrs = W_ptr + off_k[:, None] * K + off_n[None, :]
        w = tl.load(w_ptrs, mask=(off_k[:, None] < K) & (off_n[None, :] < K), other=0.0)
        w = w.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, w)

    # Write back C tile
    c_ptrs = C_ptr + off_m[:, None] * K + off_n[None, :]
    tl.store(c_ptrs, acc, mask=(off_m[:, None] < M) & (off_n[None, :] < K))


@triton.jit
def _split_streams_kernel(
    C_ptr,          # *const float, shape [M, K], contiguous, M=B*(T+I), K=H
    out_e_ptr,      # *float,       shape [B, T, K], contiguous
    out_i_ptr,      # *float,       shape [B, I, K], contiguous
    M, T, I, K,     # int32 scalars
):
    # One program per batch
    b = tl.program_id(0)
    if b >= B:
        return

    # Iterate over l in [0, T+I)
    total_L = T + I
    l = 0
    while l < total_L:
        m = b * (T + I) + l
        # Copy C[m, :] to appropriate output
        # if l < T -> out_e[b, l, :]
        # else -> out_i[b, l - T, :]
        k = 0
        while k < K:
            c_val = tl.load(C_ptr + m * K + k, mask=(m < M) & (k < K), other=0.0)
            if l < T:
                out_addr = b * (T * K) + l * K + k
            else:
                rel = l - T
                out_addr = b * (I * K) + rel * K + k
            tl.store(out_e_ptr + out_addr, c_val, mask=(m < M) & (k < K))
            k += 1
        l += 1


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation that performs:
          - Concatenation along sequence dimension
          - Batched GEMM via Triton kernel
          - Splitting into two streams

        Returns:
          (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2, "Invalid input shapes"
        B, I, H = hidden_states.shape
        T = encoder_hidden_states.shape[1]
        device = hidden_states.device

        # Ensure inputs are on the same device and contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W = process_weight.contiguous()  # [H, H]

        # 1) Concatenate via Triton: out_cat [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=torch.float32, device=device)
        _concatenator_kernel[(B,)](
            enc, hid, out_cat,
            B, T, I, H,
            num_warps=1, num_stages=1
        )

        # 2) GEMM via Triton: C = out_cat @ W
        # out_cat: [B, T+I, H] -> A: [M, K], M = B*(T+I), K = H
        M = B * (T + I)
        K = H
        A = out_cat.view(M, K)  # [M, K], contiguous
        C = torch.empty((M, K), dtype=torch.float32, device=device)

        # Launch 2D grid over tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(K, meta['BLOCK_N']))

        _gemma_triton_kernel[grid](
            A, W, C,
            M, K,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

        # 3) Split into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, K), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, K), dtype=torch.float32, device=device)

        # We need B from inputs
        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            M, T, I, K,
            num_warps=1, num_stages=1
        )

        # Return as floats (K == H); original code returns [B, T, H], [B, I, H]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
