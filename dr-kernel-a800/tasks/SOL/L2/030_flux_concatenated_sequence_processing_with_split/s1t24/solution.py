import torch
import triton
import triton.language as tl


# Triton kernel: concatenate along sequence dimension.
# Acat [B, S, K], encoder [B, T, K], hidden [B, P, K], S = T + P
@triton.jit
def _concatenation_kernel(
    encoder_ptr, hidden_ptr, acat_ptr,
    B, T, P, K,
    enc_s, enc_k, enc_b,
    hid_s, hid_k, hid_b,
    ac_s, ac_k, ac_b,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    # If s >= T, then source is hidden at index s - T
    if s < T:
        src_s = s
        is_hidden = False
    else:
        src_s = s - T
        is_hidden = True

    k_offsets = tl.arange(0, BLOCK_K)
    mask = k_offsets < K

    # Pointers for encoder and hidden
    enc_ptrs = encoder_ptr + b * enc_b + src_s * enc_s + k_offsets * enc_k
    hid_ptrs = hidden_ptr + b * hid_b + src_s * hid_s + k_offsets * hid_k

    # Select source based on is_hidden
    vals = tl.load(enc_ptrs, mask=mask, other=0.0)
    if is_hidden:
        vals = tl.load(hid_ptrs, mask=mask, other=0.0)

    # Store into Acat
    ac_ptrs = acat_ptr + b * ac_b + s * ac_s + k_offsets * ac_k
    tl.store(ac_ptrs, vals, mask=mask)


# Triton kernel: split concatenated [B, S, K] into two outputs
# C_in [B, S, K], processed_encoder [B, T, K], processed_hidden [B, P, K]
@triton.jit
def _split_encoder_kernel(
    c_in_ptr, out_ptr,
    B, T, K,
    c_b, c_s, c_k,
    out_b, out_k, out_s,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    s = tl.program_id(1)  # s in [0, T)
    k_offsets = tl.arange(0, BLOCK_K)
    mask = k_offsets < K

    in_ptrs = c_in_ptr + b * c_b + s * c_s + k_offsets * c_k
    out_ptrs = out_ptr + b * out_b + s * out_s + k_offsets * out_k

    vals = tl.load(in_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def _split_hidden_kernel(
    c_in_ptr, out_ptr,
    B, T, P, K,
    c_b, c_s, c_k,
    out_b, out_s, out_k,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    s_rel = tl.program_id(1)  # s_rel in [0, P)
    s = T + s_rel
    k_offsets = tl.arange(0, BLOCK_K)
    mask = k_offsets < K

    in_ptrs = c_in_ptr + b * c_b + s * c_s + k_offsets * c_k
    out_ptrs = out_ptr + b * out_b + s_rel * out_s + k_offsets * out_k

    vals = tl.load(in_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


# Triton kernel: GEMM with A [M, K] = concatenated, W [K, K], output C_flat [M, K]
# M = B * (T + P), K is arbitrary.
@triton.jit
def _gemm_simple_kernel(
    A_ptr, W_ptr, C_ptr,
    M, K,
    A_b, A_m, A_k,
    W_k, W_n,  # W has shape [K, K] so stride along k (rows) and k (cols)
    C_b, C_m, C_k,
):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)
    # Bounds check (grid is (M, K), but do it anyway for safety)
    if row_id >= M or col_id >= K:
        return

    # Compute the dot product for this (row_id, col_id)
    acc = 0.0
    # Loop over K in chunks of 1 (since we index scalar), but better to vectorize:
    # We'll iterate in chunks of BLOCK_K=32
    BLOCK_K = 32
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        a_vals = tl.load(A_ptr + row_id * A_m + k_offsets * A_k, mask=k_mask, other=0.0)
        w_vals = tl.load(W_ptr + k_offsets * W_k + col_id * W_n, mask=k_mask, other=0.0)
        # Accumulate: scalar multiply of vectors (reduce to scalar)
        # Compute elementwise product then reduce
        prod = a_vals * w_vals
        # Reduce to scalar: sum along the vector
        acc += tl.sum(prod, axis=0)

    # Store result
    out_ptr = C_ptr + row_id * C_m + col_id * C_k
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        1) Concatenate encoder_hidden_states and hidden_states along the sequence dimension into Acat [B, T+P, K]
        2) Compute processed = Acat @ process_weight.T using a Triton GEMM kernel
        3) Split processed into processed_encoder [B, T, K] and processed_hidden [B, P, K] using Triton kernels
        Returns (processed_encoder, processed_hidden)
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, dim, K]"
        assert hidden_states.shape[2] == encoder_hidden_states.shape[2] == process_weight.shape[0] == process_weight.shape[1], "hidden_dim mismatch"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Ensure tensors are on CUDA and contiguous for Triton
        device = hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA device"
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not encoder_hidden_states.is_contiguous():
            encoder_hidden_states = encoder_hidden_states.contiguous()
        if not process_weight.is_contiguous():
            process_weight = process_weight.contiguous()

        S = T + P

        # 1) Triton concatenation: Acat [B, S, K]
        Acat = torch.empty((B, S, K), device=device, dtype=torch.float32)
        # Strides
        enc_strides = encoder_hidden_states.stride()  # (B, S, K)
        hid_strides = hidden_states.stride()
        acat_strides = Acat.stride()
        # Launch 3D grid over (B, S, 1)
        BLOCK_K = 128
        grid = (B, S, 1)
        _concatenation_kernel[grid](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            enc_strides[0], enc_strides[1], enc_strides[2],
            hid_strides[0], hid_strides[1], hid_strides[2],
            acat_strides[0], acat_strides[1], acat_strides[2],
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Triton GEMM: A is Acat flattened to [M, K], W is process_weight.T
        M = B * S
        A_flat = Acat.reshape(M, K).contiguous()  # [M, K]
        W_T = process_weight.t().contiguous()     # [K, K]
        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        # Launch per-element GEMM kernel over (M, K)
        _gemm_simple_kernel[(M, K)](
            A_flat, W_T, C_flat,
            M, K,
            A_flat.stride(0), A_flat.stride(1),
            W_T.stride(0), W_T.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            num_warps=1, num_stages=1
        )

        # Reshape back to [B, S, K]
        processed = C_flat.view(B, S, K)

        # 3) Triton split: processed_encoder [B, T, K] and processed_hidden [B, P, K]
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        # For split, we use 3D grid (B, size_of_split, 1)
        grid_e = (B, T, 1)
        grid_h = (B, P, 1)
        BLOCK_K_split = 128

        _split_encoder_kernel[grid_e](
            processed, processed_encoder,
            B, T, K,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        _split_hidden_kernel[grid_h](
            processed, processed_hidden,
            B, T, P, K,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
