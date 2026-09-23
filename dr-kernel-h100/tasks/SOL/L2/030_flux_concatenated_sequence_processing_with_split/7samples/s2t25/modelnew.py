import torch
import triton
import triton.language as tl

# Kernel 1: Concatenate encoder_hidden_states and hidden_states along sequence dimension
@triton.jit
def concatenate_seqs_kernel(
    enc_ptr,              # *float32, [B, T, H]
    hid_ptr,              # *float32, [B, I, H]
    out_ptr,              # *float32, [B, S, H], S = T + I
    B: tl.constexpr,      # int
    T: tl.constexpr,      # int
    I: tl.constexpr,      # int
    H: tl.constexpr,      # int
    enc_stride0, enc_stride1, enc_stride2,   # strides for encoder: (B, T, H)
    hid_stride0, hid_stride1, hid_stride2,   # strides for hidden: (B, I, H)
    out_stride0, out_stride1, out_stride2,   # strides for output: (B, S, H)
    BLOCK_T: tl.constexpr = 64,              # tile for T
    BLOCK_I: tl.constexpr = 64,              # tile for I
):
    pid_b = tl.program_id(0)  # batch index
    # Copy encoder rows into out[:, 0:T, :]
    for t0 in range(0, T, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T
        # each t_offsets contributes a vector of H elements
        for h in range(0, H):
            enc_addr = enc_ptr + pid_b * enc_stride0 + t_offsets * enc_stride1 + h * enc_stride2
            out_addr = out_ptr + pid_b * out_stride0 + t_offsets * out_stride1 + h * out_stride2
            val = tl.load(enc_addr, mask=mask_t, other=0.0)
            tl.store(out_addr, val, mask=mask_t)

    # Copy hidden rows into out[:, T:T+I, :]
    for i0 in range(0, I, BLOCK_I):
        i_offsets = i0 + tl.arange(0, BLOCK_I)
        mask_i = i_offsets < I
        for h in range(0, H):
            hid_addr = hid_ptr + pid_b * hid_stride0 + i_offsets * hid_stride1 + h * hid_stride2
            out_addr = out_ptr + pid_b * out_stride0 + (T + i_offsets) * out_stride1 + h * out_stride2
            val = tl.load(hid_addr, mask=mask_i, other=0.0)
            tl.store(out_addr, val, mask=mask_i)

# Kernel 2: Matmul C = A @ B, where A is [M, H] with M = B*S, B is [H, H], C is [M, H]
@triton.jit
def matmul_seqs_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,                        # M = B*S, N=H, K=H
    A_stride0, A_stride1,          # strides for A (M, H)
    B_stride0, B_stride1,          # strides for B (H, H)
    C_stride0, C_stride1,          # strides for C (M, H)
    BLOCK_M: tl.constexpr = 1,     # M is not tiled here; we launch one row per program
    BLOCK_N: tl.constexpr = 128,   # tile over N
    BLOCK_K: tl.constexpr = 128,   # tile over K
):
    pid_m = tl.program_id(0)  # row index in A (i.e., batch*seq index)
    pid_n = tl.program_id(1)  # tile index over N
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # A[pid_m, k] vector of length BLOCK_K
        a_addr = A_ptr + pid_m * A_stride0 + k_offsets * A_stride1
        a_vec = tl.load(a_addr, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # B[k, n] tile of shape [BLOCK_K, BLOCK_N]
        b_addr = B_ptr + k_offsets[:, None] * B_stride0 + n_offsets[None, :] * B_stride1
        b_tile = tl.load(b_addr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # acc += a_vec[:, None] * b_tile
        acc += tl.sum(b_tile * a_vec[:, None], axis=0)

    # Store acc into C[pid_m, :]
    c_addr = C_ptr + pid_m * C_stride0 + n_offsets * C_stride1
    tl.store(c_addr, acc, mask=mask_n)

# Kernel 3a: Split C into processed_encoder = C[:, :T, :]
@triton.jit
def split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, H,
    C_stride0, C_stride1, C_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_T: tl.constexpr = 64,   # tile for T
):
    pid_b = tl.program_id(0)
    for t0 in range(0, T, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T
        for h in range(0, H):
            src_addr = C_ptr + pid_b * C_stride0 + t_offsets * C_stride1 + h * C_stride2
            dst_addr = out_ptr + pid_b * out_stride0 + t_offsets * out_stride1 + h * out_stride2
            vals = tl.load(src_addr, mask=mask_t, other=0.0)
            tl.store(dst_addr, vals, mask=mask_t)

# Kernel 3b: Split C into processed_hidden = C[:, T:, :]
@triton.jit
def split_hidden_kernel(
    C_ptr, out_ptr,
    B, I, H,
    C_stride0, C_stride1, C_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_I: tl.constexpr = 64,   # tile for I
):
    pid_b = tl.program_id(0)
    for i0 in range(0, I):
        # process one row at a time (I can be dynamic)
        # write to out[b, i0, :]
        for h in range(0, H):
            src_addr = C_ptr + pid_b * C_stride0 + (I + i0) * C_stride1 + h * C_stride2
            dst_addr = out_ptr + pid_b * out_stride0 + i0 * out_stride1 + h * out_stride2
            val = tl.load(src_addr)
            tl.store(dst_addr, val)

class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation that:
          1) Concatenates encoder_hidden_states and hidden_states along sequence dim in Triton.
          2) Computes processed = concatenated @ process_weight.T in Triton (GEMM).
          3) Splits processed back into encoder and hidden streams in Triton.

        Inputs:
          encoder_hidden_states: [B, T, H]
          hidden_states: [B, I, H]
          process_weight: [H, H]

        Returns:
          processed_encoder: [B, T, H]
          processed_hidden: [B, I, H]
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B, T, H = encoder_hidden_states.shape
        I, H2 = hidden_states.shape
        assert H == H2, "Hidden dim mismatch."
        S = T + I

        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Bw = process_weight.contiguous()

        # 1) Concatenate sequences into [B, S, H]
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)

        grid_concat = (B,)
        concatenate_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_T=64, BLOCK_I=64,
            num_warps=2, num_stages=1,
        )

        # 2) Matmul: C = concatenated @ process_weight.T, where process_weight.T is [H, H]
        # Treat concatenated as [M, H], process_weight.T as [H, H], output C as [M, H]
        M = B * S
        C = torch.empty((M, H), device=enc.device, dtype=enc.dtype)

        grid_matmul = (M, triton.cdiv(H, 128))
        # Important: pass strides such that A is [M, H] and B is [H, H]
        # For concatenated [B, S, H], A strides: (S*H, H)
        # We can build A view by interpreting concatenated linearly: A[i, h] = concatenated[b, s, h] where i = b*S + s
        # However, Triton kernel expects pointer to data; we can pass concatenated and adjust strides accordingly.
        # Since A is [M, H], A_stride0 = H (distance between rows), A_stride1 = 1.
        A_stride0 = concatenated.stride(1)  # should be H
        A_stride1 = concatenated.stride(2)  # should be 1
        Bw_T = Bw.transpose(0, 1)  # get [H, H]; ensure contiguous
        B_stride0 = Bw_T.stride(0)  # H
        B_stride1 = Bw_T.stride(1)  # H

        C_stride0 = C.stride(0)  # H
        C_stride1 = C.stride(1)  # 1

        # We pass M, N, K as runtime ints; Triton will handle.
        matmul_seqs_kernel[grid_matmul](
            concatenated, Bw_T, C,
            M, H, H,  # N=H, K=H
            A_stride0, A_stride1,
            B_stride0, B_stride1,
            C_stride0, C_stride1,
            BLOCK_M=1, BLOCK_N=128, BLOCK_K=128,
            num_warps=4, num_stages=3,
        )

        # 3) Split C back into encoder and hidden streams
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)

        grid_split_e = (B,)
        split_encoder_kernel[grid_split_e](
            C, processed_encoder,
            B, T, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_T=64,
            num_warps=2, num_stages=1,
        )

        # processed_hidden: C[:, T:, :] -> rows indices T..T+I-1
        # Implement as a kernel that copies one row per program
        for b in range(B):
            for i in range(I):
                for h in range(0, H):
                    src = C[b * S + T + i, h]
                    dst = processed_hidden[b, i, h]
                    # Triton kernels cannot do Python assignments; instead, we can write a tiny kernel for hidden split.
                    pass  # This loop is not valid in Triton; we replace it with a proper Triton kernel below.

        # Proper Triton hidden split kernel:
        grid_split_h = (B,)
        split_hidden_kernel[grid_split_h](
            C, processed_hidden,
            B, I, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_I=64,
            num_warps=2, num_stages=1,
        )

        return processed_encoder, processed_hidden