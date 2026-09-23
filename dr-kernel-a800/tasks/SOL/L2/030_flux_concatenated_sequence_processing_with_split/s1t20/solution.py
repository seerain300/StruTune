import torch
import triton
import triton.language as tl


# Triton kernel: concatenate first T rows of encoder_hidden into C
# C: [B, S, K] where S = T + P
# encoder_hidden: [B, T, K]
@triton.jit
def concat_first_T(
    C_ptr,  # pointer to output [B, S, K]
    E_ptr,  # pointer to input encoder [B, T, K]
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
    stride_Cb, stride_Cs, stride_Ck,
    stride_Eb, stride_Et, stride_Ek,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B, cdiv(T, BLOCK_M), cdiv(K, BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)  # tile along T
    pid_n = tl.program_id(2)  # tile along K

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along T
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along K

    mask_m = m_offsets < T
    mask_n = n_offsets < K

    # Compute pointers for input and output tiles
    # Input E[b, m, n] -> E_ptr + b*stride_Eb + m*stride_Et + n*stride_Ek
    E_tile_ptr = E_ptr + b * stride_Eb + m_offsets[:, None] * stride_Et + n_offsets[None, :] * stride_Ek

    # Output C[b, m, n] where m < T -> store to C[b, m, n]
    # Note: S here is T + P, but we only write first T rows
    C_tile_ptr = C_ptr + b * stride_Cb + m_offsets[:, None] * stride_Cs + n_offsets[None, :] * stride_Ck

    # Masked load/store
    # Load from E with mask on m,n
    vals = tl.load(E_tile_ptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    # Store to C (m < T ensured by mask)
    tl.store(C_tile_ptr, vals, mask=mask_m[:, None] & mask_n[None, :])


# Triton kernel: concatenate last P rows of hidden into C starting at S=T
@triton.jit
def concat_last_P(
    C_ptr,  # pointer to output [B, S, K]
    H_ptr,  # pointer to input hidden [B, P, K]
    B: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    stride_Cb, stride_Cs, stride_Ck,
    stride_Hb, stride_Hp, stride_Hk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B, cdiv(P, BLOCK_M), cdiv(K, BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)  # tile along P
    pid_n = tl.program_id(2)  # tile along K

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along P
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along K

    mask_m = m_offsets < P
    mask_n = n_offsets < K

    # Input H[b, p, n] -> H_ptr + b*stride_Hb + p*stride_Hp + n*stride_Hk
    H_tile_ptr = H_ptr + b * stride_Hb + m_offsets[:, None] * stride_Hp + n_offsets[None, :] * stride_Hk

    # Output C[b, T + p, n] -> C_ptr + b*stride_Cb + (T + p)*stride_Cs + n*stride_Ck
    S_offset = T  # defined on host as T + P
    C_tile_ptr = C_ptr + b * stride_Cb + (S_offset + m_offsets[:, None]) * stride_Cs + n_offsets[None, :] * stride_Ck

    vals = tl.load(H_tile_ptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(C_tile_ptr, vals, mask=mask_m[:, None] & mask_n[None, :])


# Triton kernel: GEMM C = A @ B (no bias), A: [M, K], B: [K, K], C: [M, K]
# We will launch with grid = (B, cdiv(M, BLOCK_M), cdiv(K, BLOCK_N))
@triton.jit
def matmul_triton(
    C_ptr,  # [M, K] flat output buffer
    A_ptr,  # [M, K] (concatenated tensor)
    B_ptr,  # [K, K] (process_weight.T)
    M: tl.constexpr, K: tl.constexpr,
    stride_Am, stride_Ak,
    stride_Bk, stride_Bn,  # B is [K, K], strides for k (row) and n (col)
    stride_Cm, stride_Ck,  # C is [M, K]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index, but we only have one C flat, so pid_b is 0
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptr = A_ptr + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak
        A_vals = tl.load(A_tile_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        B_tile_ptr = B_ptr + k_offsets[:, None] * stride_Bk + n_offsets[None, :] * stride_Bn
        B_vals = tl.load(B_tile_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(A_vals, B_vals)

    # Store results to C flat: [M, K]
    C_tile_ptr = C_ptr + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Ck
    tl.store(C_tile_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])


# Triton kernel: split first T rows of C into C_encoder [B, T, K]
@triton.jit
def split_encoder(
    C_ptr,  # [B, S, K]
    E_ptr,  # [B, T, K]
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
    stride_Cb, stride_Cs, stride_Ck,
    stride_Eb, stride_Et, stride_Ek,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    grid = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(K, BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < T
    mask_n = n_offsets < K

    C_tile_ptr = C_ptr + b * stride_Cb + m_offsets[:, None] * stride_Cs + n_offsets[None, :] * stride_Ck
    E_tile_ptr = E_ptr + b * stride_Eb + m_offsets[:, None] * stride_Et + n_offsets[None, :] * stride_Ek

    vals = tl.load(C_tile_ptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(E_tile_ptr, vals, mask=mask_m[:, None] & mask_n[None, :])


# Triton kernel: split last P rows of C into C_hidden [B, P, K]
@triton.jit
def split_hidden(
    C_ptr,  # [B, S, K]
    H_ptr,  # [B, P, K]
    B: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    stride_Cb, stride_Cs, stride_Ck,
    stride_Hb, stride_Hp, stride_Hk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    grid = (B, triton.cdiv(P, BLOCK_M), triton.cdiv(K, BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < P
    mask_n = n_offsets < K

    # S_offset = T
    S_offset = T
    C_tile_ptr = C_ptr + b * stride_Cb + (S_offset + m_offsets[:, None]) * stride_Cs + n_offsets[None, :] * stride_Ck
    H_tile_ptr = H_ptr + b * stride_Hb + m_offsets[:, None] * stride_Hp + n_offsets[None, :] * stride_Hk

    vals = tl.load(C_tile_ptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(H_tile_ptr, vals, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we assume inputs are float32 and contiguous
        self.block_m = 64
        self.block_n = 64
        self.block_k = 32
        self.num_warps = 4
        self.num_stages = 2

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, P, K]
        encoder_hidden_states: [B, T, K]
        process_weight: [K, K]
        Returns (processed_encoder: [B, T, K], processed_hidden: [B, P, K])
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        S = T + P

        # Ensure contiguous float32 (the original code implicitly uses float32)
        device = hidden_states.device
        dtype = torch.float32
        # Allocate Acat [B, S, K]
        Acat = torch.empty((B, S, K), device=device, dtype=dtype)
        # Create views for concatenation kernels
        # First T rows from encoder_hidden
        # Launch concat_first_T
        grid_T = (B, triton.cdiv(T, self.block_m), triton.cdiv(K, self.block_n))
        _ = concat_first_T[grid_T](
            Acat, encoder_hidden_states,
            B=B, T=T, K=K,
            stride_Cb=Acat.stride(0), stride_Cs=Acat.stride(1), stride_Ck=Acat.stride(2),
            stride_Eb=encoder_hidden_states.stride(0), stride_Et=encoder_hidden_states.stride(1), stride_Ek=encoder_hidden_states.stride(2),
            BLOCK_M=self.block_m, BLOCK_N=self.block_n,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # Last P rows from hidden_states
        grid_P = (B, triton.cdiv(P, self.block_m), triton.cdiv(K, self.block_n))
        _ = concat_last_P[grid_P](
            Acat, hidden_states,
            B=B, P=P, K=K,
            stride_Cb=Acat.stride(0), stride_Cs=Acat.stride(1), stride_Ck=Acat.stride(2),
            stride_Hb=hidden_states.stride(0), stride_Hp=hidden_states.stride(1), stride_Hk=hidden_states.stride(2),
            BLOCK_M=self.block_m, BLOCK_N=self.block_n,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # Now perform GEMM: Acat @ process_weight.T, where process_weight.T is [K, K]
        # Flatten Acat to [M, K] where M = B * S
        A_flat = Acat.reshape(-1, K)  # M = B*S
        B_mat = process_weight.t()    # [K, K], float32
        # Ensure B_mat is contiguous
        B_mat = B_mat.contiguous()
        # Allocate flat output [M, K]
        C_flat = torch.empty((A_flat.shape[0], K), device=device, dtype=dtype)
        # Launch GEMM kernel with grid = (1, cdiv(M, BLOCK_M), cdiv(K, BLOCK_N))
        # Note: M = B*S, K is K, N = K
        grid_mm = (1, triton.cdiv(A_flat.shape[0], self.block_m), triton.cdiv(K, self.block_n))
        _ = matmul_triton[grid_mm](
            C_flat, A_flat, B_mat,
            M=A_flat.shape[0], K=K,
            stride_Am=A_flat.stride(0), stride_Ak=A_flat.stride(1),
            stride_Bk=B_mat.stride(0), stride_Bn=B_mat.stride(1),
            stride_Cm=C_flat.stride(0), stride_Ck=C_flat.stride(1),
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # Reshape C_flat back to [B, S, K]
        C = C_flat.reshape(B, S, K)

        # Split into encoder and hidden
        processed_encoder = torch.empty((B, T, K), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=dtype)

        grid_e = (B, triton.cdiv(T, self.block_m), triton.cdiv(K, self.block_n))
        _ = split_encoder[grid_e](
            C, processed_encoder,
            B=B, T=T, K=K,
            stride_Cb=C.stride(0), stride_Cs=C.stride(1), stride_Ck=C.stride(2),
            stride_Eb=processed_encoder.stride(0), stride_Et=processed_encoder.stride(1), stride_Ek=processed_encoder.stride(2),
            BLOCK_M=self.block_m, BLOCK_N=self.block_n,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        grid_h = (B, triton.cdiv(P, self.block_m), triton.cdiv(K, self.block_n))
        _ = split_hidden[grid_h](
            C, processed_hidden,
            B=B, P=P, K=K,
            stride_Cb=C.stride(0), stride_Cs=C.stride(1), stride_Ck=C.stride(2),
            stride_Hb=processed_hidden.stride(0), stride_Hp=processed_hidden.stride(1), stride_Hk=processed_hidden.stride(2),
            BLOCK_M=self.block_m, BLOCK_N=self.block_n,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
