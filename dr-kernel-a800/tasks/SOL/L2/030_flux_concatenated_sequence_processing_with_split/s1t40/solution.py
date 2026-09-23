import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden_states [B, T, K] and hidden_states [B, P, K]
# into Acat [B, T+P, K]. Each program handles (b, l) and vectorizes across K.
@triton.jit
def _concat_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, P, K,
    encoder_stride_b, encoder_stride_t, encoder_stride_k,
    hidden_stride_b, hidden_stride_p, hidden_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_block = tl.program_id(2)

    # Vector of K indices this program handles
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    # Mask for valid K
    k_mask = k_offsets < K

    # Base pointers
    # If l < T: source is encoder, else source is hidden at index l - T
    # We compute pointers elementwise
    # Note: Triton requires element pointers; we build pointer for each vectorized K
    # Initialize with None to let masked loads avoid invalid pointers
    # For l < T:
    enc_ptrs = encoder_ptr + b * encoder_stride_b + l * encoder_stride_t + k_offsets * encoder_stride_k
    # For l >= T:
    # l - T might be negative, but we guard with mask so we don't use it; we set masked pointers to 0.
    # We still pass valid hidden pointer; masked load will ignore when l < T.
    hid_ptrs = hidden_ptr + b * hidden_stride_b + (l - T) * hidden_stride_p + k_offsets * hidden_stride_k

    # Validity flags
    is_encoder = l < T

    # We will load using a mask combining is_encoder and k_mask.
    # Triton masked load expects a pointer tensor; when is_encoder False, we mask out.
    # Construct a mask tensor for load: valid when is_encoder and k_mask
    load_mask = (k_mask & is_encoder)

    # Load from source; if is_encoder is False, masked load will be all False
    # Triton doesn't have "if" for scalars at element level, so we rely on masked load/store.
    vals = tl.load(hid_ptrs, mask=k_mask, other=0.0)  # default load; we'll overwrite where is_encoder is False
    # If is_encoder: use enc_ptrs instead; else keep vals (zeros)
    # Triton doesn't support per-element scalar conditional assignment; we need to do it via masked store
    # We'll store to out for all valid k, but only when is_encoder: we should load from enc_ptrs, otherwise load zeros.

    # We need to construct vals from enc_ptrs where is_encoder, else zeros.
    # Triton allows masked load based on combined mask. However, to ensure correctness, we do:
    # Load enc only when is_encoder True, else we can skip by using masked store with zeroed vals for encoder path.
    # Simpler approach: create a single pointer tensor and load with combined mask, but Triton doesn't support mixing
    # pointers based on scalar. Therefore, we perform two masked loads into separate buffers and select via mask:
    # Triton doesn't support select, so we'll implement via two stores: one for encoder, one for hidden. We need
    # to combine them. The clean way is to use tl.where-like at tile level isn't available; instead, we do two
    # stores guarded by is_encoder flag. Since Triton kernel doesn't support dynamic branching, we will
    # compute both possible loads but only use the one based on is_encoder via masked store.
    # Practical approach: compute vals_enc and vals_hid separately and store with masks.

    # Load encoder when is_encoder, else use zeros
    vals_enc = tl.load(enc_ptrs, mask=(k_mask & is_encoder), other=0.0)
    vals_hid = tl.load(hid_ptrs, mask=(k_mask & (~is_encoder)), other=0.0)
    # Now we need to assemble final vals for store. Triton doesn't support where on tensor with scalar; we store
    # using masked store: store enc values when is_encoder, store hid values when not is_encoder. We can do this
    # by computing final vals as a combination, but since masked store supports per-element mask, we will perform
    # two masked stores.

    # Prepare out pointers
    out_ptrs = out_ptr + b * out_stride_b + l * out_stride_l + k_offsets * out_stride_k
    # Store enc values where is_encoder
    tl.store(out_ptrs, vals_enc, mask=(k_mask & is_encoder))
    # Store hid values where not is_encoder
    tl.store(out_ptrs, vals_hid, mask=(k_mask & (~is_encoder)))

# Triton kernel: GEMM on A [M, K] and W [K, K] -> C [M, K]
# Grid is (B, tiles over M, tiles over N), with M = B*(T+P), N = K
@triton.jit
def _matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    M, K, N,
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)  # batch dimension is not used here because M is flattened; we keep grid as (B, TM, TN)
    TM = tl.num_programs(0)  # only for completeness; we map b via offset calculation
    # Instead of using TM, we recover b from sum of all program ids: we can't, so we assume grid (B, TM, TN).
    # Better: redesign grid to (TM, TN). To keep compatibility, we relaunch with proper grid below.

    # We will instead define grid as (TM, TN) and pass B as input; Triton kernel will be relaunched with correct grid.
    # Therefore, the above block_M/N/K loop is not accessible here. We need to fix the kernel signature accordingly.

    # Note: The above kernel signature was incorrect for flattened M. We'll provide a correct version below.

# Correct Triton GEMM kernel with 3D grid (TM, TN, TK)
@triton.jit
def _matmul_kernel_3d(
    A_ptr, W_ptr, C_ptr,
    M, K, N,
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tm = tl.program_id(0)  # tile index over M dimension (rows)
    tn = tl.program_id(1)  # tile index over N dimension (cols)
    tk = tl.program_id(2)  # tile index over K reduction dimension

    # Compute row/col indices for this tile
    m_start = tm * BLOCK_M
    n_start = tn * BLOCK_N
    k_start = tk * BLOCK_K

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = k_start + tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Current K offsets
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = (m_offsets[:, None] < M) & (k_mask[None, :])
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W tile: shape [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        W_mask = (k_mask[:, None]) & (n_offsets[None, :] < N)
        W_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store result to C
    C_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)

# Triton kernel: split C [B, T+P, K] into processed_encoder [B, T, K] and processed_hidden [B, P, K]
@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_t, C_stride_k,
    out_stride_b, out_stride_p, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t_block = tl.program_id(1)
    k_block = tl.program_id(2)

    t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    t_mask = t_offsets < T
    k_mask = k_offsets < K

    # Load from C at positions (b, t, k)
    C_ptrs = C_ptr + b * C_stride_b + t_offsets[:, None] * C_stride_t + k_offsets[None, :] * C_stride_k
    C_mask = t_mask[:, None] & k_mask[None, :]
    vals = tl.load(C_ptrs, mask=C_mask, other=0.0)

    # Store to out (we will split into out_t [B, T, K] in host, but since this kernel is for encoder, we use out_ptr
    # pointing to out_t. The signature expects out_ptr for encoder; here we assume host passes pointer for encoder stream.
    # For clarity: out_ptr points to out_t, i.e., out_t is [B, T, K] contiguous.
    # We write to out_t[b, t, k]
    out_ptrs = out_ptr + b * out_stride_b + t_offsets[:, None] * out_stride_p + k_offsets[None, :] * out_stride_k  # out_stride_p is stride over T
    tl.store(out_ptrs, vals, mask=C_mask)

@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_t, C_stride_k,
    out_stride_b, out_stride_p, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    p_block = tl.program_id(1)
    k_block = tl.program_id(2)

    p_offsets = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    p_mask = p_offsets < P
    k_mask = k_offsets < K

    # Load from C at positions (b, T + p, k)
    C_ptrs = C_ptr + b * C_stride_b + (T + p_offsets)[:, None] * C_stride_t + k_offsets[None, :] * C_stride_k
    C_mask = p_mask[:, None] & k_mask[None, :]
    vals = tl.load(C_ptrs, mask=C_mask, other=0.0)

    # Store to out at positions (b, p, k)
    out_ptrs = out_ptr + b * out_stride_b + p_offsets[:, None] * out_stride_p + k_offsets[None, :] * out_stride_k
    tl.store(out_ptrs, vals, mask=C_mask)

# Now the ModelNew that uses Triton kernels
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure device and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."
        B = hidden_states.shape[0]
        P = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, K), "encoder_hidden_states shape must be [B, T, K]"
        assert process_weight.shape == (K, K), "process_weight must be [K, K]"

        # Allocate Acat [B, T+P, K]
        T_plus_P = T + P
        Acat = torch.empty((B, T_plus_P, K), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel
        # Grid: (B, ceil_div(T+P, BLOCK_L), ceil_div(K, BLOCK_K))
        BLOCK_K_cat = 128
        grid_cat = (B, triton.cdiv(T_plus_P, 1), triton.cdiv(K, BLOCK_K_cat))
        _concat_kernel[grid_cat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K_cat,
            num_warps=4, num_stages=2
        )

        # GEMM: Acat [B*(T+P), K] x W_t [K, K] -> C_flat [B*(T+P), K]
        M = B * T_plus_P
        N = K
        # We need to pass Acat as [M, K], but Triton kernel above expects 3D (M,K,N). Better to reshape pointers via strides.
        # Here we'll write a wrapper that launches _matmul_kernel_3d with proper grid:
        # M = B*(T+P), N = K, and we pass Acat strides. W_t is process_weight.T [K, K].
        # Launch matmul kernel
        C_flat = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)

        # Compute grid for matmul: (tiles over M, tiles over N, tiles over K)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_mat = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
        _matmul_kernel_3d[grid_mat](
            Acat, process_weight.t(), C_flat,
            M, K, N,
            Acat.stride(0), Acat.stride(2),  # A strides: (rows=M dim stride, K stride)
            process_weight.t().stride(0), process_weight.t().stride(1),
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )

        # Reshape C_flat to [B, T+P, K]
        C = C_flat.view(B, T_plus_P, K)

        # Allocate outputs
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        # Launch split kernels
        # Split encoder: copy C[:, :T, :]
        BLOCK_T_split = 128
        grid_split_e = (B, triton.cdiv(T, BLOCK_T_split), triton.cdiv(K, 64))
        _split_encoder_kernel[grid_split_e](
            C, processed_encoder,
            B, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_T=BLOCK_T_split, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Split hidden: copy C[:, T:, :]
        BLOCK_P_split = 128
        grid_split_h = (B, triton.cdiv(P, BLOCK_P_split), triton.cdiv(K, 64))
        _split_hidden_kernel[grid_split_h](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_P=BLOCK_P_split, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
