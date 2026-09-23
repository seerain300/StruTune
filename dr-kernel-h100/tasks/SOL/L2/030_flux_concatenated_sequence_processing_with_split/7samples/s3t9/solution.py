import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,  # *T, shape [N, L_total, K]
    enc_ptr,  # *T, shape [N, L_txt, K]
    hid_ptr,  # *T, shape [N, L_img, K]
    N, L_txt, L_img, K,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, L_total, tiles_along_K)
    pid_n = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # concatenated sequence index
    pid_k = tl.program_id(2)  # tile along K

    # Compute k offsets for this tile
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source tensor: first L_txt rows from encoder, rest from hidden
    is_encoder = pid_t < L_txt
    src_n = pid_n
    src_t = pid_t  # for encoder
    # For hidden: corresponding source t index is pid_t - L_txt
    src_t_hid = pid_t - L_txt

    # Compute base offsets
    out_base = (pid_n * L_total + pid_t) * K
    enc_base = src_n * (L_txt * K) + src_t * K
    hid_base = src_n * (L_img * K) + src_t_hid * K

    # Load from source
    val = tl.load(
        enc_ptr + enc_base + k_offsets if is_encoder else hid_ptr + hid_base + k_offsets,
        mask=mask_k,
        other=0.0,
    )

    # Store to output
    tl.store(out_ptr + out_base + k_offsets, val, mask=mask_k)


@triton.jit
def _matmul_row_kernel(
    C_ptr,  # *float32, shape [N_rows, K]
    A_ptr,  # *float32, shape [N_rows, K]
    B_ptr,  # *float32, shape [K, K]
    N_rows: tl.constexpr,  # not used, but kept for signature symmetry
    K: tl.constexpr,       # reduction dimension
    BLOCK_K: tl.constexpr, # chunk size for reduction
):
    pid_row = tl.program_id(0)  # row index in [0, N_rows)
    # Accumulator
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    # Iterate over K in chunks
    i = 0
    while i < K:
        k_offsets = i + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A row chunk
        a = tl.load(A_ptr + pid_row * K + k_offsets, mask=mask_k, other=0.0)

        # Load B chunk (columns k_offsets)
        b = tl.load(B_ptr + k_offsets[:, None] * K + k_offsets[None, :], mask=mask_k[:, None], other=0.0)

        # Accumulate dot product for this chunk: sum_j A[i,j] * B[j,k]
        # b shape [BLOCK_K, BLOCK_K], a shape [BLOCK_K], we sum over axis=1
        # We'll implement elementwise multiply and reduce along axis=1
        # Note: Triton will handle broadcasting; ensure we sum correctly.
        # Manual outer-product accumulation over BLOCK_K
        # Initialize acc from first chunk
        if i == 0:
            # Single-chunk init
            # Compute dot: sum(a * b, axis=1)
            acc = tl.sum(a[:, None] * b, axis=1)
        else:
            # For subsequent chunks, compute partial dot and add
            a_i = a
            b_i = b
            # Sum outer products: a_i * b_i[:, k]
            # Reduce along axis=1
            partial = tl.sum(a_i[:, None] * b_i, axis=1)
            acc += partial

        i += BLOCK_K

    # Store accumulated result
    tl.store(C_ptr + pid_row * K + tl.arange(0, BLOCK_K), acc, mask=tl.arange(0, BLOCK_K) < K)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim in Triton.
        - Compute processed = concatenated @ process_weight.T in Triton (per-row loop).
        - Split into encoder and hidden outputs.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton."
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        device = hidden_states.device

        # 1) Concatenate in Triton: out_cat [N, L_total, K]
        out_cat = torch.empty((N, L_total, K), device=device, dtype=hidden_states.dtype)
        BLOCK_K = 256 if K >= 256 else 128
        grid = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid](
            out_cat, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = out_cat @ process_weight.T in Triton (per-row kernel)
        # We'll do accumulation in float32 for numerical stability; inputs/outputs are float32 in typical cases.
        # If dtypes differ, we'll cast to float32 for compute and cast back at the end.
        # Create flattened A_rows = out_cat.reshape(N_rows, K), B = process_weight.t()
        N_rows = N * L_total
        A_rows = out_cat.reshape(N_rows, K).contiguous()  # [N_rows, K]
        # Ensure process_weight.T is contiguous [K, K]
        B = process_weight.t().contiguous()  # [K, K], same dtype/device

        # Allocate output rows as float32
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        BLOCK_K_GEMM = 256 if K >= 256 else 128
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B, N_rows, K, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)

        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Match original dtypes for outputs (inputs are typically float32; if not, cast back)
        if processed_encoder.dtype != encoder_hidden_states.dtype:
            processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        if processed_hidden.dtype != hidden_states.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
