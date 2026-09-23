import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,          # *fp32, [N, L_total, K]
    enc_ptr,          # *fp32, [N, L_txt, K]
    hid_ptr,          # *fp32, [N, L_img, K]
    L_txt: tl.constexpr,
    L_img: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    # grid: (N, L_total, tiles_along_K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    k_block = tl.program_id(2)

    # tile along K dimension
    k_start = k_block * 128  # tile size 128; grid third dim ensures full K coverage
    k_offsets = k_start + tl.arange(0, 128)
    mask = k_offsets < K

    # Choose source based on t < L_txt
    is_text = t < L_txt
    base_src = enc_ptr + n * (L_txt * K) if is_text else hid_ptr + n * (L_img * K)

    # Compute addresses for load and store
    # For enc: src[n, t, k] => pointer = base_src + t*K + k
    # For hid: src[n, t-L_txt, k] => pointer = base_src + (t - L_txt)*K + k
    if is_text:
        src_ptrs = base_src + t * K + k_offsets
    else:
        src_ptrs = base_src + (t - L_txt) * K + k_offsets

    # Destination: out[n, t, k] => pointer = out_ptr + n*(L_total*K) + t*K + k
    out_base = out_ptr + n * (L_txt + L_img) * K + t * K
    dst_ptrs = out_base + k_offsets

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


@triton.jit
def _matmul_row_kernel(
    C_row_ptr,        # *fp32, [N_rows, K], each row is length K
    A_row_ptr,        # *fp32, [N_rows, K], each row is length K
    B_ptr,            # *fp32, [K, K]
    N_rows: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program instance computes one output row (corresponds to one (n, t) pair)
    row_id = tl.program_id(0)
    if row_id >= N_rows:
        return

    # Accumulator for this row
    acc = tl.zeros([K], dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A[row_id, :] chunk
        a_ptrs = A_row_ptr + row_id * K + k_offsets
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)  # shape [BLOCK_K], fp32

        # B[:, :] chunk (BLOCK_K x BLOCK_K)
        b_ptrs = B_ptr + k_offsets[:, None] * K + tl.arange(0, BLOCK_K)[None, :]  # shape [BLOCK_K, BLOCK_K]
        b_mask = k_mask[:, None]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # fp32

        # Multiply: [BLOCK_K, 1] * [BLOCK_K, BLOCK_K] -> [BLOCK_K, BLOCK_K]
        # Then sum along axis=1 -> [BLOCK_K]
        partial = tl.dot(a[:, None], b)[0, :]  # multiply each a[i] with the corresponding b[i, :] vector
        acc += partial

        k0 += BLOCK_K

    # Store accumulated result for this row
    c_ptrs = C_row_ptr + row_id * K + tl.arange(0, K)
    tl.store(c_ptrs, acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        - Apply linear projection via GEMM (no bias).
        - Split back into separate encoder and image streams.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be 3D: [N, L, K]"
        N = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_txt = encoder_hidden_states.shape[1]
        assert encoder_hidden_states.shape[0] == N and encoder_hidden_states.shape[2] == K, "Mismatched shapes"
        assert process_weight.shape == (K, K), "process_weight must be [K, K]"

        # Ensure inputs are contiguous and on CUDA
        device = hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        # We'll compute in fp32; cast process_weight if needed
        B = process_weight.contiguous()
        if B.dtype != torch.float32:
            B = B.float()

        # 1) Concatenate into out_cat [N, L_total, K] in fp32
        L_total = L_txt + L_img
        out_cat = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        grid_concat = (N, L_total, triton.cdiv(K, 128))
        _concat_sequences_kernel[grid_concat](
            out_cat, enc, hid, L_txt, L_img, K, N,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: out_cat @ B, where out_cat is [N_rows, K] with N_rows = N * L_total
        concatenated = out_cat.reshape(N * L_total, K).contiguous()  # [N_rows, K]
        N_rows = N * L_total
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Choose reduction tile
        BLOCK_K = 128 if K >= 128 else 64

        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, concatenated, B, N_rows, K,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype if needed
        if processed_encoder.dtype != hidden_states.dtype:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
        if processed_hidden.dtype != hidden_states.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
