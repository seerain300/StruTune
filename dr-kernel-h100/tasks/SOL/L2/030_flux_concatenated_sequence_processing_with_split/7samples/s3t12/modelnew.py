import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_row_kernel(
    C,           # *ptr* to output [N_rows, K], float32
    A,           # *ptr* to A_rows [N_rows, K], float32
    B,           # *ptr* to process_weight.T [K, K], float32
    N_rows,      # int
    K,           # int
    BLOCK_K: tl.constexpr,
):
    # Each program handles one output row (one (n, t) pair)
    pid = tl.program_id(axis=0)
    if pid >= N_rows:
        return

    # Initialize accumulator for this row
    acc = tl.zeros((1, K), dtype=tl.float32)

    # Reduction over K in chunks
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A_row chunk: shape [BLOCK_K]
        a = tl.load(A + pid * K + k_offsets, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load B chunk: shape [BLOCK_K, K]
        b = tl.load(B + k_offsets[:, None] * K + tl.arange(0, K), mask=mask_k[:, None], other=0.0)  # [BLOCK_K, K]

        # Outer product: [BLOCK_K, 1] * [1, K] -> [BLOCK_K, K], then sum over rows -> [1, K]
        contrib = a[:, None] * b            # [BLOCK_K, K]
        acc += tl.sum(contrib, axis=0)      # [1, K]

        k0 += BLOCK_K

    # Store the accumulated row
    tl.store(C + pid * K + tl.arange(0, K), acc[0, :], mask=True)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim.
        - Apply linear projection using Triton GEMM: concat @ process_weight.T.
        - Split back into separate encoder and image streams.
        """
        # Ensure inputs are on CUDA and contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "Triton kernels require CUDA device."
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate using PyTorch (robust and simple for variable sizes)
        # Cast to float32 for compute to avoid dtype issues in Triton
        enc_f32 = encoder_hidden_states.to(torch.float32)
        hid_f32 = hidden_states.to(torch.float32)
        concatenated = torch.cat([enc_f32, hid_f32], dim=1)  # [N, L_total, K]

        # 2) GEMM via Triton: processed = concatenated @ process_weight.T
        # process_weight: [K, K] (transpose of original [K, K])
        # Ensure process_weight is float32 and contiguous
        B = process_weight.to(torch.float32).t().contiguous()  # [K, K]
        A_rows = concatenated.reshape(N * L_total, K).contiguous()  # [N_rows, K]
        C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)

        # Launch Triton per-row GEMM kernel
        # Choose BLOCK_K based on K
        BLOCK_K = 256 if K >= 256 else 128
        grid = (N * L_total,)
        _matmul_row_kernel[grid](C_rows, A_rows, B, N * L_total, K, BLOCK_K, num_warps=4, num_stages=2)

        # 3) Reshape back to [N, L_total, K]
        processed = C_rows.view(N, L_total, K)

        # 4) Split into two streams
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype of hidden_states (common output dtype)
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden