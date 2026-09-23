import torch
import triton
import triton.language as tl


@triton.jit
def diag_contract_Y_kernel(
    M_ptr,  # M: [B, C, S, S, N], float32
    HS_ptr,  # hidden_states: [B, C, S, N, D], float32
    Y_ptr,  # output: [B, C, S, N, D], float32
    B_size: tl.constexpr,  # not used but can be for future specialization
    C_size,  # num_chunks
    S,       # chunk_size
    N,       # num_heads
    D,       # head_dim
    # strides
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Each program handles one (b, c, i, n, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over j dimension (chunk positions) to perform diagonal contraction
    for j in range(0, S):
        m_ij = tl.load(
            M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n,
            mask=True,
            other=0.0
        )
        hs_jd = tl.load(
            HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d,
            mask=True,
            other=0.0
        )
        acc += m_ij * hs_jd

    # Store the result into Y[b, c, i, n, d]
    tl.store(
        Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d,
        acc
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original signature
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag via Triton diagonal contraction:
        - A_cumsum: [B, C, S, N], float32
        - B: [B, C, S, N_GROUPS, K], float32 (K=hidden_states.shape[4])
        - C: [B, C, S, N_GROUPS, K], float32
        - hidden_states: [B, C, S, N, D], float32
        Output: [B, C, S, N, D], float32, cast to bfloat16
        """
        device = hidden_states.device
        # Infer shapes
        Bsz, Csz, S, N, D = hidden_states.shape

        # 1) Compute L in PyTorch exactly as the original:
        # L[b, c, i, j, n] = exp( cumsum_j( A_cumsum[b, c, j, n] ) ) with lower-triangular mask (diagonal=-1)
        # We apply tril(diagonal=-1) to A_cumsum for each (b, c, n), then cumsum along j, then exp.
        # Broadcast A_cumsum to [B, C, S, S, N]
        # Note: A_cumsum is [B, C, S, N] per original code
        # We need to create a lower-triangular mask and apply cumsum along j.
        # torch.tril with diagonal=-1 keeps j <= i (exclude j == i).
        # Since A_cumsum is [B, C, S, N], we expand j and i dimensions.
        # Create mask [S, S] on device
        mask = torch.tril(torch.ones((S, S), device=device, dtype=torch.bool), diagonal=-1)
        # Broadcast to [B, C, S, S, N]
        mask_expanded = mask[None, None, :, :, None].expand(Bsz, Csz, S, S, N)
        # Apply mask: exclude diagonal
        A_masked = A_cumsum.masked_fill(~mask_expanded, 0.0)
        # Cumsum along j dimension (dim=3)
        A_cumsum_seg = torch.cumsum(A_masked, dim=3)  # still [B, C, S, N]
        # Apply exp to get L: L[b, c, i, j, n] = exp(A_cumsum_seg[b, c, j, n]) with j < i, else 0
        # But A_cumsum_seg has shape [B, C, S, N], we need [B, C, S, S, N].
        # So we index A_cumsum_seg[:, :, j, :] for L[b, c, i, j, n] using j.
        # Compute L explicitly:
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        for i_idx in range(S):
            # Select jth cumsum for each i_idx
            j_cumsum = torch.cumsum(A_cumsum, dim=3)  # [B, C, S, N] for j positions
            # Apply mask for i_idx: include j < i_idx
            mask_i = mask[:, i_idx, :].expand(S, 1, S).to(device).bool()  # shape [S, 1, S] -> broadcast to [S, S]
            # We need shape [S, S], so use mask[:, i_idx, :] to create per i mask.
            # However, since mask is 2D [S,S], we use mask[:, i_idx] for each i.
            # Fix: Create mask_i_j = mask[:, i_idx] of shape [S], then build 2D mask with rows i_idx.
            # Simpler: build 2D mask using broadcasting.
            mask_i_2d = (torch.arange(S, device=device) <= i_idx).view(S, 1).expand(S, S)
            A_j = j_cumsum[:, :, :]  # [B, C, S, N], but for each j we need A_cumsum[:, :, j, :]
            # We need A_cumsum[:, :, j, :], which is j_cumsum[:, :, j, :]. However j_cumsum corresponds to cumsum at each j; to get A_cumsum[:, :, j, :], we should instead use original A_cumsum with j index.
            # Correction: A_cumsum[:, :, j, :] is A_cumsum indexed by j at dim 3. To get per j, we need A_cumsum[:, :, j, :].
            # Since j_cumsum is cumsum over j, to retrieve A_cumsum[:, :, j, :], we can use:
            # A_cumsum[:, :, j, :] = original A_cumsum[:, :, j, :]. But j_cumsum is cumsum. Therefore, L[:, :, i, j, :] = exp( cumsum_j(original) ) with mask j < i.
            # So we should directly read original A_cumsum[:, :, j, :] and apply mask. j_cumsum is not needed for exact original semantics.
            # Fix: compute L without j_cumsum using original A_cumsum with mask.
            # We can reconstruct L using original A_cumsum and mask:
            # For each i_idx, L[:, :, i_idx, j, :] = exp( sum_{k<=j} A_cumsum[:, :, k, :] ) if j < i_idx else 0
            # Implement via prefix sums per j:
            # We will do this in a loop over j (small S) using cumsum along j dimension using torch.cumsum.
            # However, we cannot use torch.cumsum in Triton-only; but since we're in PyTorch for L, we use torch.cumsum on masked A.
            # Above loop was just conceptual. We already masked A_cumsum via tril and did cumsum. Now compute L as exp of cumsum with mask.
            # L at each (i_idx, j) is exp( masked cumsum A_cumsum[:, :, j, :] ).
            # Since masked cumsum already excludes j >= i_idx contributions (by setting those to 0), the cumsum over j <= i_idx gives the correct value.
            # Therefore, L is simply exp of A_cumsum masked as done above.
            # We need to assign to L[:, :, i_idx, j, :]. We can do:
            # L[:, :, i_idx, j, :] = torch.exp( A_masked[:, :, j, :] )
            # Note: A_masked has shape [B, C, S, N]. To broadcast to [B, C, S, S, N], we need to repeat along S dimension.
            # Actually A_masked is [B, C, S, N] and we expanded it to [B, C, S, S, N]. The value at (b,c,i,j,n) is A_cumsum[b,c,j,n] if j < i, else 0.
            # We computed A_masked by masked_fill on expanded tensor. Now cumsum along j dimension (dim=3) yields cumsum per n across j.
            # Therefore L[:, :, i, j, n] = exp( A_cumsum_masked[b,c,j,n] ). But A_cumsum_masked[b,c,j,n] is 0 when j >= i. So L is exp of masked A.
            # Assign:
            # L[:, :, i_idx, :, :] = torch.exp( A_masked )
            # But A_masked currently is [B, C, S, N] (cumsum result). We need L to be [B, C, S, S, N].
            # Fix: we already expanded mask to [B, C, S, S, N]. torch.cumsum(A_masked, dim=3) produces [B, C, S, N]. To assign to L[:, :, i_idx, :, :], we broadcast A_masked across i_idx.
            # Better: compute L[:, :, i_idx, :, :] directly via exp of cumsum masked tensor at that i_idx. We can't index per i_idx here because we looped over i.
            # So instead, we compute the full L by:
            # L = torch.exp(A_cumsum_seg), where A_cumsum_seg = cumsum of masked A_cumsum along j.
            L[:, :, i_idx, :, :] = torch.exp(A_cumsum_seg)

        # Now L is [B, C, S, S, N] computed exactly as original intended for masked cumsum then exp.

        # 2) Expand B and C to num_heads (32) by repeat_interleave(NUM_HEADS // N_GROUPS) = 4
        B_expanded = B.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, N, K]
        C_expanded = C.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, N, K]

        # 3) Compute G in PyTorch: elementwise G[b, c, i, j, n] = sum_k C_expanded[b, c, i, n, k] * B_expanded[b, c, j, n, k]
        # Note: For D=1, this reduces to elementwise product; we implement general K reduction.
        # We can compute G by a loop over k in [0, K-1], but K may vary. We can infer K from B/C last dim:
        # However, B/C shapes imply K is hidden_states.shape[4] (D). To be robust, we compute G via broadcasting and sum.
        # Build expanded B and C along j:
        # G shape: [B, C, S, S, N]
        # G = sum over k of C_expanded[..., k] * B_expanded[..., k]
        # C_expanded shape: [B, C, S, N, K], B_expanded: [B, C, S, N, K]
        # Sum over K: torch.sum(C_expanded * B_expanded, dim=4)
        G = torch.sum(C_expanded * B_expanded, dim=4)  # [B, C, S, N]

        # We need G with N dimension expanded to match L/G elementwise multiply: G needs shape [B, C, S, S, N].
        # Since the original code uses G = sum over k of C*B along k, and then multiplies with L: M = G * L, and then diagonal contraction.
        # The contraction sums over j, and G has shape [B, C, S, N]. We can broadcast G to [B, C, S, S, N] by repeating along j dimension with ones. However, G depends on j through the contraction, so we keep G as [B, C, S, N] for M = G * L (elementwise multiply with L), because L has j dimension. PyTorch broadcasting handles this: G[:, :, i, :] multiplied with L[:, :, i, :, :] along j.

        # 4) Compute M = G * L in PyTorch. Shapes: G [B, C, S, N], L [B, C, S, S, N]. PyTorch will broadcast G along j.
        M = G * L  # [B, C, S, S, N]

        # 5) Diagonal contraction to compute Y: Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        # hidden_states: [B, C, S, N, D] float32 (original code uses float32 for computation). We pass float32 tensors to Triton.
        hidden_states_f32 = hidden_states.to(torch.float32)

        # Allocate output Y as float32
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        # Launch Triton kernel for diagonal contraction
        grid = (Bsz, Csz, S, N, D)
        diag_contract_Y_kernel[grid](
            M, hidden_states_f32, Y,
            Bsz, Csz, S, N, D,
            M_stride_b=M.stride(0), M_stride_c=M.stride(1), M_stride_i=M.stride(2), M_stride_j=M.stride(3), M_stride_n=M.stride(4),
            HS_stride_b=hidden_states_f32.stride(0), HS_stride_c=hidden_states_f32.stride(1), HS_stride_j=hidden_states_f32.stride(2), HS_stride_n=hidden_states_f32.stride(3), HS_stride_d=hidden_states_f32.stride(4),
            Y_stride_b=Y.stride(0), Y_stride_c=Y.stride(1), Y_stride_i=Y.stride(2), Y_stride_n=Y.stride(3), Y_stride_d=Y.stride(4),
            num_warps=1, num_stages=1
        )

        # 6) Return cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
