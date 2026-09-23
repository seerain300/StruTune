import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: concatenate along sequence dimension for each batch.
# dst[:, :L_txt, :] = encoder_hidden_states
# dst[:, L_txt:, :] = hidden_states
@triton.jit
def concat_seqs_kernel(
    encoder_ptr, hidden_ptr, dst_ptr,
    B, L_txt, L_img, D,
    enc_stride_b, enc_stride_s, enc_stride_d,
    hid_stride_b, hid_stride_s, hid_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_s_tile = tl.program_id(1)  # tile along sequence length
    pid_d_tile = tl.program_id(2)  # tile along hidden dim

    s_offsets = pid_s_tile * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    d_offsets = pid_d_tile * 64 + tl.arange(0, 64)  # hidden dim block (64)

    M = L_txt + L_img
    valid_s = s_offsets < M
    valid_d = d_offsets < D
    mask = valid_s[:, None] & valid_d[None, :]

    # Base pointers for this batch
    encoder_base = encoder_ptr + pid_b * enc_stride_b
    hidden_base = hidden_ptr + pid_b * hid_stride_b
    dst_base = dst_ptr + pid_b * dst_stride_b

    # Determine which source to use per s: if s < L_txt, use encoder; else use hidden with offset L_txt
    s_lt = s_offsets < L_txt
    s_ge = s_offsets >= L_txt

    # Prepare addresses for loads/stores
    enc_addr = encoder_base + (s_offsets[:, None] * enc_stride_s + d_offsets[None, :] * enc_stride_d)
    hid_addr = hidden_base + ((s_offsets[:, None] + L_txt) * hid_stride_s + d_offsets[None, :] * hid_stride_d)
    dst_addr = dst_base + (s_offsets[:, None] * dst_stride_s + d_offsets[None, :] * dst_stride_d)

    # Masks for each branch
    mask_lt = (s_lt[:, None] & valid_d[None, :])  # [BLOCK_S, 64]
    mask_ge = (s_ge[:, None] & valid_d[None, :])  # [BLOCK_S, 64]

    # Load and store branches; Triton requires elementwise masks; we can load with mask and store with mask
    if tl.any(mask_lt):
        vals_lt = tl.load(encoder_ptr + enc_addr, mask=mask_lt, other=0.0)
        tl.store(dst_ptr + dst_addr, vals_lt, mask=mask_lt)
    if tl.any(mask_ge):
        vals_ge = tl.load(hidden_ptr + hid_addr, mask=mask_ge, other=0.0)
        tl.store(dst_ptr + dst_addr, vals_ge, mask=mask_ge)


# Triton kernel: split processed tensor into two outputs.
# processed_encoder = processed[:, :L_txt, :]
# processed_hidden = processed[:, L_txt:, :]
@triton.jit
def split_seqs_kernel(
    src_ptr, out1_ptr, out2_ptr,
    B, L_txt, L_img, D,
    src_stride_b, src_stride_s, src_stride_d,
    out1_stride_b, out1_stride_s, out1_stride_d,
    out2_stride_b, out2_stride_s, out2_stride_d,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_s_tile = tl.program_id(1)  # tile along sequence length
    pid_d_tile = tl.program_id(2)  # tile along hidden dim

    s_offsets = pid_s_tile * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    d_offsets = pid_d_tile * 64 + tl.arange(0, 64)  # hidden dim block (64)

    M = L_txt + L_img
    valid_s = s_offsets < M
    valid_d = d_offsets < D
    mask = valid_s[:, None] & valid_d[None, :]

    src_base = src_ptr + pid_b * src_stride_b
    out1_base = out1_ptr + pid_b * out1_stride_b
    out2_base = out2_ptr + pid_b * out2_stride_b

    # Copy encoder part: src[:, :L_txt, :] -> out1
    s_lt = s_offsets < L_txt
    mask_lt = s_lt[:, None] & valid_d[None, :]
    if tl.any(mask_lt):
        vals = tl.load(src_ptr + (src_base + s_offsets[:, None] * src_stride_s + d_offsets[None, :] * src_stride_d), mask=mask_lt, other=0.0)
        tl.store(out1_ptr + (out1_base + s_offsets[:, None] * out1_stride_s + d_offsets[None, :] * out1_stride_d), vals, mask=mask_lt)

    # Copy hidden part: src[:, L_txt:, :] -> out2
    s_ge = s_offsets >= L_txt
    mask_ge = s_ge[:, None] & valid_d[None, :]
    if tl.any(mask_ge):
        vals = tl.load(src_ptr + (src_base + (s_offsets[:, None] + L_txt) * src_stride_s + d_offsets[None, :] * src_stride_d), mask=mask_ge, other=0.0)
        tl.store(out2_ptr + (out2_base + (s_offsets[:, None] - L_txt) * 0 + d_offsets[None, :] * out2_stride_d), vals, mask=mask_ge)  # correction below
        # Correct store: out2 uses s_offsets - L_txt in output indexing; but output s dimension is L_txt/L_img, not src s. So we write based on out s indices.
        # We need to map src s to out s: for out1, s < L_txt; for out2, s - L_txt maps to out s in [0, L_img). Let's recompute with correct out s.
        # Simpler: since out1/out2 are newly allocated with L_txt/L_img, we can use out s indices directly. However, our split stores map src s to out s as:
        # out1: s <= L_txt-1, out2: s >= L_txt. For out2, we write to out2 at row s - L_txt, using src data at s.
        # Implement corrected store below:
        # Compute out2 dst addresses: out2 row index is s - L_txt for src s, and d_offsets for hidden dim.
        tl.store(out2_ptr + (out2_base + (s_offsets[:, None] - L_txt) * src_stride_s + d_offsets[None, :] * out2_stride_d), vals, mask=mask_ge)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure inputs are on the same device and contiguous
        B = hidden_states.shape[0]
        D = hidden_states.shape[2]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        M = L_txt + L_img

        # 1) Concatenate encoder_hidden_states and hidden_states along sequence dimension using Triton.
        dst = torch.empty((B, M, D), dtype=hidden_states.dtype, device=hidden_states.device)

        if TRITON_AVAILABLE:
            enc = encoder_hidden_states
            hid = hidden_states
            # Strides
            enc_stride_b, enc_stride_s, enc_stride_d = enc.stride()
            hid_stride_b, hid_stride_s, hid_stride_d = hid.stride()
            dst_stride_b, dst_stride_s, dst_stride_d = dst.stride()
            # Launch concatenation kernel: grid over (B, tiles of M, tiles of D)
            BLOCK_S = 128
            BLOCK_D = 64
            grid_concat = (B, triton.cdiv(M, BLOCK_S), triton.cdiv(D, BLOCK_D))
            concat_seqs_kernel[grid_concat](
                enc, hid, dst,
                B, L_txt, L_img, D,
                enc_stride_b, enc_stride_s, enc_stride_d,
                hid_stride_b, hid_stride_s, hid_stride_d,
                dst_stride_b, dst_stride_s, dst_stride_d,
                BLOCK_S=BLOCK_S,
                num_warps=4, num_stages=2
            )
        else:
            # Fallback: torch.cat if Triton not available
            dst = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 2) Apply linear projection using PyTorch matmul (robust and fast). Keep dtype/device consistent.
        # process_weight is [D, D]; we need A @ process_weight.T, so transpose process_weight.
        weight_t = process_weight.transpose(0, 1).contiguous()  # [D, D] -> [D, D], already correct
        processed = torch.matmul(dst, weight_t)  # [B, M, D]

        # 3) Split outputs using Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=processed.dtype, device=processed.device)

        src_stride_b, src_stride_s, src_stride_d = processed.stride()
        out1_stride_b, out1_stride_s, out1_stride_d = processed_encoder.stride()
        out2_stride_b, out2_stride_s, out2_stride_d = processed_hidden.stride()

        BLOCK_S_SPLIT = 128
        BLOCK_D_SPLIT = 64
        grid_split = (B, triton.cdiv(L_txt, BLOCK_S_SPLIT), triton.cdiv(D, BLOCK_D_SPLIT))
        split_seqs_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            src_stride_b, src_stride_s, src_stride_d,
            out1_stride_b, out1_stride_s, out1_stride_d,
            out2_stride_b, out2_stride_s, out2_stride_d,
            BLOCK_S=BLOCK_S_SPLIT,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
