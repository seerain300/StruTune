import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,           # *encoder_hidden_states: [B, L_txt, D]
    hs_ptr,            # *hidden_states: [B, L_img, D]
    dst_ptr,           # *dst: [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
):
    # 2D grid: axis 0 over batch, axis 1 over tiles of sequence length
    b = tl.program_id(axis=0)
    seq_tile = tl.program_id(axis=1)
    seq_off = seq_tile * 128 + tl.arange(0, 128)
    valid_seq = seq_off < (L_txt + L_img)
    b_mask = b < B

    # Determine whether current seq_off belongs to encoder or hidden part
    is_encoder = seq_off < L_txt
    enc_mask = b_mask & is_encoder & valid_seq
    hid_mask = b_mask & (~is_encoder) & valid_seq

    # d dimension
    d = tl.arange(0, 128)
    valid_d = d < D

    # Destination base pointers
    dst_base = dst_ptr + b * dst_stride_b + seq_off[:, None] * dst_stride_s + d[None, :] * dst_stride_d

    # Masks for load/store
    load_store_mask = (enc_mask[:, None] | hid_mask[:, None]) & valid_d[None, :]

    # Source base pointers
    # Encoder source: ehs[b, seq_off, :]
    ehs_base = ehs_ptr + b * ehs_stride_b + seq_off[:, None] * ehs_stride_s + d[None, :] * ehs_stride_d
    # Hidden source: hs[b, seq_off - L_txt, :]
    hs_seq_idx = seq_off - L_txt
    hs_base = hs_ptr + b * hs_stride_b + hs_seq_idx[:, None] * hs_stride_s + d[None, :] * hs_stride_d

    # Perform masked loads and store into dst
    ehs_vals = tl.load(ehs_base, mask=enc_mask[:, None] & valid_d[None, :], other=0.0)
    hs_vals = tl.load(hs_base, mask=hid_mask[:, None] & valid_d[None, :], other=0.0)
    # Combined values: where enc_mask True -> ehs_vals, else hs_vals
    # Triton allows elementwise operations; we use masked loads above. For store, we only write at positions where load_store_mask is True.
    # Triton masked load fills zeros where mask is False; combined store with load_store_mask writes zeros for non-matching positions.
    # To ensure correct values, we store using masked loads; when a mask is False, value is 0. Since output must be exact, we avoid masked mixed store.
    # Implement separate stores for encoder and hidden to be explicit:
    # Note: Triton doesn't support per-element conditional store with pointer selection, but masked loads ensure correctness when we store with combined mask.
    # However, to be extra safe, we compute vals explicitly:
    # vals = where(is_encoder, ehs_vals, hs_vals), but Triton doesn't support pointer-based where.
    # Triton supports tl.where for data, but not for pointer selection; masked loads with combined mask is the safest approach.
    # We'll store using combined mask; masked loads for enc/hid produce zeros where mask is False, and store will write zeros where necessary.
    # This approach works: masked loads ensure ehs_vals/hs_vals contain the correct data for positions where mask is True; store with combined mask writes those.
    tl.store(dst_base, tl.where(enc_mask[:, None], ehs_vals, hs_vals), mask=load_store_mask)


@triton.jit
def split_seqs_kernel(
    src_ptr,           # *processed: [B, L, D], L = L_txt + L_img
    out1_ptr,          # *processed_encoder: [B, L_txt, D]
    out2_ptr,          # *processed_hidden: [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    src_stride_b: tl.int32, src_stride_s: tl.int32, src_stride_d: tl.int32,
    out1_stride_b: tl.int32, out1_stride_s: tl.int32, out1_stride_d: tl.int32,
    out2_stride_b: tl.int32, out2_stride_s: tl.int32, out2_stride_d: tl.int32,
):
    # 3D grid: (batch, tiles along L_txt, tiles along D)
    b = tl.program_id(axis=0)
    s_off = tl.program_id(axis=1) * 128 + tl.arange(0, 128)
    d_off = tl.program_id(axis=2) * 128 + tl.arange(0, 128)

    valid_s = s_off < L_txt
    valid_d = d_off < D
    b_mask = b < B

    # Encode output
    src_base_e = src_ptr + b * src_stride_b + s_off[:, None] * src_stride_s + d_off[None, :] * src_stride_d
    out1_base = out1_ptr + b * out1_stride_b + s_off[:, None] * out1_stride_s + d_off[None, :] * out1_stride_d
    load_mask_e = b_mask & valid_s[:, None] & valid_d[None, :]
    vals_e = tl.load(src_base_e, mask=load_mask_e, other=0.0)
    tl.store(out1_base, vals_e, mask=load_mask_e)

    # Hidden output: indices s_off = s in [L_txt, L_txt+L_img)
    src_base_h = src_ptr + b * src_stride_b + (s_off + L_txt)[:, None] * src_stride_s + d_off[None, :] * src_stride_d
    out2_base = out2_ptr + b * out2_stride_b + s_off[:, None] * out2_stride_s + d_off[None, :] * out2_stride_d
    load_mask_h = b_mask & (s_off < L_img)[:, None] & valid_d[None, :]
    vals_h = tl.load(src_base_h, mask=load_mask_h, other=0.0)
    tl.store(out2_base, vals_h, mask=load_mask_h)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension using Triton.
        - Computes the linear projection using torch.matmul (GPU), to ensure correctness across varied shapes.
        - Splits the processed tensor into processed_encoder and processed_hidden using Triton.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be 3D tensors"
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == D, "Mismatched batch or hidden_dim"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # 1) Concatenate along sequence dimension using Triton: dst [B, L_txt + L_img, D]
        M = L_txt + L_img
        dst = torch.empty((B, M, D), dtype=hidden_states.dtype, device=hidden_states.device)

        ehs = encoder_hidden_states
        hs = hidden_states

        ehs_stride_b, ehs_stride_s, ehs_stride_d = ehs.stride()
        hs_stride_b, hs_stride_s, hs_stride_d = hs.stride()
        dst_stride_b, dst_stride_s, dst_stride_d = dst.stride()

        grid_concat = (B, triton.cdiv(M, 128))
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst,
            B, L_txt, L_img, D,
            int(ehs_stride_b), int(ehs_stride_s), int(ehs_stride_d),
            int(hs_stride_b), int(hs_stride_s), int(hs_stride_d),
            int(dst_stride_b), int(dst_stride_s), int(dst_stride_d),
            num_warps=4, num_stages=2
        )

        # 2) Compute processed = dst @ process_weight.T using torch (GPU). Ensure weight is on the same device.
        Wt = process_weight.transpose(0, 1).contiguous()  # [D, D], same dtype as process_weight
        processed = torch.matmul(dst, Wt)  # [B, M, D], dtype follows dst

        # 3) Split using Triton into processed_encoder [B, L_txt, D] and processed_hidden [B, L_img, D]
        processed_encoder = torch.empty((B, L_txt, D), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=processed.dtype, device=processed.device)

        src_stride_b, src_stride_s, src_stride_d = processed.stride()
        out1_stride_b, out1_stride_s, out1_stride_d = processed_encoder.stride()
        out2_stride_b, out2_stride_s, out2_stride_d = processed_hidden.stride()

        grid_split = (B, triton.cdiv(L_txt, 128), triton.cdiv(D, 128))
        split_seqs_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            int(src_stride_b), int(src_stride_s), int(src_stride_d),
            int(out1_stride_b), int(out1_stride_s), int(out1_stride_d),
            int(out2_stride_b), int(out2_stride_s), int(out2_stride_d),
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
