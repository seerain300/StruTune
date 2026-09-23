import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: concatenate along sequence dimension
# dst[b, t, d] = encoder_hidden_states[b, t, d] if t < L_txt else hidden_states[b, t - L_txt, d]
@triton.jit
def concat_copy_kernel(
    ehs_ptr, hs_ptr, dst_ptr,
    B, L_txt, L_img, D,
    ehs_stride_b, ehs_stride_s, ehs_stride_d,
    hs_stride_b, hs_stride_s, hs_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    # 1D grid over total elements: idx ranges [0, B*M*D)
    idx = tl.program_id(0)
    total = B * M * D
    if idx >= total:
        return

    # Compute b, t, d from linear index
    # Let M = L_txt + L_img
    b = idx // (M * D)
    rem = idx % (M * D)
    t = rem // D
    d = rem % D

    # Skip out-of-batch
    if b >= B:
        return

    # Determine source: encoder or hidden
    is_encoder = t < L_txt
    dst_off = b * dst_stride_b + t * dst_stride_s + d * dst_stride_d

    if is_encoder:
        ehs_off = b * ehs_stride_b + t * ehs_stride_s + d * ehs_stride_d
        val = tl.load(ehs_ptr + ehs_off)
        tl.store(dst_ptr + dst_off, val)
    else:
        hs_idx = t - L_txt
        hs_off = b * hs_stride_b + hs_idx * hs_stride_s + d * hs_stride_d
        val = tl.load(hs_ptr + hs_off)
        tl.store(dst_ptr + dst_off, val)


# Triton copy kernel for split: copy src[:, s, :] into out[:, s, :]
@triton.jit
def copy_seqs_kernel(
    src_ptr, out_ptr,
    B, S, D,
    src_stride_b, src_stride_s, src_stride_d,
    out_stride_b, out_stride_s, out_stride_d,
):
    # 1D grid over total elements: idx ranges [0, B*S*D)
    idx = tl.program_id(0)
    total = B * S * D
    if idx >= total:
        return

    b = idx // (S * D)
    s = (idx % (S * D)) // D
    d = idx % D

    if b >= B or s >= S:
        return

    src_off = b * src_stride_b + s * src_stride_s + d * src_stride_d
    out_off = b * out_stride_b + s * out_stride_s + d * out_stride_d

    val = tl.load(src_ptr + src_off)
    tl.store(out_ptr + out_off, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence (Triton).
        - Perform linear projection using torch.matmul.
        - Split outputs (Triton).
        """
        # If Triton not available or tensors not on CUDA, fall back to PyTorch reference path.
        # The evaluation environment uses CUDA, but we guard for safety.
        if not TRITON_AVAILABLE or not hidden_states.is_cuda:
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight.t())
            processed_encoder = processed[:, :encoder_hidden_states.shape[1]]
            processed_hidden = processed[:, encoder_hidden_states.shape[1]:]
            return processed_encoder, processed_hidden

        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        M = L_txt + L_img

        # Allocate dst for concatenation
        dst = torch.empty((B, M, D), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch Triton concat kernel over a 1D grid
        total_elems = B * M * D
        grid = (total_elems,)
        concat_copy_kernel[grid](
            encoder_hidden_states, hidden_states, dst,
            B, L_txt, L_img, D,
            *encoder_hidden_states.stride(), *hidden_states.stride(),
            *dst.stride(),
            num_warps=4, num_stages=2
        )

        # Linear projection using PyTorch (highly optimized and reliable)
        # W_T: [D, D], processed: [B, M, D]
        W_T = process_weight.transpose(0, 1).contiguous()  # [D, D]
        processed = torch.matmul(dst, W_T)

        # Split into two outputs using Triton copy kernels
        processed_encoder = torch.empty((B, L_txt, D), dtype=processed.dtype, device=hidden_states.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=processed.dtype, device=hidden_states.device)

        # Copy processed[:, :L_txt, :]
        src1_stride_b, src1_stride_s, src1_stride_d = processed.stride()
        out1_stride_b, out1_stride_s, out1_stride_d = processed_encoder.stride()
        total1 = B * L_txt * D
        grid1 = (total1,)
        copy_seqs_kernel[grid1](
            processed, processed_encoder,
            B, L_txt, D,
            int(src1_stride_b), int(src1_stride_s), int(src1_stride_d),
            int(out1_stride_b), int(out1_stride_s), int(out1_stride_d),
            num_warps=4, num_stages=2
        )

        # Copy processed[:, L_txt:, :]
        src2_slice = processed[:, L_txt:, :]  # shape [B, L_img, D]
        src2_stride_b, src2_stride_s, src2_stride_d = src2_slice.stride()
        out2_stride_b, out2_stride_s, out2_stride_d = processed_hidden.stride()
        total2 = B * L_img * D
        grid2 = (total2,)
        copy_seqs_kernel[grid2](
            src2_slice, processed_hidden,
            B, L_img, D,
            int(src2_stride_b), int(src2_stride_s), int(src2_stride_d),
            int(out2_stride_b), int(out2_stride_s), int(out2_stride_d),
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
