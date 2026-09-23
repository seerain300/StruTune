import torch
import triton
import triton.language as tl

@triton.jit
def concat_linear_kernel_tiled(
    image_ptr,           # *f16/f32 [B, I, H]
    encoder_ptr,         # *f16/f32 [B, T, H]
    weight_ptr,          # *f16/f32 [H, H]
    out_ptr,             # *f16/f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_T: tl.constexpr,  # number of output tokens per program
    BLOCK_H: tl.constexpr,  # number of hidden dims per program
    BLOCK_K: tl.constexpr,  # K-chunk size for accumulation
):
    # program ids
    pid_b = tl.program_id(0)
    pid_t_blk = tl.program_id(1)
    pid_h_blk = tl.program_id(2)

    # compute token range for this program
    t_start = pid_t_blk * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < (T + I)

    # compute hidden offsets for this program
    h_start = pid_h_blk * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # initialize output accumulator [BLOCK_T, BLOCK_H] in float32
    output = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # loop over K (input hidden dimension) in chunks
    for k_off in range(0, H, BLOCK_K):
        k_offsets = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Determine which tokens come from encoder (t < T) and image (t >= T)
        # We compute per-token masks: valid_token indicates whether t is a real token (t_mask)
        valid_token = t_mask  # [BLOCK_T]
        encoder_mask = t_offsets < T  # [BLOCK_T]
        # For each token, compute its source index into image or encoder:
        # src_i = t_offsets - T for tokens >= T; else t_offsets (but we only use when in range)
        src_i = t_offsets - T  # [BLOCK_T]
        # Build per-token pointers for input vectors (will be masked)
        # Input ptr: base + src_index * stride_i
        # We'll select based on encoder_mask; for tokens >= T, src_i must be in [0, I).
        # Note: We don't store the input vectors; we load them lazily per chunk.

        # Load input vector chunks for both streams and select per token.
        # We construct two candidate pointers and use tl.where to select.
        # For tokens < T, source is encoder[b, t, k:k+BLOCK_K], else source is image[b, t-T, k:k+BLOCK_K].
        # Build 2D pointers of shape [BLOCK_T, BLOCK_K] for each stream.

        # Pointer base for encoder and image
        # For tokens >= T, src_i = t_offsets - T, else src_i = t_offsets (but we guard with valid_token & encoder_mask)
        # Note: Triton allows pointer arithmetic with boolean masks; we mask with where.
        # First, compute ptrs for encoder and image separately.
        # ptrs shape: [BLOCK_T, BLOCK_K]

        # We need to build ptrs for each stream and select. Triton supports advanced indexing when combining aranges.
        # Create a [BLOCK_T, 1] pointer for each stream using t_offsets and k_offsets.
        # Stream 1: encoder
        # stream 2: image
        # We'll use tl.where to select per token.

        # Construct [BLOCK_T, 1] pointers using t_offsets[:, None], then expand over k_offsets via [:, None].
        # For stream 1 (encoder), t_offsets < T decides.
        # For stream 2 (image), t_offsets >= T decides.

        # Build the two candidate ptrs:
        # ptr_enc: encoder_ptr + pid_b*encoder_stride_b + (t_offsets[None, :])*(encoder_stride_t) + (k_offsets[:, None])*(encoder_stride_h)
        ptr_enc = encoder_ptr + pid_b * encoder_stride_b + (t_offsets[:, None]) * encoder_stride_t + (k_offsets[:, None]) * encoder_stride_h
        # ptr_img: image_ptr + pid_b*image_stride_b + (src_i[:, None])*(image_stride_i) + (k_offsets[:, None])*(image_stride_h)
        ptr_img = image_ptr + pid_b * image_stride_b + (src_i[:, None]) * image_stride_i + (k_offsets[:, None]) * image_stride_h

        # Select per token: if encoder_mask[token], use ptr_enc; else use ptr_img.
        # Note: We also guard with k_mask so that we only load valid K.
        # Triton supports combining masks; we build a boolean select matrix.
        select_mat = encoder_mask[:, None]  # [BLOCK_T, 1], broadcast along K
        # When selecting, ensure that for tokens >= T, ptr_img is valid (src_i in range). We rely on t_mask and h_mask for output store, and k_mask for input loads.
        # We combine select with k_mask: ptrs that exceed k_mask won't be used (masked loads return 0).
        ptrs = tl.where(select_mat, ptr_enc, ptr_img)  # [BLOCK_T, BLOCK_K]
        # Apply k_mask: tl.load returns 0 for masked elements. We include k_mask as part of the load.
        x_chunk = tl.load(ptrs, mask=valid_token[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_T, BLOCK_K], float32 accumulation

        # Load weight block [BLOCK_H, BLOCK_K]
        weight_ptrs = weight_ptr + h_offsets[:, None] * weight_stride_w + k_offsets[None, :] * weight_stride_k
        weight_block = tl.load(weight_ptrs, mask=h_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_H, BLOCK_K]

        # Compute partial output: [BLOCK_T, BLOCK_H] = sum over K of weight_block * x_chunk
        # Broadcast x_chunk [BLOCK_T, BLOCK_K] and weight_block.T [BLOCK_K, BLOCK_H] to match.
        # We need [BLOCK_T, BLOCK_K, BLOCK_H] then reduce along K. Triton does not require explicit reshape, we can do elementwise and tl.sum along axis=1.
        # Instead, use matrix multiply-like pattern: (x_chunk * weight_block.T) and sum along K.
        # Create expanded tensors with proper broadcasting:
        # x_chunk[:, :, None] and weight_block.T[None, :, :] won't work as-is; better to use tl.dot or manual outer-product:
        # We can compute partial by outer-product reduction: sum over K of weight_block[:, kk] * x_chunk[:, kk]
        # Implement via loop over kk in BLOCK_K:
        # However, Triton supports tl.dot(A, B) for 2D matrices. Here, we can transpose weight_block to [BLOCK_K, BLOCK_H] and multiply:
        # But we want [BLOCK_T, BLOCK_H] = x_chunk @ weight_block^T
        # Triton does not support direct tl.dot on 2D here, so we do manual reduction:
        # Build a [BLOCK_T, BLOCK_H] by iterating kk in range(BLOCK_K) with masks.

        # Since we have x_chunk [BLOCK_T, BLOCK_K] and weight_block [BLOCK_H, BLOCK_K],
        # We need x_chunk @ weight_block^T. Triton does not expose a built-in outer-product here,
        # so we compute partial as:
        # For kk in range(BLOCK_K):
        #   contrib = x_chunk[:, kk][:, None] * weight_block[:, kk][None, :]
        #   output += contrib
        # We need to guard with k_mask: only kk where k_mask[kk] is true contribute.

        # Implement this loop safely:
        for kk in range(0, BLOCK_K):
            # mask kk within k_mask
            valid_kk = (k_off + kk) < H
            # If not valid_kk, skip by using zero x_chunk[:, kk]
            x_col = x_chunk[:, kk]  # [BLOCK_T]
            w_row = weight_block[:, kk]  # [BLOCK_H]
            # Compute contrib [BLOCK_T, BLOCK_H] = x_col[:, None] * w_row[None, :]
            contrib = x_col[:, None] * w_row[None, :]
            # Accumulate into output
            output += contrib

    # Store the output tile to out[b, t_offsets, h_offsets]
    out_ptrs = out_ptr + pid_b * out_stride_b + t_offsets[:, None] * out_stride_t + h_offsets[None, :] * out_stride_h
    # Output mask: only valid tokens and valid hidden offsets
    out_mask = valid_token[:, None] & h_mask[None, :]
    # Cast to float32 for store; Triton will handle pointer type. We assume out_ptr is float32; if not, cast outside.
    tl.store(out_ptrs, output, mask=out_mask)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized linear projection of concatenated sequences.
    Returns processed of shape [B, T+I, H].
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]

    # Ensure contiguous layout for predictable strides
    image = hidden_states
    encoder = encoder_hidden_states
    weight = process_weight

    out = torch.empty((B, T + I, H), device=image.device, dtype=torch.float32)  # accumulate and output in float32

    # Strides in elements
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()  # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h = out.stride()

    # Choose tile sizes; tune for performance
    BLOCK_T = 16  # number of tokens per program
    BLOCK_H = 64  # number of hidden dims per program
    BLOCK_K = 64  # K-chunk size for accumulation

    grid = (B, triton.cdiv(T + I, BLOCK_T), triton.cdiv(H, BLOCK_H))

    concat_linear_kernel_tiled[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    return out


# Optional: keep a ModelNew as requested. This is just a wrapper for the Triton path.
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden