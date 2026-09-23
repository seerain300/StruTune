import torch
import triton
import triton.language as tl

# Triton kernel: concatenate along sequence dimension without torch.cat.
# Input:
#   - encoder: [B, T, H]
#   - hidden: [B, I, H]
# Output:
#   - out_cat: [B, T + I, H]
# The kernel launches a 2D grid over (batch, tiles of L), and writes either encoder or hidden
# to out_cat depending on the sequence index. It uses masked loads/stores for safety.
@triton.jit
def concat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_b_e: tl.int32, stride_t_e: tl.int32, stride_h_e: tl.int32,
    stride_b_h: tl.int32, stride_i_h: tl.int32, stride_h_h: tl.int32,
    stride_b_o: tl.int32, stride_l_o: tl.int32, stride_h_o: tl.int32,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch id
    pid_l = tl.program_id(1)  # tile id along sequence length

    l_start = pid_l * BLOCK_L
    offs = l_start + tl.arange(0, BLOCK_L)
    L_total = T + I

    mask = offs < L_total
    is_encoder = offs < T
    is_hidden = offs >= T  # equivalent to (offs < L_total) & (offs >= T) but mask handles it

    # Loop over hidden dimension H and perform masked stores
    for h in range(0, H):
        # Compute pointers
        encoder_ptrs = encoder_ptr + pid_b * stride_b_e + offs * stride_t_e + h * stride_h_e
        hidden_ptrs = hidden_ptr + pid_b * stride_b_h + (offs - T) * stride_i_h + h * stride_h_h
        out_ptrs = out_ptr + pid_b * stride_b_o + offs * stride_l_o + h * stride_h_o

        # Load encoder values where applicable
        val_e = tl.load(encoder_ptrs, mask=mask & is_encoder, other=0.0)
        # Load hidden values where applicable
        val_h = tl.load(hidden_ptrs, mask=mask & is_hidden, other=0.0)

        # Store: for positions < T, out_cat[:, offs, h] = encoder[:, offs, h]
        #        for positions >= T, out_cat[:, offs, h] = hidden[:, offs - T, h]
        tl.store(out_ptrs, val_e, mask=mask & is_encoder)
        tl.store(out_ptrs, val_h, mask=mask & is_hidden)

class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized concatenation; matmul done in PyTorch for correctness and speed.
        Returns (processed_encoder, processed_hidden).
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, dim, H]"
        assert process_weight.dim() == 2, "process_weight must be [H, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match between encoder and image streams"

        # Ensure contiguous tensors
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # Allocate output concatenated tensor [B, T + I, H]
        L = T + I
        out_cat = torch.empty((B, L, H), dtype=encoder.dtype, device=encoder.device)

        # Launch Triton kernel for concatenation
        # 2D grid: (B, ceil_div(L, BLOCK_L))
        BLOCK_L = 128
        grid = (B, triton.cdiv(L, BLOCK_L))
        concat_seq_kernel[grid](
            encoder, hidden, out_cat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            BLOCK_L=BLOCK_L,
            num_warps=4,
            num_stages=2,
        )

        # Apply linear projection using torch in float32 for numerical stability and speed
        # processed = out_cat @ process_weight.T
        # Cast to float32 for matmul, then cast back to original dtype to match original behavior.
        out_cat_f32 = out_cat.float()
        weight_f32 = weight.float()
        processed_f32 = torch.matmul(out_cat_f32, weight_f32.t())
        processed = processed_f32.to(out_cat.dtype)

        # Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
