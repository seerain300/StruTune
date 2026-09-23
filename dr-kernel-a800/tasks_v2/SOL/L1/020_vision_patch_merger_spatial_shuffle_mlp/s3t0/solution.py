import torch
import math
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr,            # *ptr to input hidden (num_patches, hidden_size), bfloat16
    y_ptr,            # *ptr to output normalized + affine (num_patches, hidden_size), bfloat16
    ln_weight_ptr,    # *ptr to ln_weight (hidden_size), bfloat16
    ln_bias_ptr,      # *ptr to ln_bias (hidden_size), bfloat16
    N,                # num_patches (rows)
    C,                # hidden_size (columns)
    eps,              # epsilon for stability (float32)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton LayerNorm per row:
    For each row i in [0, N):
      - Compute sum and sum of squares across C columns in chunks of BLOCK_SIZE (float32).
      - Compute mean and variance, inv_std.
      - Second pass: normalize and apply affine: y = ((x - mean) * inv_std) * ln_weight + ln_bias.
      - Store y as bfloat16.
    """
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C

    # Pass 1: compute mean and variance
    sum_val = 0.0
    sum_sq = 0.0
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE

    mean = sum_val / C
    var = sum_sq / C
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, then store
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor):
        """
        hidden: [num_patches, hidden_size], bfloat16, on CUDA
        grid_thw: [num_grids, 3], int64, per grid (T, H, W), H and W divisible by merge_size (2)
        ln_weight, ln_bias: [hidden_size], bfloat16, on CUDA
        fc1_weight, fc1_bias, fc2_weight, fc2_bias: bfloat16, on CUDA
        """
        # Ensure CUDA
        assert hidden.is_cuda, "hidden must be on CUDA device"
        assert ln_weight.is_cuda and ln_bias.is_cuda, "LayerNorm params must be on CUDA"
        assert fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, "MLP params must be on CUDA"

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]

        # Output tensor for LayerNorm result
        out_hidden_norm = torch.empty_like(hidden)

        # Launch Triton LayerNorm kernel: one program per row
        BLOCK_SIZE = 1024  # works for hidden_size=1536; loops handle larger C as well
        grid = (num_patches,)
        _layer_norm_kernel[grid](
            hidden, out_hidden_norm,
            ln_weight, ln_bias,
            num_patches, hidden_size,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # reasonable default
            num_stages=2, # pipeline stages
        )

        # Step 2: Spatial shuffle using PyTorch view/permute (metadata ops).
        # Build hidden_shuffled: [num_merged_patches, hidden_size_expanded] where hidden_size_expanded = hidden_size * 4
        # We reconstruct per-grid contributions and concatenate.
        offset = 0
        shuffled_patches = []
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            # Merge size is 2 -> hidden_size_expanded = hidden_size * 2 * 2 = 6144
            h_merged = h // 2
            w_merged = w // 2
            num_patches_grid = t * h * w

            patches = out_hidden_norm[offset:offset + num_patches_grid]  # [t*h*w, hidden_size], bfloat16
            # Reshape to (T, H, W, hidden_size), permute to (T, H//2, W//2, 2, 2, hidden_size), then flatten
            patches = patches.view(t, h, w, patches.shape[1])  # (T, H, W, hidden_size)
            # permute (T, H//2, W//2, 2, 2, hidden_size)
            # Positions for the 2x2 merge are (H//2, W//2, h_idx in [H//2, H-1], w_idx in [W//2, W-1])
            # After permute, the two patches correspond to the last two dims (2, 2), and the channel last dim is hidden_size.
            # Flattening merges 4 channels into one dimension of size 4*hidden_size.
            patches = patches.permute(0, h_merged, w_merged, h - h_merged, w - w_merged, 4)
            # Note: h - h_merged and w - w_merged correspond to the two original 2x2 positions within the 2x2 merge window.
            patches = patches.reshape(t * h_merged * w_merged, 4 * patches.shape[4])
            shuffled_patches.append(patches)
            offset += num_patches_grid

        hidden_shuffled = torch.cat(shuffled_patches, dim=0)  # [num_merged_patches, hidden_size_expanded]

        # Step 3: Two-layer MLP with GELU (PyTorch functional). Keep matmul in PyTorch (functional.linear).
        hidden_fc1 = torch.nn.functional.linear(hidden_shuffled, fc1_weight, fc1_bias)
        hidden_gelu = torch.nn.functional.gelu(hidden_fc1)
        output = torch.nn.functional.linear(hidden_gelu, fc2_weight, fc2_bias)

        return output


def run(*args):
    return ModelNew()(*args)
