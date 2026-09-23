import math
import torch
import triton
import triton.language as tl

# Triton kernel for LayerNorm over the last dimension.
# Assumes input is (num_rows, hidden_size), applies per-row mean/var,
# then scales and shifts by ln_weight (shape: [hidden_size]) and ln_bias (shape: [hidden_size]).
# Outputs are stored in bfloat16. Computation is done in float32 for numerical stability.
@triton.jit
def _layernorm_rows_kernel(
    X_ptr,          # *const float16 or bfloat16: input
    W_ptr,          # *const bfloat16: ln_weight (length == hidden_size)
    B_ptr,          # *const bfloat16: ln_bias (length == hidden_size)
    Y_ptr,          # *float16 or bfloat16: output
    N_ROWS,         # int: number of rows
    HIDDEN_SIZE,    # int: number of columns (1536)
    EPS,            # float32
    BLOCK_SIZE: tl.constexpr,  # compile-time block size (should be == HIDDEN_SIZE here)
):
    row_id = tl.program_id(0)
    if row_id >= N_ROWS:
        return

    # Pointers to the start of this row
    row_in_ptr = X_ptr + row_id * HIDDEN_SIZE
    row_out_ptr = Y_ptr + row_id * HIDDEN_SIZE

    # Load row into fp32 for computation
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE
    x = tl.load(row_in_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / HIDDEN_SIZE

    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / HIDDEN_SIZE

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + EPS)
    norm = diff * inv_std

    # Load per-feature weights and bias
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = norm * w + b  # fp32

    # Store as bfloat16
    y_bf16 = y.to(tl.bfloat16)
    tl.store(row_out_ptr + cols, y_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor):
        """
        Triton-optimized forward:
        - LayerNorm is computed with a Triton kernel (per-row LN).
        - Spatial shuffle remains in PyTorch (view/permute/reshape).
        - Two linear layers and GELU are in PyTorch.
        """
        # hidden: [num_patches, hidden_size] in bfloat16
        # Ensure contiguous for Triton
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()

        num_patches, hidden_size = hidden.shape
        assert hidden_size == 1536, "Expected hidden_size=1536 as per provided code."

        # Allocate output for LN in bfloat16
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)

        # Launch Triton LN kernel: one program per row
        grid = (num_patches,)
        _layernorm_rows_kernel[grid](
            hidden, ln_weight, ln_bias, hidden_norm,
            num_patches, hidden_size, self.eps,
            BLOCK_SIZE=hidden_size  # hidden_size is a compile-time constant for the kernel
        )

        # Spatial shuffle: implemented with PyTorch reshape operations (data movement, not compute-heavy)
        # The original code performs:
        # - patches = hidden_norm.view(t, h_merged, merge_size, w_merged, merge_size, C)
        # - patches = patches.permute(0, 1, 3, 2, 4, 5)
        # - patches = patches.reshape(t * h_merged * w_merged, merge_size^2 * C)
        # Since hidden_norm is [num_patches, 1536], we reuse the same logic with the provided grid_thw.
        # Note: This logic is deterministic and based on grid_thw. We'll follow it exactly.

        # Reconstruct t, h, w per grid using grid_thw. Here grid_thw shape is (num_grids, 3).
        # We assume the same mapping as original code (merge_size=2).
        # For each grid, compute t,h,w, then reshape accordingly.
        # We'll iterate over grids and append the reshaped patches to a list.

        offset = 0
        shuffled_patches = []
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            patches = hidden_norm[offset:offset + num_patches_this]

            h_merged = h // 2  # merge_size=2
            w_merged = w // 2

            # Reshape to (T, H/merge_size, merge_size, W/merge_size, merge_size, C)
            patches = patches.view(t, h_merged, 2, w_merged, 2, hidden_size)
            # Permute to (T, H/merge_size, W/merge_size, merge_size, merge_size, C)
            patches = patches.permute(0, 1, 3, 2, 4, 5)
            # Flatten spatial merge groups: (T * H/merge_size * W/merge_size, merge_size^2 * C)
            patches = patches.reshape(t * h_merged * w_merged, hidden_size * (2 * 2))
            # Now patches shape is (num_patches_this, hidden_size_expanded)
            shuffled_patches.append(patches)
            offset += num_patches_this

        hidden_shuffled = torch.cat(shuffled_patches, dim=0)  # shape: (num_merged_patches, hidden_size_expanded)

        # Two-layer MLP with GELU in between. Keep these in PyTorch to ensure correctness and performance.
        hidden_fc1 = torch.nn.functional.linear(hidden_shuffled, fc1_weight, fc1_bias)
        hidden_gelu = torch.nn.functional.gelu(hidden_fc1)
        output = torch.nn.functional.linear(hidden_gelu, fc2_weight, fc2_bias)

        return output


def run(*args):
    return ModelNew()(*args)
