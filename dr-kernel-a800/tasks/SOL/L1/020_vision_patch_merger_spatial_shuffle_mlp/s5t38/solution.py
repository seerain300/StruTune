import math
import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate inputs with valid grid_thw that matches num_patches."""
    num_patches = axes_and_scalars["num_patches"]
    num_merged_patches = axes_and_scalars["num_merged_patches"]
    num_grids = axes_and_scalars["num_grids"]
    hidden_size = 1536
    hidden_size_expanded = 6144
    out_hidden_size = 3584
    merge_size = 2
    eps = 1e-6

    # Generate grid_thw such that total patches matches num_patches
    # Each grid contributes T * H * W patches
    patches_per_grid = num_patches // num_grids

    # Try to pick T, H, W from sqrt of patches_per_grid, enforcing divisibility by merge_size
    sqrt_patches = int(math.sqrt(patches_per_grid))
    # Round to nearest multiple of merge_size
    h = (sqrt_patches // merge_size) * merge_size
    if h == 0:
        h = merge_size
    w = (patches_per_grid // h // merge_size) * merge_size
    if w == 0:
        w = merge_size
    t = patches_per_grid // (h * w)
    if t == 0:
        t = 1

    # Adjust to match exactly
    actual_patches_per_grid = t * h * w

    # Create grid_thw tensor
    grid_thw = torch.zeros((num_grids, 3), dtype=torch.int64, device=device)
    remaining_patches = num_patches
    for i in range(num_grids):
        if i == num_grids - 1:
            patches_for_this = remaining_patches
        else:
            patches_for_this = actual_patches_per_grid

        sqrt_p = int(math.sqrt(patches_for_this))
        h_i = (sqrt_p // merge_size) * merge_size
        if h_i == 0:
            h_i = merge_size
        w_i = (patches_for_this // h_i // merge_size) * merge_size
        if w_i == 0:
            w_i = merge_size
        t_i = patches_for_this // (h_i * w_i)
        if t_i == 0:
            t_i = 1

        grid_thw[i, 0] = t_i
        grid_thw[i, 1] = h_i
        grid_thw[i, 2] = w_i
        remaining_patches -= t_i * h_i * w_i

    hidden = torch.randn(num_patches, hidden_size, dtype=torch.bfloat16, device=device)
    ln_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    ln_bias = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)
    fc1_weight = torch.randn(hidden_size_expanded, hidden_size_expanded, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size_expanded)
    fc1_bias = torch.randn(hidden_size_expanded, dtype=torch.bfloat16, device=device)
    fc2_weight = torch.randn(out_hidden_size, hidden_size_expanded, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size_expanded)
    fc2_bias = torch.randn(out_hidden_size, dtype=torch.bfloat16, device=device)

    return {
        "hidden": hidden,
        "grid_thw": grid_thw,
        "ln_weight": ln_weight,
        "ln_bias": ln_bias,
        "fc1_weight": fc1_weight,
        "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight,
        "fc2_bias": fc2_bias,
        "eps": eps,
    }


@torch.no_grad()
def run(
    hidden: torch.Tensor,
    grid_thw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    eps: float,
):
    """
    Vision patch merger with spatial shuffling and MLP.
    Steps:
      1) LayerNorm (pre-shuffle) per patch: normalize each vector of length 1536, apply ln_weight and ln_bias.
      2) Spatial shuffle to merge 2x2 patches, producing hidden_shuffled with shape (num_merged_patches, 6144).
         This involves per-grid reshapes/permutes; we implement it exactly as in the original.
      3) Two-layer MLP with GELU activation:
         - First linear: (num_merged_patches, 6144) @ (6144, 6144)^T + bias
         - GELU
         - Second linear: (num_merged_patches, 6144) @ (3584, 6144)^T + bias
    """
    merge_size = 2
    hidden_size = 1536
    hidden_size_expanded = 6144
    out_hidden_size = 3584

    # Step 1: LayerNorm (pre-shuffle, on hidden_size dimension)
    # Using torch.nn.functional.layer_norm to match PyTorch behavior and ensure stability.
    hidden_fp32 = hidden.to(torch.float32)
    mean = hidden_fp32.mean(dim=-1, keepdim=True)
    var = hidden_fp32.var(dim=-1, unbiased=False, keepdim=True)  # population variance
    rstd = torch.rsqrt(var + eps)
    hidden_norm = (hidden_fp32 - mean) * rstd  # (num_patches, hidden_size), float32
    # Apply learnable affine in float32, then cast back to bfloat16
    hidden_norm = hidden_norm * ln_weight.to(torch.float32) + ln_bias.to(torch.float32)
    hidden_norm = hidden_norm.to(torch.bfloat16)  # (num_patches, hidden_size)

    # Step 2: Spatial shuffle to merge patches into shape (num_merged_patches, 6144)
    offset = 0
    shuffled_patches = []
    for i in range(grid_thw.shape[0]):
        t = int(grid_thw[i, 0].item())
        h = int(grid_thw[i, 1].item())
        w = int(grid_thw[i, 2].item())
        num_patches_this = t * h * w
        patches = hidden_norm[offset:offset + num_patches_this]

        # Reshape to (t, h, w, hidden_size)
        patches = patches.view(t, h, w, hidden_size)

        # Merge 2x2 with merge_size=2: reshape to (t, h//2, 2, w//2, 2, hidden_size)
        h_merged = h // merge_size
        w_merged = w // merge_size
        patches = patches.view(t, h_merged, merge_size, w_merged, merge_size, hidden_size)

        # Permute to (t, h_merged, w_merged, merge_size, merge_size, hidden_size)
        patches = patches.permute(0, 1, 3, 2, 4, 5)

        # Flatten each 2x2 group into 6144 features
        patches = patches.reshape(t * h_merged * w_merged, hidden_size_expanded)  # (num_patches_per_grid, 6144)

        shuffled_patches.append(patches)
        offset += num_patches_this

    hidden_shuffled = torch.cat(shuffled_patches, dim=0)  # (num_merged_patches, 6144), bfloat16

    # Step 3: Two-layer MLP with GELU
    # First linear: (num_merged_patches, 6144) @ (6144, 6144)^T + fc1_bias
    # Note: PyTorch's F.linear expects (M, K) @ (K, N). We provide fc1_weight as (6144, 6144) which is (K, N).
    # So F.linear(hidden_shuffled @ fc1_weight.t() + fc1_bias)
    fc1_weight_t = fc1_weight.transpose(0, 1).to(torch.float32)  # (6144, 6144)
    fc1_bias_f32 = fc1_bias.to(torch.float32)
    hidden_fc1 = torch.nn.functional.linear(hidden_shuffled.to(torch.float32), fc1_weight_t, fc1_bias_f32)  # (num_merged, 6144), float32

    # GELU activation
    hidden_gelu = torch.nn.functional.gelu(hidden_fc1)  # (num_merged, 6144), float32

    # Second linear: (num_merged, 6144) @ (3584, 6144)^T + fc2_bias
    fc2_weight_t = fc2_weight.transpose(0, 1).to(torch.float32)  # (6144, 3584)
    fc2_bias_f32 = fc2_bias.to(torch.float32)
    output = torch.nn.functional.linear(hidden_gelu, fc2_weight_t, fc2_bias_f32)  # (num_merged, 3584), float32

    # Return in bfloat16 to match original model's dtype
    return output.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We expect the same inputs as the original 'run' signature: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps
        if len(args) != 9:
            raise RuntimeError("ModelNew.forward expects 9 arguments: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps")
        hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps = args
        return run(hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps)


def run(*args):
    return ModelNew()(*args)
