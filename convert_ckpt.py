import torch
import torch.nn as nn

# 1. Define paths
# Use the ORIGINAL standard checkpoint for this conversion
load_path = "./runs_2/convnext_tiny_finetune/convnext_tiny_1k_224_ema.pth"
save_path = "./runs_2/convnext_tiny_finetune/convnext_tiny_mhdw_converted.pth"

print(f"Loading original checkpoint from: {load_path}")
checkpoint = torch.load(load_path, map_location='cpu')
state_dict = checkpoint['model']

new_state_dict = {}

# Track dimensions for each stage to initialize fusion layers
stage_dims = {
    0: 96,   # Stage 0: 96 channels
    1: 192,  # Stage 1: 192 channels
    2: 384,  # Stage 2: 384 channels
    3: 768   # Stage 3: 768 channels
}

# Count blocks per stage for ConvNeXt-Tiny [3, 3, 9, 3]
blocks_per_stage = {0: 3, 1: 3, 2: 9, 3: 3}

print("Converting checkpoint to Multi-Head Depthwise (MHDW) with 1:2:1 ratio...")

for k, v in state_dict.items():
    # We only modify the depthwise convolutions inside the blocks
    # Standard key format: "stages.0.0.dwconv.weight" or similar
    if "stages" in k and "dwconv" in k and "weight" in k:
        
        # 1. Get dimensions and calculate 1:2:1 split
        total_dim = v.shape[0]
        chunk_dim = total_dim // 4  # Base unit
        local_dim = chunk_dim       # 1/4 for local (25%)
        standard_dim = chunk_dim * 2  # 2/4 for standard (50%)
        global_dim = total_dim - local_dim - standard_dim  # Remaining 1/4 for global (25%)

        print(f"Splitting {k} (Dim: {total_dim}) with 1:2:1 ratio -> [Local:{local_dim}, Standard:{standard_dim}, Global:{global_dim}]")

        # 2. Split the weights into 3 chunks with 1:2:1 ratio
        # v shape is [Groups, 1, 7, 7] for depthwise conv
        w_local_orig, w_standard, w_global = torch.split(v, [local_dim, standard_dim, global_dim], dim=0)

        # 3. Process Local Path: Center crop 7x7 -> 3x3
        # Center indices of 7x7 are rows/cols 2, 3, 4
        w_local_cropped = w_local_orig[:, :, 2:5, 2:5]
        
        # 4. Create new keys for the three paths
        base_key = k.replace(".weight", "")
        
        new_state_dict[f"{base_key}.dwconv_local.weight"] = w_local_cropped
        new_state_dict[f"{base_key}.dwconv_standard.weight"] = w_standard
        new_state_dict[f"{base_key}.dwconv_global.weight"] = w_global
        
    # Handle Biases (similar 1:2:1 split, no cropping needed)
    elif "stages" in k and "dwconv" in k and "bias" in k:
        total_dim = v.shape[0]
        chunk_dim = total_dim // 4
        local_dim = chunk_dim
        standard_dim = chunk_dim * 2
        global_dim = total_dim - local_dim - standard_dim
        
        b_local, b_standard, b_global = torch.split(v, [local_dim, standard_dim, global_dim], dim=0)
        
        base_key = k.replace(".bias", "")
        new_state_dict[f"{base_key}.dwconv_local.bias"] = b_local
        new_state_dict[f"{base_key}.dwconv_standard.bias"] = b_standard
        new_state_dict[f"{base_key}.dwconv_global.bias"] = b_global

    else:
        # Copy everything else (Downsample layers, Norms, Pointwise convs) as is
        new_state_dict[k] = v

# Initialize fusion_norm and fusion layers for each block
print("\nInitializing fusion_norm and fusion layers...")
for stage_idx, num_blocks in blocks_per_stage.items():
    dim = stage_dims[stage_idx]
    for block_idx in range(num_blocks):
        base_key = f"stages.{stage_idx}.{block_idx}.dwconv"
        
        # Initialize fusion_norm (LayerNorm parameters)
        new_state_dict[f"{base_key}.fusion_norm.weight"] = torch.ones(dim)
        new_state_dict[f"{base_key}.fusion_norm.bias"] = torch.zeros(dim)
        
        # Initialize fusion (1x1 Conv2d parameters)
        # Use truncated normal initialization like the rest of ConvNeXt
        fusion_weight = torch.empty(dim, dim, 1, 1)
        nn.init.trunc_normal_(fusion_weight, std=0.02)
        new_state_dict[f"{base_key}.fusion.weight"] = fusion_weight
        new_state_dict[f"{base_key}.fusion.bias"] = torch.zeros(dim)
        
        print(f"  Initialized fusion layers for {base_key} (dim={dim})")

# Save
print(f"\nSaving converted checkpoint to {save_path}...")
torch.save({'model': new_state_dict}, save_path)
print("Success! Checkpoint ready for training.")
print("\nSummary:")
print("- Depthwise convs split with 1:2:1 ratio (Local:Standard:Global = 25%:50%:25%)")
print("- Fusion layers initialized with standard ConvNeXt initialization")
print("- All other layers copied from original checkpoint")
