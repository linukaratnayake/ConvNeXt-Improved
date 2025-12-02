import torch
import os

# 1. Define paths
load_path = "./runs/convnext_tiny_finetune/convnext_tiny_1k_224_ema.pth"
save_path = "./runs/convnext_tiny_finetune/convnext_tiny_patchmerging_converted.pth"

print(f"Loading checkpoint from: {load_path}")
checkpoint = torch.load(load_path, map_location='cpu')
state_dict = checkpoint['model']

new_state_dict = {}

print("Converting checkpoint...")

for k, v in state_dict.items():
    # Only process downsample layers
    if "downsample_layers" in k:
        # Extract the stage index (e.g., downsample_layers.0.0.weight -> 0)
        parts = k.split('.')
        # The format is downsample_layers.{index}.{sublayer}.{param}
        stage_index = int(parts[1])

        # --- CASE 1: The Stem (Index 0) ---
        # The Stem is NOT a PatchMerging layer. It remains standard Conv -> LN.
        # We copy it exactly as is.
        if stage_index == 0:
            new_state_dict[k] = v
            
        # --- CASE 2: The Downsample Layers (Indices 1, 2, 3) ---
        # These are converted from [LN -> Conv] to [PatchMerging(LN -> Linear)]
        else:
            # 1. Convert LayerNorm (Old layer 0 -> New .norm)
            if ".0.weight" in k:
                new_k = k.replace(f".{stage_index}.0.weight", f".{stage_index}.norm.weight")
                new_state_dict[new_k] = v
                print(f"Mapped LN (Stage {stage_index}): {k} -> {new_k}")
                
            elif ".0.bias" in k:
                new_k = k.replace(f".{stage_index}.0.bias", f".{stage_index}.norm.bias")
                new_state_dict[new_k] = v

            # 2. Convert Conv2d to Linear (Old layer 1 -> New .reduction)
            elif ".1.weight" in k:
                # v is standard Conv2d weights: [Out, In, 2, 2]
                out_ch, in_ch, h, w = v.shape
                
                # Reshape to Linear weights: [Out, In*4]
                # We flatten the 2x2 spatial kernel into the input channel dimension
                new_v = v.permute(0, 1, 3, 2).reshape(out_ch, in_ch * 4)
                
                new_k = k.replace(f".{stage_index}.1.weight", f".{stage_index}.reduction.weight")
                new_state_dict[new_k] = new_v
                print(f"Converted Conv->Linear (Stage {stage_index}): {k} -> {new_k} | Shape: {new_v.shape}")
                
            elif ".1.bias" in k:
                new_k = k.replace(f".{stage_index}.1.bias", f".{stage_index}.reduction.bias")
                new_state_dict[new_k] = v

    # Copy all other layers (stages, head, etc.) as is
    else:
        new_state_dict[k] = v

# Save the new checkpoint
print(f"Saving converted checkpoint to {save_path}...")
torch.save({'model': new_state_dict}, save_path)
print("Success!")
