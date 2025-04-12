import torch
import re
import os

# Define paths
pretrained_path = "../../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"  # Pretrained model path
fine_tuned_path = "../../Depth-Anything-V2/checkpoints/latest.pth"                # Fine-tuned model path
merged_model_path = "../../Depth-Anything-V2/checkpoints/merged_fine_tuned_depth_anything_v2.pth"  # Merged model save path
merged_metadata_path = "../../Depth-Anything-V2/checkpoints/merged_fine_tuned_metadata.pth"          # Merged metadata save path

# Load the pretrained checkpoint
print(f"Loading pretrained model from: {pretrained_path} ...")
pretrained_checkpoint = torch.load(pretrained_path, map_location="cpu")

# Find the correct section in the pretrained checkpoint that contains the key "pretrained.cls_token"
all_sections = list(pretrained_checkpoint.keys())
found_section = None
for section in all_sections:
    if isinstance(pretrained_checkpoint[section], dict) and "pretrained.cls_token" in pretrained_checkpoint[section]:
        found_section = section
        break

if found_section:
    print(f"✅ Found 'pretrained.cls_token' under section: '{found_section}'")
    pretrained_state_dict = pretrained_checkpoint[found_section]
else:
    print("⚠️ 'pretrained.cls_token' not found in any section. Checking root keys...")
    if "pretrained.cls_token" in pretrained_checkpoint:
        found_section = None  # Use root-level keys.
        pretrained_state_dict = pretrained_checkpoint
        print("✅ Found 'pretrained.cls_token' at root level.")
    else:
        print("❌ 'pretrained.cls_token' not found in the pretrained checkpoint.")
        exit(1)

# Load the fine-tuned checkpoint
print(f"Loading fine-tuned model from: {fine_tuned_path} ...")
fine_tuned_checkpoint = torch.load(fine_tuned_path, map_location="cpu")

# Extract the fine-tuned state dictionary; it may be inside the "model" key.
fine_tuned_state_dict = fine_tuned_checkpoint.get("model", fine_tuned_checkpoint)

# Remove the "module." prefix from fine-tuned state dictionary keys (if present).
fine_tuned_state_dict = {k.replace("module.", ""): v for k, v in fine_tuned_state_dict.items()}

# Filter the fine-tuned state dictionary: keep only keys that exist in the pretrained state dict.
cleaned_state_dict = {k: v for k, v in fine_tuned_state_dict.items() if k in pretrained_state_dict}

# Merge fine-tuned weights into the pretrained state dictionary.
pretrained_state_dict.update(cleaned_state_dict)

# Save the merged model separately.
print(f"Saving merged model to: {merged_model_path} ...")
torch.save(pretrained_state_dict, merged_model_path)

# Preserve training metadata in a separate file.
metadata = {
    "epoch": fine_tuned_checkpoint.get("epoch", 0),
    "previous_best": fine_tuned_checkpoint.get("previous_best", None)
}
torch.save(metadata, merged_metadata_path)

print("✅ Merged model and metadata saved successfully!")