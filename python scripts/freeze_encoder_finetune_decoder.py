import tkinter as tk
import torch
import torch.nn as nn
import argparse
import logging
import os
import pprint
import random
import warnings
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tkinter import filedialog
from depth_anything_v2.dpt import DepthAnythingV2


# Hide the main Tkinter window
root = tk.Tk()
root.withdraw()

print("Select the existing .pth file to load the model weights...")
pth_path = filedialog.askopenfilename(
    title="Select Existing .pth File",
    filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
)
if not pth_path:
    print("No file selected. Exiting.")

# Prompt user to choose where to save the new .pth file
print("Select where to save the new .pth file with frozen encoder...")
new_pth_path = filedialog.asksaveasfilename(
    title="Save Frozen Model .pth File",
    defaultextension=".pth",
    filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
)
if not new_pth_path:
    print("No save file selected. Exiting.")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    'vitln': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
model = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 20})

model.load_state_dict(torch.load(pth_path, map_location='cpu'), strict=False)

# ----------------------------------------------------
# Freeze the encoder (pretrained) parameters
# ----------------------------------------------------
for param in model.pretrained.parameters():
    param.requires_grad = False

# The decoder (model.depth_head) remains trainable
for param in model.depth_head.parameters():
    param.requires_grad = True

print("Encoder frozen. Decoder remains trainable.")

# Save the model's state dict
# Note: Setting requires_grad=False is recognized at runtime.
# We'll just re-save the entire state_dict.
frozen_state_dict = model.state_dict()
torch.save(frozen_state_dict, new_pth_path)
print(f"New .pth file saved: {new_pth_path}")