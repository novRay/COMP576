#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from typing import Tuple, Dict

# Import from train.py
from train import Config, get_model, load_and_split_hf_dataset, get_data_loaders_from_hf_splits, set_seed

# ==========================================
# Grad-CAM
# ==========================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        try:
            self.target_layer.register_full_backward_hook(backward_hook)
        except Exception:
            try:
                self.target_layer.register_backward_hook(backward_hook)
            except Exception:
                pass

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        self.model.eval()
        x = x.requires_grad_(True)
        outputs = self.model(x)

        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1)[0].item())
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx.item())

        target = outputs[0, class_idx]
        self.model.zero_grad()
        target.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target layer for hook registration.")

        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to input size
        target_h, target_w = x.shape[2], x.shape[3]
        cam_resized = cv2_resize_cam(cam, (target_h, target_w))
        return cam_resized

def cv2_resize_cam(cam: np.ndarray, target_size: Tuple[int, int]):
    """
    cam: 2D float array in [0,1]
    target_size: (H, W)
    """
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_uint8)  # mode 'L'
    # PIL takes (width, height)
    cam_img = cam_img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0
    return cam_arr

def generate_gradcam(model: nn.Module, dataloader, device: torch.device, model_name: str, config: Config, classes: list, samples_per_class: int = 2):
    target_layer = None
    if "resnet" in model_name:
        target_layer = model.layer4[-1]
    elif "efficientnet" in model_name:
        candidate = model.features[-1]
        target_layer = candidate
    else:
        print(f"Skipping Grad-CAM for {model_name}: target layer unknown")
        return

    grad_cam = GradCAM(model, target_layer)

    collected_samples = defaultdict(list)
    classes_needed = set(range(len(classes)))
    
    print(f"Collecting {samples_per_class} samples per class for Grad-CAM...")
    
    for inputs, labels in dataloader:
        if not classes_needed:
            break
            
        for i in range(len(inputs)):
            label_idx = int(labels[i].item())
            if label_idx in classes_needed:
                if len(collected_samples[label_idx]) < samples_per_class:
                    collected_samples[label_idx].append(inputs[i])
                    
                if len(collected_samples[label_idx]) >= samples_per_class:
                    if label_idx in classes_needed:
                        classes_needed.remove(label_idx)
            
            if not classes_needed:
                break

    if not collected_samples:
        print("No samples collected for Grad-CAM.")
        return

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for label_idx, imgs in collected_samples.items():
        class_name = classes[label_idx]
        for i, img_tensor in enumerate(imgs):
            img_input = img_tensor.unsqueeze(0).to(device)
            
            heatmap = grad_cam(img_input, class_idx=label_idx)

            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            plt.title(f"Original: {class_name}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.title(f"Grad-CAM: {model_name}")
            plt.axis('off')

            plt.tight_layout()
            safe_class_name = "".join([c if c.isalnum() else "_" for c in class_name])
            plt.savefig(os.path.join(config.GRADCAM_DIR, f'{model_name}_{safe_class_name}_{i}.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate GradCAM Heatmaps")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., resnet50)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. If None, uses default path in results/")
    parser.add_argument("--samples", type=int, default=2, help="Samples per class")
    args = parser.parse_args()

    config = Config()
    set_seed(config.SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Using device: {device}")

    hf_splits = load_and_split_hf_dataset(config, seed=config.SEED)
    classes = hf_splits['train'].features['style'].names
    
    dataloaders, _ = get_data_loaders_from_hf_splits(hf_splits, config, batch_size=16)
    test_loader = dataloaders['test']

    print(f"Loading model: {args.model_name}")
    model = get_model(args.model_name, len(classes))
    
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(config.MODEL_DIR, f"best_{args.model_name}.pth")
        
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    print(f"Loading weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    generate_gradcam(model, test_loader, device, args.model_name, config, classes, samples_per_class=args.samples)

if __name__ == "__main__":
    main()
