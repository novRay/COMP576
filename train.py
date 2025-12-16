#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import random
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except Exception:
    HAS_FVCORE = False
    print("fvcore not found. FLOPs calculation will be skipped or estimated.")

from typing import Dict, Any, Tuple

# ==========================================
# 1. Configuration
# ==========================================

class Config:
    DATA_DIR = "dataset"
    OUTPUT_DIR = "results"
    PLOT_DIR = "plots"
    MODEL_DIR = "checkpoints"
    GRADCAM_DIR = "gradcam_outputs"
    
    IMG_SIZE = 224
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 1e-4
    SEED = 42
    NUM_WORKERS = 4
    
    MODELS_TO_TRAIN = ["resnet50", "efficientnet_b0", "efficientnet_b4"]
    
    def __init__(self, args=None):
        self.OUTPUT_DIR = os.environ.get("AIP_MODEL_DIR", args.output_dir if args and args.output_dir else "results")
        self.MODEL_DIR = os.environ.get("AIP_MODEL_DIR", self.OUTPUT_DIR)
        self.PLOT_DIR = os.path.join(self.OUTPUT_DIR, "plots")
        self.GRADCAM_DIR = os.path.join(self.OUTPUT_DIR, "gradcam_outputs")
        
        if args:
            if args.data_dir:
                self.DATA_DIR = args.data_dir
            if args.epochs:
                self.EPOCHS = args.epochs
            if args.batch_size:
                self.BATCH_SIZE = args.batch_size
            if args.model:
                self.MODELS_TO_TRAIN = args.model.split(',')
            if args.num_workers is not None:
                self.NUM_WORKERS = args.num_workers

        for d in [self.OUTPUT_DIR, self.PLOT_DIR, self.MODEL_DIR, self.GRADCAM_DIR]:
            os.makedirs(d, exist_ok=True)

def set_seed(seed=42, reproducible=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
from datasets import load_dataset, concatenate_datasets

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None, label_key='style'):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_key = label_key
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        label = int(item[self.label_key])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def build_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

def get_data_loaders_from_hf_splits(hf_splits: Dict[str, Any], config: Config, batch_size: int = None, img_size: int = None):
    """
    hf_splits: dict with keys 'train','val','test' containing datasets.Dataset objects
    Returns dataloaders and classes list
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    if img_size is None:
        img_size = config.IMG_SIZE

    train_transform, val_test_transform = build_transforms(img_size)

    train_ds = HFDatasetWrapper(hf_splits['train'], transform=train_transform)
    val_ds = HFDatasetWrapper(hf_splits['val'], transform=val_test_transform)
    test_ds = HFDatasetWrapper(hf_splits['test'], transform=val_test_transform)

    classes = hf_splits['train'].features['style'].names

    dl_kwargs = {
        "batch_size": batch_size,
        "pin_memory": True if torch.cuda.is_available() else False,
    }
    if config.NUM_WORKERS and config.NUM_WORKERS > 0:
        dl_kwargs.update({
            "num_workers": config.NUM_WORKERS,
            "persistent_workers": True,
            "prefetch_factor": 2
        })
    dataloaders = {
        'train': DataLoader(train_ds, shuffle=True, **dl_kwargs),
        'val': DataLoader(val_ds, shuffle=False, **dl_kwargs),
        'test': DataLoader(test_ds, shuffle=False, **dl_kwargs)
    }
    return dataloaders, classes

def load_and_split_hf_dataset(config: Config, seed=42):
    print("Loading HF dataset 'yonigozlan/wikiart-tiny' ... (this may take time on first run)")
    ds = load_dataset("yonigozlan/wikiart-tiny")
    if 'train' in ds and len(ds) == 1:
        full_ds = ds['train']
    else:
        full_ds = concatenate_datasets([ds[k] for k in ds.keys()])

    train_test = full_ds.train_test_split(test_size=0.2, seed=seed)
    train_ds_hf = train_test['train']
    test_val = train_test['test'].train_test_split(test_size=0.5, seed=seed)
    val_ds_hf = test_val['train']
    test_ds_hf = test_val['test']

    print(f"HF split sizes: Train={len(train_ds_hf)}, Val={len(val_ds_hf)}, Test={len(test_ds_hf)}")
    return {'train': train_ds_hf, 'val': val_ds_hf, 'test': test_ds_hf}

# ==========================================
# 3. Model Architectures & Helper
# ==========================================

def get_model(model_name: str, num_classes: int, freeze_resnet_until_layer: str = "layer3"):
    """
    Return model (unfrozen or partially frozen).
    freeze_resnet_until_layer controls which ResNet layers to keep trainable (e.g. "layer3" means freeze layers before layer3)
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for name, param in model.named_parameters():
            if freeze_resnet_until_layer not in name and not name.startswith("layer3") and not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name.startswith("efficientnet"):
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        elif model_name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
            model = models.efficientnet_b4(weights=weights)
        else:
             raise ValueError(f"Unknown efficientnet model: {model_name}")

        for param in model.parameters():
            param.requires_grad = True
            
        if hasattr(model, "classifier"):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            # fallback
            num_ftrs = model.get_classifier().in_features if hasattr(model, "get_classifier") else None
            if num_ftrs:
                model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

# ==========================================
# 4. Training Loop
# ==========================================

def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], criterion, optimizer, scheduler, num_epochs: int, device: torch.device, model_name: str, config: Config):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience = 5
    counter = 0
    best_loss = float('inf')

    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\nStarting training for {model_name} (AMP={'on' if use_amp else 'off'}) ...")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if use_amp:
                        autocast_ctx = torch.cuda.amp.autocast
                        with autocast_ctx():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"best_{model_name}.pth"))

                # early stopping based on val loss
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    counter = 0
                else:
                    counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# 5. Evaluation & Visualization
# ==========================================

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, classes: list, model_name: str, config: Config):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    report = classification_report(all_labels, all_preds, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)

    return report

def plot_curves(histories: Dict[str, dict], config: Config):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history['train_loss'], label=f'{name} Train')
        plt.plot(history['val_loss'], linestyle='--', label=f'{name} Val')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history['train_acc'], label=f'{name} Train')
        plt.plot(history['val_acc'], linestyle='--', label=f'{name} Val')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, 'accuracy_loss_curves.png'))
    plt.close()



# ==========================================
# 7. Efficiency Metrics
# ==========================================

def get_model_efficiency(model: nn.Module, device: torch.device, input_size: int = 224):
    params = sum(p.numel() for p in model.parameters())

    flops = "N/A"
    if HAS_FVCORE:
        try:
            dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
            flops_counter = FlopCountAnalysis(model, dummy_input)
            flops = flops_counter.total()
        except Exception as e:
            print(f"Error calculating FLOPs: {e}")
            flops = "Error"
    else:
        flops = "fvcore_missing"

    return params, flops

# ==========================================
# 8. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train Artwork Style Classifier")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory", default=None)
    parser.add_argument("--output_dir", type=str, help="Path to output directory", default=None)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=None)
    parser.add_argument("--num_workers", type=int, help="num workers", default=4)
    parser.add_argument("--model", type=str, help="Comma-separated list of models to train (e.g. resnet50,efficientnet_b0)", default=None)
    args = parser.parse_args()

    config = Config(args)
    set_seed(config.SEED, reproducible=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model_hyperparams = {
        'resnet50': {'lr': 1e-3, 'batch_size': 128, 'weight_decay': 0.02, 'img_size': 224},
        'efficientnet_b0': {'lr': 1e-3, 'batch_size': 64, 'weight_decay': 0.02, 'img_size': 224},
        'efficientnet_b4': {'lr': 1e-3, 'batch_size': 32, 'weight_decay': 0.02, 'img_size': 380}
    }

    hf_splits = load_and_split_hf_dataset(config, seed=config.SEED)
    classes = hf_splits['train'].features['style'].names
    print(f"Found {len(classes)} classes.")

    results = {}
    histories = {}
    best_model_name = None
    best_f1 = 0.0
    best_model = None

    for model_name in config.MODELS_TO_TRAIN:
        print(f"\n{'='*30}\nProcessing model: {model_name}\n{'='*30}")
        hyper = model_hyperparams.get(model_name, {'lr': 1e-3, 'batch_size': config.BATCH_SIZE, 'weight_decay': 0.02, 'img_size': config.IMG_SIZE})
        model_lr = hyper['lr']
        model_batch_size = hyper['batch_size']
        model_weight_decay = hyper['weight_decay']
        model_img_size = hyper.get('img_size', config.IMG_SIZE)
        print(f"Using LR={model_lr}, Batch Size={model_batch_size}, Weight Decay={model_weight_decay}, Img Size={model_img_size}")

        dataloaders, _ = get_data_loaders_from_hf_splits(hf_splits, config, batch_size=model_batch_size, img_size=model_img_size)

        model = get_model(model_name, len(classes))
        model = model.to(device)

        params, flops = get_model_efficiency(model, device, input_size=model_img_size)

        criterion = nn.CrossEntropyLoss()
        use_fused = (device.type == 'cuda')
        try:
            optimizer = optim.AdamW(model.parameters(), lr=model_lr, weight_decay=model_weight_decay, fused=use_fused)
        except TypeError:
            optimizer = optim.AdamW(model.parameters(), lr=model_lr, weight_decay=model_weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.EPOCHS))

        model, history = train_model(model, dataloaders, criterion, optimizer, scheduler, config.EPOCHS, device, model_name, config)
        histories[model_name] = history

        report = evaluate_model(model, dataloaders['test'], device, classes, model_name, config)

        results[model_name] = {
            "test_accuracy": report.get('accuracy', None) if isinstance(report, dict) else None,
            "macro_f1": report['macro avg']['f1-score'] if isinstance(report, dict) and 'macro avg' in report else None,
            "weighted_f1": report['weighted avg']['f1-score'] if isinstance(report, dict) and 'weighted avg' in report else None,
            "parameters": params,
            "flops": flops
        }

        cur_macro = results[model_name]['macro_f1'] or 0.0
        if cur_macro > best_f1:
            best_f1 = cur_macro
            best_model_name = model_name
            best_model = model

        print(f"Finished {model_name}: params={params}, flops={flops}")

    with open(os.path.join(config.OUTPUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(results, f, indent=4)

    plot_curves(histories, config)

    print("\n" + "="*80)
    print(f"{'Model':<20} | {'Acc':<10} | {'F1 (Macro)':<10} | {'Params':<12} | {'FLOPs':<15}")
    print("-" * 80)
    for name, metrics in results.items():
        flops_str = metrics['flops']
        if isinstance(flops_str, (int, float)):
            try:
                flops_str = f"{flops_str/1e9:.2f}G"
            except Exception:
                flops_str = str(flops_str)
        print(f"{name:<20} | {metrics['test_accuracy'] or 0.0:<10.4f} | {metrics['macro_f1'] or 0.0:<10.4f} | {metrics['parameters']:<12} | {flops_str:<15}")
    print("="*80)
    print(f"All results saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
