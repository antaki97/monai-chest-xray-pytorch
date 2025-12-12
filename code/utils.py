import torch 

def find_image_path(filename, root, n_folders=12):
    """Return full Path to filename by searching images_001 .. images_00{n_folders}."""
    for i in range(1, n_folders + 1):
        folder = root / f"images_{i:03d}" / "images"
        path = folder / filename
        if path.exists():
            return path
    return None

from PIL import Image, UnidentifiedImageError 

def plot_random_xray(df, root, n_folders=12):
    """Pick a random row from df, load the image safely and plot with ground-truth title."""
    row = df.sample(1).iloc[0]
    filename = row["Image Index"]
    label = row["Finding Labels"] 

    path = find_image_path(filename, root=root, n_folders=n_folders)
    if path is None:
        raise FileNotFoundError(f"Could not find image {filename} in {root}")

    try:
        with Image.open(path) as img:
            # PIL.Image.open returns a file-like object; use copy() if you plan to keep it after closing
            img = img.convert("L")  # ensure grayscale
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap="gray")
            plt.title(label)
            plt.axis("off")
            plt.show()
    except UnidentifiedImageError:
        print(f"Image {path} appears to be corrupted or is not a supported image.")
    except Exception as e:
        print(f"Unexpected error opening {path}: {e}")


def check_for_corrupted_images(root):
    corrupted = []
    
    # Write a loop to loop through all image folders: images_001, images_002, images_003, ... images_012
    for i in range(1, 13):
        folder = root / f"images_{i:03d}" / "images"
        print(f"Checking folder: {folder}")
        
        for img_path in folder.glob("*.png"):
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img.load()
                
            except Exception as e:
                corrupted.append((img_path, str(e)))
    return corrupted

import os
from pathlib import Path

INPUT_DIR = r"C:\Users\aanta\Documents\ML\Projects\ChestXRays"
root = Path(r"C:\Users\aanta\Documents\ML\Projects\ChestXRays\kagglehub\data")

def get_image_path(filename):
    """Get the image path given a filename.

    Args:
        filename (str): The filename itself

    Raises:
        FileNotFoundError: Raise file not found error if the file does not exist

    Returns:
        candidate (str): The folder and filepath
    """
    for i in range(1, 13):
        folder = root / f"images_{i:03d}" / "images"
        candidate = folder / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Image not found in any folder: {filename}")

def make_file_list(df_subset, label_cols):
    files = []
    missing = []
    for _, row in df_subset.iterrows():
        fname = row["Image Index"]
        try:
            path = get_image_path(fname)
        except FileNotFoundError:
            missing.append(fname)
            continue
        label_vec = row[label_cols].astype(float).tolist()
        files.append({"img": path, "label": label_vec})
    if missing:
        print(f"Warning: {len(missing)} missing images (first 5):", missing[:5])
    return files 

from torchvision import models 

def build_densenet(num_classes, pretrained = True, adapt_first_conv = True):
    model = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT) 
    
    num_ftrs = model.classifier.in_features
    print(f"Original input features to classifier: {num_ftrs}")
    
    model.classifier = nn.Linear(in_features = num_ftrs, 
                                 out_features = num_classes)
    
    print(f"New model output layer structure: {model.classifier}")

    for param in model.parameters():
        param.requires_grad = False 
        
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    
    return model 

from sklearn.metrics import roc_auc_score
import torch 
from tqdm import tqdm 
import numpy as np 

def validate_one_epoch(model, loader, DEVICE = "cuda"):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='val'):
            imgs = batch['img'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Handle varying number of channels - force to 1 channel if needed
            if imgs.shape[1] > 1:
                imgs = imgs[:, :1, :, :]  # Take only first channel
            
            # Repeat grayscale to 3 channels for DenseNet
            imgs = imgs.repeat(1, 3, 1, 1)
            
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    per_class_auroc = []
    for i in range(all_targets.shape[1]):
        try:
            auroc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        except ValueError:
            auroc = float('nan')
        per_class_auroc.append(auroc)
    macro_auroc = np.nanmean(per_class_auroc)
    return macro_auroc, per_class_auroc, all_probs, all_targets 

def custom_collate_fn(batch):
    """Custom collate function to handle varying image shapes"""
    images = []
    labels = []
    
    for item in batch:
        img = item['img']
        label = item['label']
        
        # Ensure image has correct shape [C, H, W]
        if isinstance(img, torch.Tensor):
            # If image has unexpected number of channels, take only first channel
            if img.ndim == 3 and img.shape[0] > 1:
                img = img[:1, :, :]  # Keep only first channel
            elif img.ndim == 2:
                img = img.unsqueeze(0)  # Add channel dimension
        
        images.append(img)
        labels.append(label)
    
    # Stack into batches
    try:
        images = torch.stack(images)
        labels = torch.stack(labels)
    except RuntimeError as e:
        print(f"Error stacking - image shapes: {[img.shape for img in images]}")
        raise e
    
    return {'img': images, 'label': labels} 

def labels_to_tensor_on_device(label_batch, imgs_batch, device, num_classes):
    import numpy as _np
    if isinstance(label_batch, torch.Tensor):
        labels = label_batch.float()
    elif isinstance(label_batch, _np.ndarray):
        labels = torch.from_numpy(label_batch).float()
    elif isinstance(label_batch, list):
        if all(isinstance(x, torch.Tensor) for x in label_batch):
            labels = torch.stack([x.float() for x in label_batch])
        else:
            labels = torch.tensor(label_batch, dtype=torch.float32)
    else:
        labels = torch.as_tensor(label_batch, dtype=torch.float32)

    B = imgs_batch.shape[0]
    if labels.dim() == 2 and labels.shape[0] != B and labels.shape[1] == B:
        labels = labels.T
    if labels.dim() == 1 and B > 1 and labels.shape[0] == num_classes:
        labels = labels.unsqueeze(0).repeat(B, 1)
    return labels.to(device, non_blocking=True)


from monai.transforms import Compose, MapTransform

# 1. Define custom transforms
class LoadAndPreprocessImageD(MapTransform):
    """Load image, ensure grayscale, resize, and normalize"""
    def __init__(self, keys, image_size=224):
        super().__init__(keys)
        self.image_size = image_size
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img_path = d[key]
            img = Image.open(img_path).convert("L")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array[np.newaxis, ...]
            d[key] = torch.from_numpy(img_array)
        return d

class RandomFlipD(MapTransform):
    """Random horizontal flip"""
    def __init__(self, keys, prob=0.5):
        super().__init__(keys)
        self.prob = prob
    
    def __call__(self, data):
        d = dict(data)
        if np.random.random() < self.prob:
            for key in self.keys:
                d[key] = torch.flip(d[key], dims=[2])
        return d

class LabelToTensorD(MapTransform):
    """Convert label list to tensor"""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.tensor(d[key], dtype=torch.float32)
        return d 
    
# helper to convert labels robustly and move to device
def labels_to_tensor_on_device(label_batch, imgs_batch, device, num_classes):
    import numpy as _np
    if isinstance(label_batch, torch.Tensor):
        labels = label_batch.float()
    elif isinstance(label_batch, _np.ndarray):
        labels = torch.from_numpy(label_batch).float()
    elif isinstance(label_batch, list):
        if all(isinstance(x, torch.Tensor) for x in label_batch):
            labels = torch.stack([x.float() for x in label_batch])
        else:
            labels = torch.tensor(label_batch, dtype=torch.float32)
    else:
        labels = torch.as_tensor(label_batch, dtype=torch.float32)

    B = imgs_batch.shape[0]
    if labels.dim() == 2 and labels.shape[0] != B and labels.shape[1] == B:
        labels = labels.T
    if labels.dim() == 1 and B > 1 and labels.shape[0] == num_classes:
        labels = labels.unsqueeze(0).repeat(B, 1)
    return labels.to(device, non_blocking=True) 

from torch import nn 

def build_effnetb2(num_classes, pretrained = True, adapt_first_conv = True):
    model = models.efficientnet_b2(weights = models.EfficientNet_B2_Weights.DEFAULT) 
    
    num_ftrs = model.classifier[1].in_features
    print(f"Original input features to classifier: {num_ftrs}")
    
    model.classifier = nn.Linear(in_features = num_ftrs, 
                                 out_features = num_classes)
    
    print(f"New model output layer structure: {model.classifier}")

    for param in model.parameters():
        param.requires_grad = False 
        
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    
    return model 

import importlib
# importlib.reload(utils)
# print("'torch' after reload? ->", "torch" in utils.__dict__)