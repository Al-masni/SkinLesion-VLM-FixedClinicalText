"""
ConvNeXt-Tiny, vision-only baseline (ISIC 2017, 3 classes).
Direct .py conversion of the notebook (cells kept in the same order, same code).
Run:   DATA_ROOT="/path/to/02 192x256 RGB" python convnext_vision_only.py
Only changes vs. the notebook: tqdm.auto instead of tqdm.notebook, figures are saved as PNG instead of shown,
data path read from the DATA_ROOT environment variable, cluster cache paths used only if they exist,
and distinct output file names.
"""

import sys
import numpy as np
import torch
# training results deterministic (Fixed seeds)
import os, random
import torch

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuBLAS determinism (important for reproducible GPU matmul)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import random

train_flag = 1
test_flag = 1
model_path = "convnext_vision_only.pth"

# Read Data
DATA_ROOT = os.environ.get("DATA_ROOT")   # folder with Train/ Valid/ Test/
train_dir = os.path.join(DATA_ROOT, "Train")
valid_dir = os.path.join(DATA_ROOT, "Valid")
test_dir = os.path.join(DATA_ROOT, "Test")


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchvision.datasets import DatasetFolder
from PIL import Image

class ImageOnlyFolder(DatasetFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, loader=self.pil_loader, extensions=('png',), transform=transform)
    
    def pil_loader(self, path):
        return Image.open(path).convert('RGB')
    
    def find_classes(self, dir):
        # B, M, SK folders
        classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None, *args, **kwargs):
        images = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            image_dir = os.path.join(directory, target_class, "image")  # focus only on "image" folder
            if not os.path.isdir(image_dir):
                continue
            for root, _, fnames in sorted(os.walk(image_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if path.lower().endswith(".png"):
                        images.append((path, class_index))
        return images


transform = transforms.Compose([
    transforms.Resize((192, 256)),
    transforms.ToTensor()
])

train_data = ImageOnlyFolder(root=train_dir, transform=transform)
valid_data = ImageOnlyFolder(root=valid_dir, transform=transform)
test_data  = ImageOnlyFolder(root=test_dir,  transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)



import torch.optim as optim
from torchvision import models

#  Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ConvNext
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

in_features = model.classifier[-1].in_features  # usually 768

model.classifier = nn.Sequential(
    model.classifier[0],        # keep the original LayerNorm (important)
    nn.Flatten(1),              # keep flatten
    
    nn.Linear(in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),

    nn.Linear(256, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),

    nn.Linear(64, 3)
)

#  Move to device
model = model.to(device)

#  Define loss (no weighting — baseline)
criterion = nn.CrossEntropyLoss()

#  Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#  Learning rate scheduler (optional, but useful)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3)

import torch.nn.functional as F
# Global lists to store metrics (optional: for saving to CSV)
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []

def train_model_with_progress(model, train_loader, valid_loader, criterion, optimizer, epochs=10, scheduler=None):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        valid_loss_list.append(val_epoch_loss)
        valid_acc_list.append(val_epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

        # Scheduler update (if ReduceLROnPlateau)
        if scheduler:
            scheduler.step(val_epoch_loss)

    return model

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import numpy as np

if train_flag:
    model = train_model_with_progress(
        model, train_loader, valid_loader, criterion, optimizer, epochs=50, scheduler=scheduler)

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    import pandas as pd
    df_metrics = pd.DataFrame({
        'epoch': list(range(len(train_acc_list))),
        'train_acc': train_acc_list,
        'valid_acc': valid_acc_list,
        'train_loss': train_loss_list,
        'valid_loss': valid_loss_list,
    })
    df_metrics.to_csv("convnext_vision_only_metrics.csv", index=False)
    print("Training/Validation metrics saved.")

if test_flag:
    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Evaluate on test set
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Classification report & Confusion Matrix
    print("\n✅ Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_data.classes))

    print("✅ Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # F1 Score (macro + per class)
    f1 = f1_score(all_labels, all_preds, average=None)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    for idx, cls in enumerate(train_data.classes):
        print(f"F1-score ({cls}): {f1[idx]:.4f}")
    print(f"F1-score (Macro Avg): {f1_macro:.4f}")

    # AUC Score
    all_labels_onehot = np.eye(len(train_data.classes))[all_labels]
    auc_score = roc_auc_score(all_labels_onehot, all_probs, average="macro", multi_class='ovr')
    print(f"AUC (Macro Avg): {auc_score:.4f}")

    # Compute ROC curve and AUC for each class
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(all_labels_onehot[:, i], np.array(all_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8,6))
    for i, name in enumerate(train_data.classes):
        plt.plot(fpr[i], tpr[i], label=f"ROC curve for {name} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend()
    plt.savefig(f"fig_{_fig_counter()}.png", dpi=150); plt.close()

try:
    import pytorch_grad_cam   # optional: pip install grad-cam
except ImportError:
    pytorch_grad_cam = None
    print('pytorch-grad-cam not installed: skipping the Grad-CAM example')

if pytorch_grad_cam is not None:
    import matplotlib.pyplot as plt

    import torch
    import numpy as np
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()

    # pick the last conv layer of ConvNext
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    images, labels = next(iter(test_loader))
    images = images.to(device)

    # Forward pass (no gradients needed here)
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    # Grad-CAM needs targets
    targets = [ClassifierOutputTarget(int(p)) for p in preds]

    # Compute CAM
    grayscale_cam = cam(
        input_tensor=images,
        targets=targets
    )  # shape: [B, H, W]

    # Visualize one example
    k=0

    img = images[k].detach().cpu().permute(1,2,0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    cam_img = show_cam_on_image(img, grayscale_cam[k], use_rgb=True)


    # 0: Benign nevus, 1: Melanoma, 2: Seborrheic keratosis
    class_names = {0: "Benign (Nevus)", 1: "Melanoma", 2: "SK"}
    true_label = class_names[int(labels[k])]
    pred_label = class_names[int(preds[k])]


    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_img)
    axes[1].set_title(f"Grad-CAM\nGT: {true_label} | Pred: {pred_label}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"fig_{_fig_counter()}.png", dpi=150); plt.close()
