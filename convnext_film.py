"""
ConvNeXt-Tiny + fixed-text FiLM (ISIC 2017, 3 classes).
Direct .py conversion of the notebook (cells kept in the same order, same code).
Run:   DATA_ROOT="/path/to/02 192x256 RGB" python convnext_film.py
Only changes vs. the notebook: tqdm.auto instead of tqdm.notebook, figures are saved as PNG instead of shown,
data path read from the DATA_ROOT environment variable, cluster cache paths used only if they exist,
and distinct output file names.
"""

# training results deterministic (Fixed seeds)
import os, random
import numpy as np
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

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import random

# ---------------- notebook cell 5 ----------------
train_flag = 1
test_flag = 1
model_path = "convnext_film.pth"

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

print("Classes:", train_data.classes)
print("Class indices:", train_data.class_to_idx)
print("Total training images:", len(train_data))
print("Total train batches:", len(train_loader))
print("Total valid images:", len(valid_data))
print("Total test images:", len(test_data))

# Display some examples:
import matplotlib.pyplot as plt
import numpy as np

# Helper function to convert tensor to numpy image
def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))  # CHW to HWC
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.savefig(f"fig_{_fig_counter()}.png", dpi=150); plt.close()

# Get one batch from train_loader
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Display first 5 images with labels
for i in range(1):
    imshow(images[i], title=f"Label: {train_data.classes[labels[i]]}")

# ---------------- notebook cell 10 ----------------
# !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import torch

# ONE GLOBAL TEXT:
global_prompt = """
Dermoscopy triage for three lesion types: benign melanocytic nevus, melanoma, and seborrheic keratosis. 
Focus on discriminative dermoscopic criteria. 
Melanoma often shows asymmetry of structure and color, atypical pigment network, irregular streaks, irregular dots/globules, 
blue-white veil, negative network, regression structures (scar-like depigmentation and peppering), irregular blotches, and atypical vascular patterns. 
Benign nevi tend to show symmetry, uniform pigment network, regular dots/globules, and homogeneous color without atypical regression. 
Seborrheic keratosis is suggested by milia-like cysts, comedo-like openings, fissures and ridges (brain-like/cerebriform pattern), sharp demarcation, 
moth-eaten borders, and hairpin vessels with whitish halo; pigment network is usually absent or non-melanocytic. Ignore hair, ruler marks, 
and illumination differences; prioritize lesion morphology and dermoscopic structures.
"""

# text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=OFFLINE_HF)

t_global = torch.tensor(text_encoder.encode(global_prompt), dtype=torch.float32)  # [384]
t_global = t_global / (t_global.norm() + 1e-8)  # normalize


# ConvNext + Text FiLM Fusion Model
import torch.nn as nn
from torchvision import models

class ConvNeXtTiny_TextFiLM(nn.Module):
    def __init__(self, text_dim=384, num_classes=3):
        super().__init__()

        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # ConvNeXt classifier (torchvision) is typically:
        # Sequential(LayerNorm2d, Flatten, Linear(768->1000))
        in_features = self.backbone.classifier[-1].in_features  # usually 768

        # Keep LN2d + Flatten, remove final Linear
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0],  # LayerNorm2d
            nn.Flatten(1),                # Flatten to [B,768]
            nn.Identity()                 # so backbone outputs [B,768]
        )
        self.num_features = in_features  # 768

        # Text → FiLM parameters
        self.film = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * self.num_features)  # gamma + beta
        )

        # Classifier head (same style as yours)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x, t_global):
        # backbone output: [B,768]
        f = self.backbone(x)

        # expand text embedding to batch
        if t_global.dim() == 1:
            t_global = t_global.unsqueeze(0).expand(f.size(0), -1)

        gamma_beta = self.film(t_global.to(f.device))
        gamma, beta = gamma_beta.chunk(2, dim=1)

        f_mod = (1.0 + gamma) * f + beta
        
#         print(f.shape, t_global.shape, gamma.shape, beta.shape)
        
        return self.classifier(f_mod)

import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtTiny_TextFiLM(text_dim=384, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

from tqdm.auto import tqdm
import torch.nn.functional as F

# Global lists to store metrics (optional: for saving to CSV)
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []

set_seed(42)

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
            
            #outputs = model(images)
            outputs = model(images, t_global.to(device))
            
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
                
                #outputs = model(images)
                outputs = model(images, t_global.to(device))
                
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
    df_metrics.to_csv("convnext_film_metrics.csv", index=False)
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
            
            #outputs = model(images)
            outputs = model(images, t_global.to(device))
            
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

    # Our model’s forward() requires two inputs: forward(self, x, t_global)
    # But pytorch-grad-cam expects a model that takes only images.
    # Solution: use a small wrapper model >> We “freeze” the text input inside a wrapper so Grad-CAM only sees images.
    class VisionOnlyWrapper(nn.Module):
        def __init__(self, model, t_global):
            super().__init__()
            self.model = model
            self.t_global = t_global

        def forward(self, x):
            return self.model(x, self.t_global)

    model.eval()

    wrapper_model = VisionOnlyWrapper(model=model, t_global=t_global.to(device)).to(device)   # fixed global text

    # pick the last conv layer of ConvNext
    #target_layers = [model.features[-1]]
    target_layers = [wrapper_model.model.backbone.features[-1]]

    cam = GradCAM(model=wrapper_model, target_layers=target_layers)

    images, labels = next(iter(test_loader))
    images = images.to(device)

    # Forward pass (no gradients needed here)
    with torch.no_grad():
        outputs = wrapper_model(images)
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
