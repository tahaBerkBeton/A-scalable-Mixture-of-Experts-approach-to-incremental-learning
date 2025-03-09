####################################
### Inference Code for Mixture-of-Experts
####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

####################################
### Model Definitions (Same as in Training)
####################################
class LeNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(LeNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

class LeNetExpert(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LeNetExpert, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, feature_dim=400):
        super(MixtureOfExperts, self).__init__()
        self.feature_extractor = LeNetFeatureExtractor(in_channels=3)
        self.feature_dim = feature_dim
        self.router = None  # Will be initialized when experts are added
        self.experts = nn.ModuleList()
        self.expert_classes = []  # To be loaded from checkpoint
    def add_expert(self, new_class_indices):
        num_new_classes = len(new_class_indices)
        new_expert = LeNetExpert(self.feature_dim, num_new_classes)
        self.experts.append(new_expert)
        self.expert_classes.append(new_class_indices)
        new_num_experts = len(self.experts)
        old_router = self.router
        new_router = nn.Linear(self.feature_dim, new_num_experts)
        if old_router is not None:
            with torch.no_grad():
                new_router.weight[:old_router.out_features] = old_router.weight
                new_router.bias[:old_router.out_features] = old_router.bias
        self.router = new_router
    def forward(self, x):
        features = self.feature_extractor(x)
        router_logits = self.router(features)
        selected_expert_indices = torch.argmax(router_logits, dim=1)
        return features, router_logits, selected_expert_indices

def evaluate_moe(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, ncols=80, desc="Inference Eval"):
            images, labels = images.to(device), labels.to(device)
            features = model.feature_extractor(images)
            router_logits = model.router(features)
            selected_expert_indices = torch.argmax(router_logits, dim=1)
            total_classes = sum(len(cls_list) for cls_list in model.expert_classes)
            batch_logits = torch.full((images.size(0), total_classes), -1e9, device=device)
            for expert_id, cls_list in enumerate(model.expert_classes):
                idx = (selected_expert_indices == expert_id).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    expert_out = model.experts[expert_id](features[idx])
                    for i, sample_idx in enumerate(idx):
                        for j, global_class in enumerate(cls_list):
                            batch_logits[sample_idx, global_class] = expert_out[i, j]
            pred = torch.argmax(batch_logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

####################################
### Inference Script
####################################
# Set paths and parameters.
checkpoint_path = "./checkpoints/moe_model_task4.pt"  # Update as needed.
batch_size = 64

# Load test dataset.
transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])
root_dir = "./data"
test_dataset  = datasets.GTSRB(root=root_dir, split='test', download=True, transform=transform_test)
test_target = pd.read_csv(
    'https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/test_target.csv',
    delimiter=',', header=None).to_numpy().squeeze().tolist()

# Create test dataloader for all classes.
all_classes = list(range(len(np.unique(test_target))))
test_loader = DataLoader(Subset(test_dataset, [i for i, label in enumerate(test_target) if label in all_classes]),
                         batch_size=batch_size, shuffle=False)

# Instantiate the model architecture.
model = MixtureOfExperts(feature_dim=400)
model.to(device)

# Load checkpoint.
checkpoint = torch.load(checkpoint_path, map_location=device)

# Reconstruct the model architecture using saved expert_classes.
for expert_cls in checkpoint['expert_classes']:
    model.add_expert(expert_cls)

# Now load the saved state dict.
model.load_state_dict(checkpoint['state_dict'])
# (At this point, model.expert_classes will match the saved mapping.)
model.eval()

# Evaluate on test set.
accuracy = evaluate_moe(model, test_loader, device)
print(f"Inference Test Accuracy: {accuracy:.2f}%")
