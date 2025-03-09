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

####################################
### Helper Function: Ground Truth Expert Info
####################################
def get_ground_truth_expert_info(labels, expert_classes, device):
    """
    For each global label, determine:
      - gt_expert: the expert index that should handle the label.
      - local_label: the label index within that expert.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    gt_expert, local_label = [], []
    for lbl in labels:
        found = False
        for expert_id, cls_list in enumerate(expert_classes):
            if lbl in cls_list:
                gt_expert.append(expert_id)
                local_label.append(cls_list.index(lbl))
                found = True
                break
        if not found:
            gt_expert.append(-1)
            local_label.append(-1)
    return torch.tensor(gt_expert, device=device), torch.tensor(local_label, device=device)

####################################
### New: Inference Evaluation with Advanced Statistics
####################################
def evaluate_moe_with_stats(model, dataloader, device):
    """
    Evaluates the model on the given dataloader and computes advanced statistics:
      - Overall accuracy.
      - Router errors: cases where the router selected the wrong expert.
      - Expert errors: cases where the correct expert was selected, but the expert misclassified.
      - Per-expert statistics.
    Returns overall accuracy (in %) and a dictionary with detailed stats.
    """
    model.eval()
    total_samples = 0
    router_error_count = 0  # Count of samples misrouted.
    expert_error_count = 0  # Count of errors from experts when routing is correct.
    
    # Initialize per-expert stats.
    expert_stats = {i: {'samples': 0, 'errors': 0} for i in range(len(model.expert_classes))}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, ncols=80, desc="Inference Eval with Stats"):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Extract features and get router outputs.
            features = model.feature_extractor(images)
            router_logits = model.router(features)
            selected_expert = torch.argmax(router_logits, dim=1)
            
            # Get ground truth expert and local labels.
            gt_expert, local_labels = get_ground_truth_expert_info(labels, model.expert_classes, device)
            
            # Count router errors: when the router-selected expert does not match the ground truth expert.
            router_errors_mask = (selected_expert != gt_expert)
            router_error_count += router_errors_mask.sum().item()
            
            # For samples with correct routing, evaluate the expert's classification.
            correct_routing_mask = (selected_expert == gt_expert)
            indices_correct = correct_routing_mask.nonzero(as_tuple=True)[0]
            if indices_correct.numel() > 0:
                # For each expert, only consider samples that belong to that expert (and are correctly routed).
                for expert_id in range(len(model.expert_classes)):
                    idx = ((gt_expert == expert_id) & (selected_expert == expert_id)).nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        expert_out = model.experts[expert_id](features[idx])
                        local_pred = expert_out.argmax(dim=1)
                        errors = (local_pred != local_labels[idx]).sum().item()
                        expert_error_count += errors
                        expert_stats[expert_id]['samples'] += idx.numel()
                        expert_stats[expert_id]['errors'] += errors
                        
    overall_correct = total_samples - router_error_count - expert_error_count
    overall_accuracy = 100 * overall_correct / total_samples
    router_error_rate = 100 * router_error_count / total_samples
    if total_samples - router_error_count > 0:
        expert_error_rate = 100 * expert_error_count / (total_samples - router_error_count)
    else:
        expert_error_rate = 0.0
        
    stats = {
        'total_samples': total_samples,
        'router_error_count': router_error_count,
        'router_error_rate': router_error_rate,
        'expert_error_count': expert_error_count,
        'expert_error_rate': expert_error_rate,
        'expert_stats': expert_stats
    }
    return overall_accuracy, stats

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

# Evaluate on test set with advanced statistics.
overall_accuracy, stats = evaluate_moe_with_stats(model, test_loader, device)
print(f"Inference Test Accuracy: {overall_accuracy:.2f}%")
print(f"Total Samples: {stats['total_samples']}")
print(f"Router Errors: {stats['router_error_count']} ({stats['router_error_rate']:.2f}%)")
print(f"Expert Errors (on correctly routed samples): {stats['expert_error_count']} ({stats['expert_error_rate']:.2f}%)")
print("Per-Expert Statistics:")
for expert_id, stat in stats['expert_stats'].items():
    if stat['samples'] > 0:
        rate = 100 * stat['errors'] / stat['samples']
    else:
        rate = 0.0
    print(f"  Expert {expert_id}: {stat['samples']} samples, {stat['errors']} errors, error rate: {rate:.2f}%")
