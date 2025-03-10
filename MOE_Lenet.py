####################################
### Useful imports
####################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, datasets
import numpy as np
import random
import time, os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import math
import copy

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory for saving model checkpoints
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

####################################
### Data loaders & Transforms 
####################################
transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

def get_dataset(root_dir, transform, train=True):
    dataset = datasets.GTSRB(root=root_dir, split='train' if train else 'test',
                              download=True, transform=transform)
    target = [data[1] for data in dataset]
    return dataset, target

def create_dataloader(dataset, targets, selected_classes, batch_size, shuffle):
    indices = [i for i, label in enumerate(targets) if label in selected_classes]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

# Dataset initialization
root_dir = './data'
train_dataset = datasets.GTSRB(root=root_dir, split='train', download=True, transform=transform_train)
test_dataset  = datasets.GTSRB(root=root_dir, split='test', download=True, transform=transform_test)

# Load targets and class names
train_target = pd.read_csv(
    'https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/train_target.csv',
    delimiter=',', header=None).to_numpy().squeeze().tolist()
test_target = pd.read_csv(
    'https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/test_target.csv',
    delimiter=',', header=None).to_numpy().squeeze().tolist()
class_names = pd.read_csv(
    'https://raw.githubusercontent.com/stepherbin/teaching/refs/heads/master/IOGS/projet/signnames.csv'
)['SignName'].tolist()

# Candidate pool for memory buffer (using test split)
buffer_pool_dataset = datasets.GTSRB(root=root_dir, split='test', download=True, transform=transform_test)
buffer_pool_dataset.targets = test_target

####################################
### Model Definitions: Mixture-of-Experts with Improved LeNet-5
####################################
class LeNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(LeNetFeatureExtractor, self).__init__()
        # Conv1: 6 filters, kernel size 5 -> output: 6 x 28 x 28
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        # Conv2: 16 filters, kernel size 5 -> output: 16 x 10 x 10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        # After pooling: 16 x 5 x 5 = 400 features
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # [B, 400]

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
        self.router = None  # Will be initialized when the first expert is added
        self.experts = nn.ModuleList()
        self.expert_classes = []  # List of lists: each expert's assigned global classes

    def add_expert(self, new_class_indices):
        """Add a new expert for the given list of global class indices."""
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
        features = self.feature_extractor(x)  # [B, 400]
        router_logits = self.router(features)   # [B, num_experts]
        selected_expert_indices = torch.argmax(router_logits, dim=1)
        return features, router_logits, selected_expert_indices

####################################
### Helper Functions
####################################
def get_ground_truth_expert_info(labels, expert_classes, device):
    """
    For each global label, return:
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

def build_memory_buffer(expert_classes, candidate_dataset, total_buffer_size, current_task_idx):
    """
    Build a memory buffer of exactly total_buffer_size items drawn equally from past tasks.
    """
    past_expert_classes = expert_classes[:current_task_idx]
    if len(past_expert_classes) == 0:
        return []
    total_past_classes = sum(len(task_classes) for task_classes in past_expert_classes)
    samples_per_class = total_buffer_size // total_past_classes
    remainder = total_buffer_size - samples_per_class * total_past_classes
    memory_buffer = []
    for task_classes in past_expert_classes:
        for cls in task_classes:
            candidate_indices = [i for i, label in enumerate(candidate_dataset.targets) if label == cls]
            n_samples = samples_per_class + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            if candidate_indices:
                selected = np.random.choice(candidate_indices, min(n_samples, len(candidate_indices)), replace=False)
                for idx in selected:
                    memory_buffer.append(candidate_dataset[idx])  # (image, label)
    return memory_buffer

def evaluate_moe(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, ncols=80, desc="Eval"):
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
###  Functions for Multiple Training Attempts
####################################
def train_initial_task(model, task_loader, test_dataset, test_target, current_classes, 
                      batch_size, num_epochs, lr, device, alignment_strength=2.0):
    """
    Trains the feature extractor and first expert for the initial task.
    Returns the best accuracy achieved during training.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_accuracy = 0
    best_model_state = None
    
    # First task typically requires more epochs (using 50 as specified in original code)
    first_task_epochs = 50
    
    for epoch in range(first_task_epochs):
        pbar = tqdm(task_loader, ncols=80, desc=f"Initial Task, Epoch {epoch+1}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            features, router_logits, _ = model(images)
            gt_expert, local_labels = get_ground_truth_expert_info(labels, model.expert_classes, device)
            
            # Get samples for the expert
            expert_id = 0  # First expert
            mask = (gt_expert == expert_id).nonzero(as_tuple=True)[0]
            
            if mask.numel() > 0:
                features_sel = features[mask]
                expert_logits = model.experts[expert_id](features_sel)
                local_labels_sel = local_labels[mask]
                classification_loss = criterion(expert_logits, local_labels_sel)
            else:
                classification_loss = 0.0
                
            # Router loss
            routing_loss = criterion(router_logits, gt_expert)
            
            # Total loss
            total_loss = classification_loss + alignment_strength * routing_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": total_loss.item()})
        
        # Evaluate after each epoch
        eval_loader = create_dataloader(test_dataset, test_target, current_classes, batch_size, shuffle=False)
        epoch_accuracy = evaluate_moe(model, eval_loader, device)
        print(f"Initial Task, Epoch {epoch+1}: Eval Accuracy = {epoch_accuracy:.2f}% (LR = {lr:.5f})")
        
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_accuracy

def train_subsequent_task(model, task_loader, buffer_loader, test_dataset, test_target, current_classes,
                         batch_size, num_epochs, lr, device, alignment_strength=2.0, buffer_weight=2.0):
    """
    Trains a new expert for a subsequent task, starting from a pre-trained model.
    Returns the best accuracy achieved during training.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    # Only train new expert and router (filter parameters that require gradients)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    best_accuracy = 0
    best_model_state = None
    
    task_iter = iter(task_loader)
    if buffer_loader is not None:
        buffer_iter = iter(buffer_loader)
        num_batches = max(len(task_loader), len(buffer_loader))
    else:
        num_batches = len(task_loader)
    
    for epoch in range(num_epochs):
        pbar = tqdm(range(num_batches), ncols=80, desc=f"Subsequent Task, Epoch {epoch+1}")
        for _ in pbar:
            # Get new task data
            try:
                images_new, labels_new = next(task_iter)
            except StopIteration:
                task_iter = iter(task_loader)
                images_new, labels_new = next(task_iter)
            
            new_images = images_new.to(device)
            new_labels = labels_new.to(device)
            
            features_new, router_logits_new, _ = model(new_images)
            gt_expert_new, local_labels_new = get_ground_truth_expert_info(new_labels, model.expert_classes, device)
            
            # Get samples for the new expert
            new_expert_id = len(model.experts) - 1
            mask_new = (gt_expert_new == new_expert_id).nonzero(as_tuple=True)[0]
            
            if mask_new.numel() > 0:
                features_new_sel = features_new[mask_new]
                expert_logits_new = model.experts[new_expert_id](features_new_sel)
                local_labels_new_sel = local_labels_new[mask_new]
                classification_loss_new = criterion(expert_logits_new, local_labels_new_sel)
            else:
                classification_loss_new = 0.0
                
            routing_loss_new = criterion(router_logits_new, gt_expert_new)
            
            # Process buffer data if available
            if buffer_loader is not None:
                try:
                    images_buf, labels_buf = next(buffer_iter)
                except StopIteration:
                    buffer_iter = iter(buffer_loader)
                    images_buf, labels_buf = next(buffer_iter)
                
                buf_images = images_buf.to(device)
                buf_labels = labels_buf.to(device)
                
                features_buf, router_logits_buf, _ = model(buf_images)
                gt_expert_buf, local_labels_buf = get_ground_truth_expert_info(buf_labels, model.expert_classes, device)
                
                classification_loss_buf = 0.0
                for expert_id, cls_list in enumerate(model.expert_classes):
                    idx = (gt_expert_buf == expert_id).nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        expert_logits_buf = model.experts[expert_id](features_buf[idx])
                        local_labels_buf_sel = local_labels_buf[idx]
                        classification_loss_buf += criterion(expert_logits_buf, local_labels_buf_sel)
                
                routing_loss_buf = criterion(router_logits_buf, gt_expert_buf)
            else:
                classification_loss_buf = 0.0
                routing_loss_buf = 0.0
            
            # Total loss
            total_loss = (classification_loss_new +
                         buffer_weight * classification_loss_buf +
                         alignment_strength * (routing_loss_new + routing_loss_buf))
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": total_loss.item()})
        
        # Evaluate after each epoch
        eval_loader = create_dataloader(test_dataset, test_target, current_classes, batch_size, shuffle=False)
        epoch_accuracy = evaluate_moe(model, eval_loader, device)
        print(f"Subsequent Task, Epoch {epoch+1}: Eval Accuracy = {epoch_accuracy:.2f}% (LR = {lr:.5f})")
        
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_accuracy

def incremental_learning_moe_with_retries(train_dataset, train_target, test_dataset, test_target,
                                num_tasks, classes_per_task, batch_size, num_epochs, lr, device,
                                buffer_size=1000, alignment_strength=2.0, buffer_weight=2.0,
                                num_retries=2):
    """
    Performs incremental learning using multiple training attempts for each task,
    selecting the best performing model for each task.
    """
    nclasses = len(np.unique(train_target))
    all_classes = list(range(nclasses))
    current_classes = []
    accuracies = []
    
    best_overall_model = None
    
    for task in range(num_tasks):
        # Define new task classes
        task_classes = all_classes[task * classes_per_task : (task + 1) * classes_per_task]
        current_classes.extend(task_classes)
        print(f"\n--- Starting Task {task+1} with classes: {task_classes} ---")
        
        best_task_accuracy = 0
        best_task_model = None
        
        # For the first task, we train the feature extractor and first expert multiple times
        # For subsequent tasks, we use the best model so far and train new router/expert multiple times
        if task == 0:
            for attempt in range(num_retries):
                print(f"\nTask {task+1}, Attempt {attempt+1}/{num_retries}")
                
                # Create a new model for each attempt
                model = MixtureOfExperts(feature_dim=400)
                model.to(device)
                model.add_expert(task_classes)
                
                # Create data loader for current task
                task_loader = create_dataloader(train_dataset, train_target, task_classes, batch_size, shuffle=True)
                
                # Train the model for this attempt
                task_accuracy = train_initial_task(model, task_loader, test_dataset, test_target, 
                                                  current_classes, batch_size, num_epochs, lr, device)
                
                print(f"Task {task+1}, Attempt {attempt+1}: Final Accuracy = {task_accuracy:.2f}%")
                
                # Update best model if this attempt is better
                if task_accuracy > best_task_accuracy:
                    best_task_accuracy = task_accuracy
                    best_task_model = copy.deepcopy(model)
                    
                    # Save this model as a checkpoint for this attempt
                    checkpoint_path = os.path.join(save_dir, f"moe_model_task{task+1}_attempt{attempt+1}.pt")
                    torch.save({
                        'state_dict': model.state_dict(),
                        'expert_classes': model.expert_classes
                    }, checkpoint_path)
        else:
            # For subsequent tasks, start from the best model so far and try training the new expert multiple times
            memory_buffer = build_memory_buffer(best_overall_model.expert_classes, buffer_pool_dataset,
                                               total_buffer_size=buffer_size, current_task_idx=task)
            
            if memory_buffer:
                buffer_images, buffer_labels = zip(*memory_buffer)
                buffer_images = torch.stack(buffer_images)
                buffer_labels = torch.tensor(buffer_labels)
                buffer_dataset = TensorDataset(buffer_images, buffer_labels)
                buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=True)
            else:
                buffer_loader = None
                
            task_loader = create_dataloader(train_dataset, train_target, task_classes, batch_size, shuffle=True)
            
            for attempt in range(num_retries):
                print(f"\nTask {task+1}, Attempt {attempt+1}/{num_retries}")
                
                # Create a copy of the best model so far
                model = copy.deepcopy(best_overall_model)
                
                # Add a new expert for the current task
                model.add_expert(task_classes)
                model.to(device)
                
                # Freeze feature extractor and previous experts
                for param in model.feature_extractor.parameters():
                    param.requires_grad = False
                for expert in model.experts[:-1]:
                    for param in expert.parameters():
                        param.requires_grad = False
                
                # Train the model for this attempt
                task_accuracy = train_subsequent_task(model, task_loader, buffer_loader, test_dataset, test_target,
                                                     current_classes, batch_size, num_epochs, lr, device,
                                                     alignment_strength, buffer_weight)
                
                print(f"Task {task+1}, Attempt {attempt+1}: Final Accuracy = {task_accuracy:.2f}%")
                
                # Update best model if this attempt is better
                if task_accuracy > best_task_accuracy:
                    best_task_accuracy = task_accuracy
                    best_task_model = copy.deepcopy(model)
                    
                    # Save this model as a checkpoint for this attempt
                    checkpoint_path = os.path.join(save_dir, f"moe_model_task{task+1}_attempt{attempt+1}.pt")
                    torch.save({
                        'state_dict': model.state_dict(),
                        'expert_classes': model.expert_classes
                    }, checkpoint_path)
        
        # Update the best overall model with the best model for this task
        best_overall_model = best_task_model
        accuracies.append(best_task_accuracy)
        
        # Save the best model for this task as the final checkpoint
        checkpoint_path = os.path.join(save_dir, f"moe_model_task{task+1}_best.pt")
        torch.save({
            'state_dict': best_overall_model.state_dict(),
            'expert_classes': best_overall_model.expert_classes
        }, checkpoint_path)
        print(f"Task {task+1}: Best Accuracy = {best_task_accuracy:.2f}%")
    
    return accuracies, best_overall_model

####################################
### Main: Incremental Learning on GTSRB using Mixture-of-Experts with Multiple Attempts
####################################
# Hyperparameters
num_tasks = 5
nclasses = len(np.unique(train_target))
classes_per_task = math.ceil(nclasses / num_tasks)
batch_size = 64
lr = 1e-3  # Constant learning rate of 0.001
num_epochs = 30
buffer_size = 1000
alignment_strength = 2.0
buffer_weight = 2.0
num_retries = 2  # Number of training attempts per task

# Run the improved incremental learning with multiple attempts per task
accuracies, final_model = incremental_learning_moe_with_retries(
    train_dataset, train_target, test_dataset, test_target,
    num_tasks, classes_per_task, batch_size, num_epochs, lr, device,
    buffer_size=buffer_size, alignment_strength=alignment_strength,
    buffer_weight=buffer_weight, num_retries=num_retries
)

print("\nIncremental Learning Accuracies per Task (best attempt):")
for i, acc in enumerate(accuracies):
    print(f"Task {i+1}: {acc:.2f}%")

# Final model is saved at './checkpoints/moe_model_task5_best.pt'
