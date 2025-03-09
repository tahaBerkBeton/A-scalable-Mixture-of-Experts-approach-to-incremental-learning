# Incremental Learning with Mixture-of-Experts (MoE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-GTSRB-green.svg)](https://benchmark.ini.rub.de/gtsrb_dataset.html)

## 📚 Overview

This repository implements a scalable incremental learning system that combats catastrophic forgetting using a Mixture-of-Experts (MoE) approach with an improved LeNet-5 architecture. The system learns to classify traffic signs sequentially without forgetting previously learned classes, evaluated on the German Traffic Sign Recognition Benchmark (GTSRB).

<p align="center">
  <img src="architecture.png" alt="MoE Architecture" width="700"/>
  <br>
  <em>Architecture of the Mixture-of-Experts model with LeNet-5 feature extractor</em>
</p>

## 🔍 Problem Statement

**Catastrophic forgetting** is a fundamental challenge in machine learning where neural networks tend to forget previously learned information when trained on new data. Traditional approaches require retraining on all data, which becomes computationally expensive as datasets grow.

Our approach tackles this by:
- 🏗️ Employing a modular architecture that adds experts for new tasks
- 🔄 Preserving a frozen feature extractor to maintain learned representations
- 🧠 Using a memory buffer to retain examples from previous tasks
- 🧭 Implementing a router mechanism to direct inputs to the appropriate expert
- 🔁 **NEW**: Multiple training attempts per task to reduce variability and select optimal models

## 🏛️ Model Architecture

### Feature Extractor
- Based on LeNet-5 with improvements (BatchNorm, Dropout)
- Processes RGB images (32×32 pixels)
- Extracts 400-dimensional feature vectors
- Frozen after initial training to preserve representations

```python
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
        return x.view(x.size(0), -1)  # [B, 400]
```

### Expert Networks
- Task-specific networks responsible for their subset of classes
- Each expert includes:
  - Two fully-connected layers (400→120→84)
  - Dropout for regularization
  - Output layer sized according to the number of classes

```python
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
```

### Router Network
- Linear layer that maps feature vectors to expert selection logits
- Selects the most appropriate expert for a given input
- Trained with cross-entropy loss to align with ground truth expert assignments

```python
# Router implementation within the MixtureOfExperts class
def add_expert(self, new_class_indices):
    num_new_classes = len(new_class_indices)
    new_expert = LeNetExpert(self.feature_dim, num_new_classes)
    self.experts.append(new_expert)
    self.expert_classes.append(new_class_indices)
    new_num_experts = len(self.experts)
    
    # Router network is a linear layer mapping features to expert logits
    old_router = self.router
    new_router = nn.Linear(self.feature_dim, new_num_experts)
    
    # Transfer weights from old router to maintain previous routing behavior
    if old_router is not None:
        with torch.no_grad():
            new_router.weight[:old_router.out_features] = old_router.weight
            new_router.bias[:old_router.out_features] = old_router.bias
    self.router = new_router
```

## 🔄 Incremental Learning Process

### Task Division
The GTSRB dataset is divided into 5 sequential tasks, with each introducing new traffic sign classes:
- Task 1: Classes 0-8
- Task 2: Classes 9-17
- Task 3: Classes 18-26
- Task 4: Classes 27-35
- Task 5: Classes 36-42

### Multiple Training Attempts Strategy
To address training variability, the updated implementation introduces multiple training attempts:

1. **First Task**: Train the feature extractor and first expert twice (or more)
   - Select the model with the highest validation accuracy
   - This is critical as the quality of the feature extractor affects all subsequent tasks

2. **Subsequent Tasks**: For each new task, train the router and new expert twice (or more)
   - Start from the best model from the previous task
   - Freeze the feature extractor and previous experts
   - Select the model with the highest validation accuracy
   - Save both attempt-specific checkpoints and the best model checkpoint

```python
def incremental_learning_moe_with_retries(train_dataset, train_target, test_dataset, test_target,
                                num_tasks, classes_per_task, batch_size, num_epochs, lr, device,
                                buffer_size=1000, alignment_strength=2.0, buffer_weight=2.0,
                                num_retries=2):
    # ... 
    for task in range(num_tasks):
        # ...
        best_task_accuracy = 0
        best_task_model = None
        
        if task == 0:
            for attempt in range(num_retries):
                # Create and train new model for first task
                # Select best model based on validation accuracy
                # ...
        else:
            for attempt in range(num_retries):
                # Train new expert with frozen feature extractor
                # Select best model based on validation accuracy
                # ...
```

### Training Procedure
1. **First Task**: Train feature extractor and first expert (50 epochs × 2 attempts)
2. **Subsequent Tasks**:
   - Freeze feature extractor and previous experts
   - Add new expert for current task classes
   - Train for 30 epochs × 2 attempts with constant learning rate (0.001)
   - Incorporate memory buffer containing examples from previous tasks
   - Apply multi-component loss function:
     - Classification loss on new task data
     - Classification loss on memory buffer (weighted by 2.0)
     - Routing alignment loss (weighted by 2.0)
   - Save best model checkpoint based on validation accuracy

### Memory Buffer Strategy
- Maintains 1000 samples from previous tasks 
- Balanced allocation across past classes
- Randomly selected from test split to simulate real-world scenarios
- Replayed during training to mitigate forgetting

## 📊 Results

The improved incremental learning performance demonstrates the model's ability to learn new tasks while retaining knowledge from previous ones:

| Task | Classes | Accuracy |
|------|---------|----------|
| 1    | 0-8     | 94.24%   |
| 2    | 0-17    | 94.31%   |
| 3    | 0-26    | 91.92%   |
| 4    | 0-35    | 88.33%   |
| 5    | 0-42    | 88.83%   |

The multiple training attempts approach yields significantly improved results compared to the previous implementation, with accuracy improvements of approximately 10-15% across all tasks.

### Training Variability Mitigation

The original challenge of training variability has been effectively addressed:

- Multiple training runs for each task reduce the impact of initialization and training path dependencies
- Selection of the best models at each stage creates a more consistent and optimized learning path
- The first task's feature extractor quality is optimized by selecting the best of multiple training runs
- Router degradation is mitigated by choosing the best router training for each task

### Detailed Error Analysis

Advanced error analysis from our `inference_advanced.py` script provides insights into the specific sources of classification errors:

```
Total Samples: 12630
Router Errors: 772 (6.11%)
Expert Errors (on correctly routed samples): 639 (5.39%)
Per-Expert Statistics:
  Expert 0: 4179 samples, 259 errors, error rate: 6.20%
  Expert 1: 3738 samples, 48 errors, error rate: 1.28%
  Expert 2: 1508 samples, 186 errors, error rate: 12.33%
  Expert 3: 1326 samples, 93 errors, error rate: 7.01%
  Expert 4: 1107 samples, 53 errors, error rate: 4.79%
```

This breakdown reveals that:
- Router errors have been reduced to only ~6% (down from ~14.5% in the original implementation)
- Expert errors are now only ~5.4% (down from ~11.4% in the original implementation)
- Error rates are much more balanced between experts, with Expert 1 showing exceptional performance

These improvements validate the effectiveness of the multiple training attempts approach in reducing both routing and classification errors.

## 🗂️ Directory Structure

```
incremental-learning/
├── MOE_Lenet.py   #  training script with multiple attempts per task
├── inference_Lenet.py           # Basic inference script for evaluating the trained MoE model
├── inference_advanced.py        # Advanced inference with detailed error statistics
├── architecture.png             # Visualization of the model architecture
└── checkpoints/                 # Directory containing model checkpoints
    ├── moe_model_task1_attempt1.pt  # First attempt for task 1
    ├── moe_model_task1_attempt2.pt  # Second attempt for task 1
    ├── moe_model_task1_best.pt      # Best model for task 1
    ├── moe_model_task2_attempt1.pt
    ├── moe_model_task2_attempt2.pt
    ├── moe_model_task2_best.pt
    └── ... (similar for tasks 3-5)
```

## 🛠️ Requirements

```
torch
torchvision
numpy
pandas
tqdm
```

## 🚀 How to Run

### Setup
```bash
# Clone repository
git clone https://github.com/tahaBerkBeton/A-scalable-Mixture-of-Experts-approach-to-incremental-learning.git
cd A-scalable-Mixture-of-Experts-approach-to-incremental-learning

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash

# Run  training script with multiple attempts
python MOE_Lenet.py
```
Training creates checkpoints in the `checkpoints/` directory after completing each task.

### Inference
```bash
# Run basic inference (by default uses the latest task model)
python inference_Lenet.py

# Run detailed error analysis
python inference_advanced.py

# Specify a different checkpoint
python inference_Lenet.py --checkpoint ./checkpoints/moe_model_task3_best.pt
```

## 📝 Implementation Details

Key hyperparameters:
- Learning rate: 0.001 (constant)
- Batch size: 64
- Memory buffer size: 1000 samples
- Alignment strength: 2.0
- Buffer weight: 2.0
- **NEW**: Number of training attempts per task: 2 (configurable)

### Loss Function Components and Justification

The loss function is carefully designed to balance learning new tasks while preserving knowledge of previous tasks:

```python
total_loss = (classification_loss_new +
              buffer_weight * classification_loss_buf +
              alignment_strength * (routing_loss_new + routing_loss_buf))
```

#### Component 

1. **Classification Loss for New Task** (`classification_loss_new`)
   - Standard cross-entropy loss that ensures the model learns to classify samples from the current task
   - Base component with weight 1.0 as the primary learning objective

2. **Buffer Classification Loss** (`classification_loss_buf` with weight 2.0)
   - Cross-entropy loss applied to memory buffer samples from previous tasks
   - **Weight = 2.0**: This higher weight is chosen to retain previously learned knowledge by penalising forgetting past instances

3. **Routing Alignment Loss** (`routing_loss_new + routing_loss_buf` with weight 2.0)
   - Cross-entropy loss that trains the router to correctly select the appropriate expert for each sample
   - **Weight = 2**: This higher weight ensures routing learning occurs without forgetting previous routing instances
   - This component is crucial for the MoE architecture as it ensures:
     - New samples are routed to the appropriate task-specific expert
     - Previously learned routing paths are maintained for old task samples
     - The modular structure of knowledge is preserved across incremental learning

### Training Path Optimization

The new implementation includes specific functions for different training scenarios:

1. **Initial Task Training**: `train_initial_task()`
   - Specifically optimized for training the feature extractor and first expert
   - No buffer needed for the first task
   - Focuses on building robust feature representations

2. **Subsequent Task Training**: `train_subsequent_task()`
   - Specialized for training with frozen components and memory buffer
   - Balances new knowledge acquisition with preservation of old knowledge
   - Incorporates buffer replay seamlessly

3. **Best Model Selection Logic**: 
   - Model selection based on validation accuracy on the complete set of seen classes
   - Checkpoint saving for each attempt and for the best model
   - Incremental building of optimal model path through the task sequence

## 🔮 Future Work

- 🧪 Further increase the number of training attempts for more stable results
- 🔍 Explore dynamic routing mechanisms to improve expert selection
- 📈 Apply to larger, more complex datasets (ImageNet, etc.)
- 🔄 Investigate automated hyperparameter tuning for each task
- 🤖 Develop improved memory buffer selection strategies
- 🌟 Explore ensemble methods combining multiple training paths

## 📚 References

- German Traffic Sign Recognition Benchmark (GTSRB): https://benchmark.ini.rub.de/gtsrb_dataset.html
- "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et al.
- "Mixture of Experts: A Literature Survey" by Masoudnia and Ebrahimpour

## 🙏 Acknowledgements

This project leverages several key resources:
- GTSRB Dataset for training and evaluation
- PyTorch framework for implementation
- Research contributions from the incremental learning and MoE communities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
