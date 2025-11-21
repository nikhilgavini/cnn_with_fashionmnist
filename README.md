# FashionMNIST CNN Classifier (PyTorch)

This project implements a convolutional neural network (CNN) in PyTorch to classify images from the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

It started as a course exercise, but the code has been refactored and documented to reflect how I would structure a small, production-adjacent experiment:
- Proper device handling (CPU/GPU)
- Data augmentation and normalization
- Clear separation of training vs evaluation
- Basic experiment tracking (losses and accuracy over epochs)

---

## Dataset

**FashionMNIST** is a drop‐in replacement for the classic MNIST digits dataset:
- 60,000 training images, 10,000 test images
- 28×28 grayscale images
- 10 clothing classes:

0. T-shirt/top  
1. Trouser  
2. Pullover  
3. Dress  
4. Coat  
5. Sandal  
6. Shirt  
7. Sneaker  
8. Bag  
9. Ankle boot  

In this project, the dataset is loaded via `torchvision.datasets.FashionMNIST` with a custom transform pipeline.

---

## Data Pipeline

Data loading is handled using `torchvision` and `DataLoader`.

```python
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(root="data/", train=True, download=True, transform=transform)
test_data  = datasets.FashionMNIST(root="data/", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)
```

---

## Model Architecture

The model architecture is a shallow 2-layer CNN implemented as a custom `nn.Module`

```python
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 5 * 5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
```
Design choices:
- 2 convolutional layers with 3×3 kernels and max pooling for spatial downsampling.
- Shallow architecture by design to balance model capacity and overfitting risk on a relatively small 28×28 grayscale dataset.
- Final layer outputs 10 logits corresponding to the 10 FashionMNIST classes.
- log_softmax in the return because of increased numerical stability & simplification of cross-entropy loss function later on.
  
---

## Training Setup

Training is done with:
- Loss: `CrossEntropyLoss` (multi-class classification)
- Optimizer: `Adam` with a learning rate of `0.001`
- Epochs: 5 (by default)

Device Handling:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionalNetwork().to(device)
```

Core Training Loop (Simplified):
```python
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0
    trn_corr = 0

    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        trn_corr += (y_pred.argmax(dim=1) == y_train).sum()

    # Evaluation phase
    model.eval()
    epoch_test_loss = 0.0
    tst_corr = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_val = model(X_test)
            loss = criterion(y_val, y_test)
            epoch_test_loss += loss.item()
            tst_corr += (y_val.argmax(dim=1) == y_test).sum()
```

---

## Results
With:
- 2 conv layers + 3 fully connected layers
- 5 training epochs
- Light augmentation and normalization

The model achieves roughly ~86% test accuracy

A plot of training and test accuracy over the training epochs is shown:
<img width="543" height="435" alt="image" src="https://github.com/user-attachments/assets/daeac886-ce5a-4ecf-8e65-d0fd6aa9c956" />

This is a reasonable baseline for a small CNN on FashionMNIST. 

Public benchmarks and deeper architectures can push this into the 90–92% range.

---

## Next Steps/Possible Improvements
If I were extending this project, I would:
- Add more robust data augmentation (random crops, color jitter if using other datasets).
- Train longer (15–20 epochs) with a learning rate schedule.
- Introduce dropout before fully connected layers to reduce overfitting.
- Experiment with deeper architectures (more filters, additional conv blocks).
- Log metrics with a tracking tool (e.g., TensorBoard, MLflow) instead of just lists in memory.

---

## Why This Project Exists
This project is part of my path toward Machine Learning / AI Engineer roles.
It demonstrates:
- Competent use of PyTorch for CNNs
- Understanding of data pipelines, augmentation, normalization
- Correct device handling (CPU/GPU)
- Separation of training vs evaluation logic
- Baseline experiment design and interpretation
