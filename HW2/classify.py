import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
import cv2 as cv
from PIL import Image
import os

# ---------- helper functions -----------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#        if batch % 100 == 0:
#            loss, current = loss.item(), batch * len(X)
#            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
def evaluation(classes,img_path):
    model.NeuralNetwork()
    arr = os.listdir("./")
    if "model.pth" in arr:
        model.load_state_dict(torch.load("./model/model.pth"))
        model.eval()
        name = "./"+img_path
        img = Image.open(name).convert('RGB')
        resize = transforms.Resize([32,32])
        img = resize(img)
        convert_tensor = transforms.ToTensor()
        tensor_ip = to_tensor(img)
        tensor_ip = tensor_ip.unsqueeze(0)
        with torch.no_grad():
            pred.model(tensor_ip)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
    else:
        print("Please train the model")

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = torchvision.datasets.CIFAR10(root='./data.cifar10')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
batchSize = 4

# ----------------- prepare training data -----------------------
transform_q = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = torchvision.datasets.CIFAR10(          # 32x32 images
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=transform_q,    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=batchSize)
# ----------------- prepare testing data -----------------------
test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=transform_q)
test_loader = Data.DataLoader(dataset=test_data, batch_size=batchSize)

# ----------------- build the model ------------------------
class Q2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,5,stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,5)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


rate_learning = 1e-3
model = Q2Net().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=rate_learning, weight_decay=1e-5)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--img_name', type=str, required=False)

args = parser.parse_args()

if args.mode.lower() == "train":
    
    for epoch in range(10):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_loader, model, loss_func, optimizer)
        test(test_loader, model, loss_func)
    
        
    torch.save(model.state_dict(), "./model/model.pth")

if args.mode.lower() == "test":
    evaluation(classes,args.img_name)
