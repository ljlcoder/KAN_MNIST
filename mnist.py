#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :mnist.py
# @Time      :2024/10/23 10:50
# @Author    :Jasonljl


import torch
import torch.nn as nn
import torchvision.datasets
from torchvision.datasets import mnist,SBDataset
# from torchvision.models import
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append(r"H:\python\KAN\KAN_VAE_MNIST-main\KAN_VAE_MNIST-main\efficient-kan-master\src")
from efficient_kan.kan import KAN

transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5))
])

trainsets=torchvision.datasets.MNIST(
    root="./mnist",download=False,train=True,transform=transforms
)
testsets=torchvision.datasets.MNIST(
    root="./mnist",download=False,train=False,transform=transforms
)

trainloaders=DataLoader(dataset=trainsets,batch_size=64,shuffle=True)
testloaders=DataLoader(dataset=testsets,batch_size=64,shuffle=False)

model=KAN([28*28,64,10])
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
optimizer=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)

criterion=nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    with tqdm(trainloaders) as phar:
        for j,(images,labels) in enumerate(phar):
            images=images.view(-1,28*28).to(device)
            optimizer.zero_grad()
            output=model(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
            acc=(output.argmax(dim=1)==labels.to(device)).float().mean()
            phar.set_postfix(loss=loss.item(),acc=acc.item(),lr=optimizer.param_groups[0]["lr"])
    model.eval()
    val_loss=0
    acc=0
    acc_list=[]
    val_list=[]
    with torch.no_grad():
        for images,labels in testloaders:
            images=images.view(-1,28*28).to(device)
            output=model(images)
            val_loss+=criterion(output,labels.to(device)).item()
            acc+=(
                (output.argmax(dim=1)==labels.to(device)).float().mean().item()
            )
    val_loss/=len(testloaders)
    acc/=len(testloaders)
    acc_list.append(acc)
    val_list.append(val_loss)

    print(f"kan_epoch:{epoch + 1},val loss:{val_loss},val_acc:{acc}\n")
    with open("result_log.txt", "a+") as f:
        res = f"kan_epoch:{epoch + 1},val loss:{val_loss},val_acc:{acc}\n"
        f.write(res)
print("done!!!")

epochs=range(1,11)
plt.plot(epochs,acc_list,label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.show()
# print(f"epoch:{epoch+1},val loss:{val_loss},val_acc:{acc}")