import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from torch.utils.data import DataLoader, TensorDataset
from skimage.transform import resize
import torch.optim as optim
import torch.nn as nn
import time

iconfig = {
    "height": 128,  # 672
    "width": 256  # 1024
}

def transform_mask(rgb_mask):
    mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
    mask = np.where(mask == 145, mask, 0)
    if np.max(mask)!=0:
        mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask))
    return mask


class Dataset:
    def __init__(self, mode):
        self.mode = mode
        self.path = r"D:\dataset_vizja\road_segmentation\CAT\Brown_Field"

    def __process_dataset(self):
        if self.mode == "train":
            image_path = self.path + r"\Train\imgs"
            masks_path = self.path + r"\Train\masks"
        elif self.mode == "test":
            image_path = self.path + r"\Test\imgs"
            masks_path = self.path + r"\Test\masks"
        images = sorted(glob.glob(fr"{image_path}\*.png"))
        masks = sorted(glob.glob(fr"{masks_path}\*.png"))
        print(np.shape(images))
        print(np.shape(masks))
        X = np.zeros((len(images), iconfig["height"], iconfig["width"], 3), dtype=np.float32)
        y = np.zeros((len(images), iconfig["height"], iconfig["width"], 1), dtype=np.float32)

        for n, (img, mask) in enumerate(zip(images, masks)):
            img = cv2.imread(img)
            mask = cv2.imread(mask)

            img = cv2.resize(img, (iconfig["width"], iconfig["height"]))
            mask = cv2.resize(mask, (iconfig["width"], iconfig["height"]))
            X[n] = img
            y[n] = transform_mask(mask)[:, :, np.newaxis]
        print("Done")
        return X, y

    def data_loader(self, batch_size, shuffle=True):
        X, y = self.__process_dataset()

        X = np.transpose(X, axes=(0, 3, 1, 2))
        y = np.transpose(y, axes=(0, 3, 1, 2))

        x_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        print(np.shape(x_tensor))
        print(np.shape(y_tensor))

        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("cpu")


model = smp.UnetPlusPlus(in_channels=3, classes=1)
model = model.to(device)

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1)


def train(train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        print("train")
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, masks.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss


def val(val_loader, model=model, loss_fn=loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            print("val")
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, masks.float())
            val_loss += loss.item()
    return val_loss


train_loader = Dataset("train").data_loader(20)
test_loader = Dataset("test").data_loader(20)


epochs = 20
for epoch in range(epochs):
    etime = time.time()
    tl = train(train_loader, model, loss_fn, optimizer)
    vl = val(test_loader, model, loss_fn)
    print(f"{epoch+1}: train_loss = {tl}, val_loss = {vl}")
    print(f"{time.time() - etime}")

    if epoch == 0:
        best_val_loss = vl
        best_train_loss = tl
        torch.save(model.state_dict(), r"D:\envs_jup\PYTORCH\PYTORCH_models\best_tl.pth")
    else:
        if tl < best_train_loss:
            best_train_loss = tl
            torch.save(model.state_dict(), r"D:\envs_jup\PYTORCH\PYTORCH_models\best_tl.pth")
            print(f"{epoch + 1} train loss")
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), r"D:\envs_jup\PYTORCH\PYTORCH_models\best_vl.pth")
            print(f"{epoch + 1} val loss")


    if epoch +1 == epochs:
        torch.save(model.state_dict(), r"D:\envs_jup\PYTORCH\PYTORCH_models\last_model.pth")
