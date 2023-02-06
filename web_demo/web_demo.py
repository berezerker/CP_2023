import os
import sys
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as tt

def metrics(preds, labels):
    acc = accuracy(preds, labels)
    prec = precision(preds, labels)
    rec = recall(preds, labels)
    return acc, prec, rec
class BaseModel(nn.Module):
    def training_step(self,batch):
        loss_func = torch.nn.BCELoss()
        images,labels = batch
        out = self(images).squeeze(1)
        labels = labels.to(torch.float32)
        loss = loss_func(torch.sigmoid(out),labels)
        acc, prec, rec = metrics(torch.sigmoid(out), labels)
        return loss, acc, prec, rec
    
    def validation_step(self,batch):
        images, labels = batch
        with torch.inference_mode():
            loss_func = torch.nn.BCELoss()
            out = self(images).squeeze(1)
            labels = labels.to(torch.float32)
            loss = loss_func(torch.sigmoid(out),labels)
        acc, rec, prec = metrics(torch.sigmoid(out), labels)
        return {"val_loss":loss.detach(),"val_acc":acc, "val_prec":prec, "val_rec":rec}
    
    def validation_epoch_end(self,outputs):
        batch_losses = [loss["val_loss"] for loss in outputs]
        loss = torch.stack(batch_losses).mean()
        batch_accuracy = [accuracy["val_acc"] for accuracy in outputs]
        acc = torch.stack(batch_accuracy).mean()
        batch_precision = [precision["val_prec"] for precision in outputs]
        prec = torch.stack(batch_precision).mean()
        batch_recall = [recall["val_rec"] for recall in outputs]
        rec = torch.stack(batch_recall).mean()
        f1 = 2 * (prec * rec) / (prec + rec)
        return {"val_loss":loss.item(),"val_acc":acc.item(), "val_prec": prec.item(), "val_rec": rec.item(), "val_f1": f1.item()}
    
    def epoch_end(self, epoch, result):
        print(f'Epoch [{epoch}], last_lr: {result["lrs"][-1]:.5f}, train_loss: {result["train_loss"]:.4f}, val_loss: {result["val_loss"]:.4f}, train_acc: {result["train_acc"]:.4f}, val_acc: {result["val_acc"]:.4f}')

class CustomModelConv(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 12, 3, 2)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Sequential(
          nn.Linear(512, 256),
          nn.Dropout(0.2),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.Dropout(0.3),
          nn.ReLU(),
          nn.Linear(64, num_classes)
        )
        
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
    def unfreeze_for_train(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.conv_first.parameters():
            param.requires_grad = True
        for param in self.model.conv1.parameters():
            param.requires_grad = True
    def forward(self, xb):
        out = self.conv_first(xb)
        out = self.model(out)
        return out


sys.path.append("../")

root_dir = os.path.dirname(__file__)
model = CustomModelConv(1)
model.load_state_dict(torch.load('../models/best_res.pt'))
model.eval()

def main():
    st.header("Detection of blurred image demo")
    st.write("Upload your image for detecting:")

    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, "your image")
        to_tensor = tt.ToTensor()
        img = to_tensor(img).unsqueeze(0)
        out = model(img)
        out = torch.sigmoid(out)
        if out > 0.5:
            st.text(f'The image is blurry with confidence of {out.item():.3f}')
        else:
            st.text(f'The image is not blurry with confidence of {1 - out.item():.3f}')


if __name__ == "__main__":
    main()
