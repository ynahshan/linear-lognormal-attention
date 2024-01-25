from __future__ import print_function
import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import zipfile

from lln.lln_transformer import LinearTransformer
from vit_pytorch.efficient import ViT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", type=str, default="softmax",
                        help="options: softmax, lln, linformer")
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    return args

args = get_args()
print(args)

# Training settings
batch_size = 64
epochs = args.epochs
lr = 3e-5
gamma = 0.7
seed = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda'

os.makedirs('data', exist_ok=True)

train_dir = 'data/train'
test_dir = 'data/test'


def check_extract_dataset():
    if not os.path.exists(train_dir):
        if not os.path.exists('train.zip'):
            print("Train dataset {} not found. You can download it from here: {}".format('train.zip', 'https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition'))
        with zipfile.ZipFile('train.zip') as train_zip:
            train_zip.extractall('data')

    if not os.path.exists(test_dir):
        if not os.path.exists('test.zip'):
            print("Train dataset {} not found. You can download it from here: {}".format('test.zip', 'https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition'))
        with zipfile.ZipFile('test.zip') as test_zip:
            test_zip.extractall('data')


check_extract_dataset()

train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

labels = [path.split('/')[-1].split('.')[0] for path in train_list]

train_list, valid_list = train_test_split(train_list,
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))


def build_model(emb_dim=128, patch_size=8, image_size=224):
    seq_len = (image_size // patch_size) ** 2 + 1  # pxp patches + 1 cls-token
    efficient_transformer = LinearTransformer(
        dim=emb_dim,
        seq_len=seq_len,
        depth=12,
        heads=8,
        k=64,
        attn_type=args.attn
    )

    model = ViT(
        dim=emb_dim,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=2,
        transformer=efficient_transformer,
        channels=3,
    ).half().to(device)

    return model


model = build_model(patch_size=7)
print(model)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-4)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

grad_max = []
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for i, (data, label) in enumerate(tqdm(train_loader)):
        data = data.half().to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in tqdm(valid_loader):
            data = data.half().to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
