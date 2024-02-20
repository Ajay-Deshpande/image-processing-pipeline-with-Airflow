from tqdm import tqdm
from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from glob import glob
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import random_split
import torch
from airflow.decorators import task

class AlexNet(nn.Module):
    def __init__(self, num_classes = 2, dropout = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            
            nn.Dropout(p=dropout),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            
            nn.Dropout(p=dropout),
            
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def init_weight_fn(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1., 0)
        nn.init.constant_(m.bias.data, 0.)

@task(task_id = 'train_cv_model')
def train_model(transformed_images_path):
    print(transformed_images_path)
    dataset = ImageFolder(f'{transformed_images_path}/', transform = v2.Compose([v2.Grayscale(), v2.ToTensor()]))
    train_ds, val_ds = random_split(dataset, [.8, .2])
    train_dl = DataLoader(train_ds, batch_size = 64, shuffle = True, pin_memory = True)
    val_dl = DataLoader(val_ds, batch_size = 64, shuffle = True, pin_memory = True)

    model = AlexNet(len(dataset.classes))
    model = model.apply(init_weight_fn)
    optimizer = optim.Adam(model.parameters(), lr = 0.01, )
    loss_func = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, .9)

    for epoch in range(100):
        model.train()
        running_loss = 0
        metric = 0
        for data in tqdm(train_dl):
            optimizer.zero_grad()
            preds = model(data[0])
            metric += (preds.argmax(axis = 1) == data[1]).sum()
            loss = loss_func(preds, data[1])
            running_loss += loss
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}\nAverage training Loss = {running_loss / len(train_dl)}\nAccuracy = {metric/len(train_ds)}")
        if (epoch % 3 == 0):
            lr_scheduler.step()
        model.eval()
        running_loss = 0
        metric = 0
        for val_data in tqdm(val_dl):
            metric += (preds.argmax(axis = 1) == data[1]).sum()
            running_loss += loss_func(model(val_data[0]), val_data[1])
        print(f"Epoch: {epoch}\nAverage Validation Loss = {running_loss / len(val_dl)}\nAccuracy = {metric/len(val_ds)}")
    return True

if __name__ == "__main__":
    train_model('')