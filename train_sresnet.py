
import torch
import os

from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import Div2kDataset
from utils import ImgTransformer
from models import SResNet

SEED = 42

# model params
SCALE = 2
LARGE_KERNEL = 9
SMALL_KERNEL = 3
N_CHANNELS = 64
N_RES_BLOCKS = 10
NORM = False

# training params
TRAIN_EPOCHS = 1
BATCH_SIZE = 4
CROP_SIZE = 64
DATA_DIR = './DIV2K'
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PT_SAVED = 'sresnet.pt'
CHECKPOINT = PT_SAVED if os.path.exists(PT_SAVED) else None

def train_epoch(loader, model, criterion, optimizer):
    train_cum_loss = 0.
    
    for [lr, hr] in tqdm(loader, total=len(loader)):
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        # zero grad and forward batch trough model
        optimizer.zero_grad()
        sr = model(lr)
        
        # calculate and backprop loss 
        loss = criterion(sr, hr)
        loss.backward()

        # adjust optimizer weights
        optimizer.step()

        train_cum_loss += loss.item() * lr.shape[0]
        
        del lr, hr, sr
        
        # tmp return after one batch
    return train_cum_loss / len(loader)

if __name__ == '__main__':
    
    torch.manual_seed(SEED) 
    torch.cuda.manual_seed(SEED)

    # setup train dataloader
    dataset = Div2kDataset(data_dir=DATA_DIR, transform=ImgTransformer('[0, 1]', '[-1, 1]', crop=CROP_SIZE, scale=SCALE))
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    # init model and optimizer values
    if not CHECKPOINT:
        model = SResNet(3, 3, N_CHANNELS, SMALL_KERNEL, LARGE_KERNEL, N_RES_BLOCKS, SCALE, NORM)
        optimizer = Adam(model.parameters(), lr=LR)
        model_epoch = 0
    else:
        checkpoint = torch.load(CHECKPOINT)
        model, optimizer, model_epoch = checkpoint['model'], checkpoint['optimizer'], checkpoint['epoch']
        print(f'[EPOCH {model_epoch}] loaded checkpoint')


    model = model.to(DEVICE)
    criterion = nn.MSELoss().to(DEVICE)

    # train model for desired number of train epochs
    for epoch in range(TRAIN_EPOCHS):

        model.train(True)
        avg_loss = train_epoch(train_loader, model, criterion, optimizer)
        model.train(False)

        print(f'[EPOCH {model_epoch}] avg loss: {avg_loss:.4f}')

        model_epoch += 1
        torch.save({'model' : model, 'optimizer' : optimizer, 'epoch' : model_epoch}, PT_SAVED)