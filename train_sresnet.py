""" Sresnet train script with model and training params configuration """

import os
import torch

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
N_RES_BLOCKS = 16

NORM = False
VALIDATE = True

# training params
TRAIN_EPOCHS = 5 
BATCH_SIZE = 16
CROP_SIZE = 96
DATA_DIR = "./DIV2K"
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MSE_LOSS_CRITERION = nn.MSELoss().to(DEVICE)

# training checkpoint params
PT_SAVED = "sresnet.pt"
CHECKPOINT = PT_SAVED if os.path.exists(PT_SAVED) else None


def train_epoch(loader, model, optimizer):
    """ Train one epoch of given model
    
    Args:
    loader -- data loader class
    model -- model to train with training data
    optimizer -- model optimizer class
    """
    train_cum_loss = 0.

    for lr, hr in tqdm(loader, total=len(loader)):
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        # zero grad and forward batch trough model
        optimizer.zero_grad()
        sr = model(lr)

        # calculate and backprop loss
        loss = MSE_LOSS_CRITERION(sr, hr)
        loss.backward()

        # adjust optimizer weights
        optimizer.step()

        train_cum_loss += loss.item() * lr.shape[0]

        del lr, hr, sr

    return train_cum_loss / len(loader)


def validate_epoch(loader, model):
    """ Run one epoch and calculate validation loss

    Args:
    loader -- data loader class with validation data
    model -- model class used for validation
    """
    valid_cum_loss = 0.

    for lr, hr in tqdm(loader, total=len(loader)):
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        with torch.no_grad():
            sr = model(lr)
            loss = MSE_LOSS_CRITERION(sr, hr)

            valid_cum_loss += loss.item() * lr.shape[0]

            del lr, hr, sr

    return valid_cum_loss / len(loader)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # setup data loaders
    dataset = Div2kDataset(data_dir=DATA_DIR, transformer=ImgTransformer("[0, 1]", "[-1, 1]", crop=CROP_SIZE, scale=SCALE))
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # init new model and optimizer
    sresnet = SResNet(3, 3, N_CHANNELS, SMALL_KERNEL, LARGE_KERNEL, N_RES_BLOCKS, SCALE, NORM)
    sresnet_optimizer = Adam(sresnet.parameters(), lr=LR)
    model_epoch = 0
    tloss, vloss = [], []

    # load model and optimizer saved state if exists
    if CHECKPOINT:
        state = torch.load(CHECKPOINT)
        sresnet, sresnet_optimizer, model_epoch, tloss, vloss = (
            state["model"],
            state["optimizer"],
            state["epoch"],
            state["tloss"],
            state["vloss"]
        )
        print(f"[EPOCH {model_epoch}] loaded checkpoint")

    sresnet = sresnet.to(DEVICE)

    # train model for desired number of train epochs
    for epoch in range(TRAIN_EPOCHS):

        # run train epoch
        print(f"[EPOCH {model_epoch}] training epoch")
        sresnet.train(True)
        avg_train_loss = train_epoch(train_loader, sresnet, sresnet_optimizer)
        sresnet.train(False)

        print(f"[EPOCH {model_epoch}] train loss: {avg_train_loss:.4f}")
        tloss.append(avg_train_loss)

        if VALIDATE:
            # run validation epoch without model learning
            print(f"[EPOCH {model_epoch}] validating epoch")
            avg_valid_loss = validate_epoch(valid_loader, sresnet)

            print(f"[EPOCH {model_epoch}] valid loss: {avg_valid_loss:.4f}")
            vloss.append(avg_valid_loss)
        else:
            vloss.append(1000.)

        # save model and training status
        model_epoch += 1
        torch.save({
            "model": sresnet, 
            "optimizer": sresnet_optimizer, 
            "epoch": model_epoch,
            "tloss": tloss,
            "vloss": vloss
        }, PT_SAVED)
