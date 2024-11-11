import torch
import torch.utils.data
import scipy.io
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms
import os
import sys
from tensorboardX import SummaryWriter
from datetime import datetime
import skimage.io
import scipy.io as scio
import glob
import torch.nn as nn
import torch.nn.init as init
from sklearn.model_selection import GroupKFold,KFold
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import GradScaler,autocast
import torchvision.transforms as transforms
import albumentations as A
pd.set_option("display.max_columns",40000)
pd.set_option("display.width",2000)
import cv2
from models import Intensity_Branch_Module,Depth_modual



import gc
current_time = datetime.now().strftime("%D")
month = current_time[:2]
day = current_time[3:5]
print(month,day)

path = month+ "_" + day
if os.path.exists("../PreTrain/"+path):
    print(" the folder exists")
else:
    print("creat dir")
    os.mkdir("../PreTrain/"+path)

writer = SummaryWriter("log")
path = "../PreTrain/"+path
DEVICE = "cuda"
class CFG:
   # data_path = "/media/xlh/cai/dataset/processedlowsbr/"
    data_path = "/public/home/l164/New_dataset2024/"
  #  data_path = "/public/home/l164/New_dataset/"
    folds = 5
    num_workers = 3
    epochs = 50
    batch_size = 1
    one_cycle = True
    one_cycle_pct_start = 0.1
    one_cycle_max_lr = 0.00005
    tv = 2
    POISSION_Loss = True
    KL_Loss = False
    DECONV = False


mat = "blur.mat"
if not CFG.KL_Loss:
    model = Intensity_Branch_Module(1,32)
else:
    model = Depth_modual(1)

spad_data = glob.glob(CFG.data_path + r"**/" + "spad*_3Drawinput_*.npy")
mat_data = glob.glob(CFG.data_path + r"**/" + "spad*.mat")
print(len(spad_data), "ssssssss")
print(len(mat_data), "mat")
rows = ["data"]
# for i in range(len(spad_data)):
df = pd.DataFrame([spad_data])
# print(df.shape)
df = df.transpose()
print(df.shape)
print(df.iloc[0, 0])
df["X"] = df[0]
kfold = KFold(n_splits=CFG.folds)
for train_idx, val_idx in kfold.split(df):
    train_data = df.loc[train_idx, "X"].values
    val_data = df.loc[val_idx, "X"].values

train_aug_list = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)
]

res = 256
class SpadDataset(torch.utils.data.Dataset):
    def __init__(self, X, res=256, mat_data=None, transforms=None):
        self.spad_files = X
        self.output_size = res
        self.mat_data = mat_data
        self.transforms = transforms

    def __len__(self):
        return len(self.spad_files)

    def tryitem(self, idx):
        # simulated spad measurements
        if self.transforms:
            spad = np.load(self.spad_files[idx]).astype(np.float32)
            rates = np.load(self.spad_files[idx].replace("3Drawinput", "3Drawtarget")).astype(np.float32)

            rates = rates[:, :, :]
        #    print(spad.shape,rates.shape)
            # print(spad[6:-6,15,15],"spad")
            # print(rates[:,15,15])
            # print(np.max(spad[6:-6,:,:]),np.max(rates),"ceshiceshiceshi")
            sample = {'spad': spad, 'rates': rates}
        else:
            spad = np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(
                self.mat_data[idx])['spad'])).astype(
                np.float32).reshape([1, res, res, -1])  # 1,64,64,1024
            spad = np.transpose(spad, (0, 3, 2, 1))
            #print(np.max(spad), "max spad")
            #      spad = spad / np.max(spad) has been normlized
            # normalized pulse as GT histogram
            rates = np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(
                self.mat_data[idx])['rates_sig'])).astype(
                np.float32).reshape([1, res, res, -1])  # 1,64,64,1024
            rates = np.transpose(rates, (0, 3, 2, 1))



            # rates = np.asarray(scipy.io.loadmat(
            #     self.mat_data[idx])['raw_rates']).astype(
            #     np.float32).reshape([1, res, res, -1])
            # rates = np.transpose(rates, (0, 3, 1, 2))
            bins = (np.asarray(scipy.io.loadmat(
                self.mat_data[idx])['bin']).astype(
                np.float32).reshape([res, res]) - 1)[None, :, :] / 1023
            rates_full = rates[0,:,:,:]
            spad_full = spad[0,:,:,:]
            rates = torch.from_numpy(rates)
            spad = torch.from_numpy(spad)
            bins = torch.from_numpy(bins)
            rates_full = torch.from_numpy(rates_full)
            spad_full = torch.from_numpy(spad_full)
            sample = {'spad': spad, 'rates': rates, 'bins': bins, "spad_full": spad_full, "rates_full": rates_full}
        return sample

    def __getitem__(self, idx):
        # print(idx,"idx")
        try:
            sample = self.tryitem(idx)
        except Exception as e:
            #    print(idx, e)
            idx = idx + 1
            sample = self.tryitem(idx)
        if self.transforms:
            spad = sample["spad"]
            rates = sample['rates']

            spad = torch.from_numpy(spad)
            spad = torch.unsqueeze(spad, 0)
            sample["spad"] = spad
            rates = torch.from_numpy(rates)
            rates = torch.unsqueeze(rates, 0)
            sample["rates"] = rates

        return sample

if not CFG.KL_Loss:
    train_data = SpadDataset(train_data, res, mat_data, transforms=A.Compose(train_aug_list))
    val_data = SpadDataset(val_data, res, mat_data, transforms=None)
    train_loader = DataLoader(train_data, batch_size=CFG.batch_size,
                              shuffle=True, num_workers=CFG.num_workers,
                              pin_memory=True)
    eval_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=3)
else:
    train_data = SpadDataset(train_data, res, mat_data, transforms=None)
    val_data = SpadDataset(val_data, res, mat_data, transforms=None)
    train_loader = DataLoader(train_data, batch_size=CFG.batch_size,
                              shuffle=True, num_workers=CFG.num_workers,
                              pin_memory=True)
    eval_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=3)

def criterion_L2(est, gt):
    criterion = nn.MSELoss()
    # est should have grad
    return torch.sqrt(criterion(est, gt))

def criterion_TV(inpt):
    return torch.mean(torch.abs(inpt[:,:, :, :, :-1] - inpt[:,:, :, :, 1:])) + \
           torch.mean(torch.abs(inpt[:,:, :, :-1, :] - inpt[:,:, :, 1:, :]))

def evaluate_model(model,eval_loader):
    torch.manual_seed(42)
    model = model.to(DEVICE)

    with torch.no_grad():
        model.eval()
        losses = []
        with tqdm(eval_loader,desc="Eval",mininterval=120) as progress:
            for i,sample in enumerate(progress):
           #     print(i,"sssssssssssssssssssss")
                if i >= 7:
                    break
                spad_full = sample["spad_full"]
                rates_full = sample["rates_full"]
                depth = sample["bins"]
                pred_rates=torch.zeros([1024,res,res])
                rates_pred = torch.zeros([1,1024,res,res])
                with autocast(enabled=True):
                    for i in range(0,1024-128,64):
                        spad = spad_full[:,i:i+128,:,:]
                        spad = spad.reshape([1,1,128,res,res])
                        if not CFG.KL_Loss:
                            y_pred,_ = model.forward(spad.to(DEVICE))
                        else:
                            y_pred = model.forward(spad.to(DEVICE))
                        y_pred = y_pred.reshape([1,-1,res,res])
                        rates_pred[:,i+12:i+12+104,:,:] = y_pred[:,12:-12,:,:]
                y_depth = torch.argmax(rates_full[:,12:-12,:,:],dim=1,keepdim=True)
                y_depth = y_depth / 1000
                y = torch.argmax(rates_pred[:,12:-12,:,:],dim=1,keepdim=True)
                y = y / 1000
                rmse = criterion_L2(y,y_depth)
                progress.set_postfix(rmse=rmse)
            losses.append(rmse.cpu().numpy())
        return np.mean(losses)



import copy
def train_model(train_loader,eval_loader):
    torch.manual_seed(42)
  #  print(model)
    optim = torch.optim.Adam(model.parameters(),lr=CFG.one_cycle_max_lr)
  #  scheduler = torch.optim.lr_scheduler.OneCycleLR(optim,max_lr=CFG.one_cycle_max_lr,epochs=CFG.epochs,steps_per_epoch=len(train_loader),pct_start=CFG.one_cycle_pct_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=2000,eta_min=0.0000005)
    scaler = GradScaler()
    best_eval_score = 100
    length = len(train_loader)
    lsmx = torch.nn.LogSoftmax(dim=2)
    criterion_KL = nn.KLDivLoss()
  #  mse = nn.MSELoss()
 #   cross = nn.CrossEntropyLoss()
    criterion = nn.PoissonNLLLoss(log_input=False, eps=1e-8, reduction="mean")
   # criterion = nn.MSELoss()
    for epoch in tqdm(range(CFG.epochs),desc="Epoch"):
        model.to(DEVICE).type(torch.cuda.FloatTensor)
        model.train()
       # rate_model = rate_model.to(DEVICE).type(torch.cuda.FloatTensor)
      #  rate_model.eval()
       # evaluate_model(model,eval_loader)
        with tqdm(train_loader,desc="Train",miniters=200) as train_progress:
            for batch_idx,sample in enumerate(train_progress):
                spad = sample["spad"].to(DEVICE).type(torch.cuda.FloatTensor)
                print(spad.shape,"ss")
                rates = sample["rates"].to(DEVICE).type(torch.cuda.FloatTensor)
                optim.zero_grad()
                with autocast():
                    if CFG.KL_Loss:
                        y = model.forward(spad.to(DEVICE))
                        loss_tv = criterion_TV(y)
                        y_lsmx = lsmx(y)
                        loss_kl = criterion_KL(y_lsmx, rates.to(DEVICE))
                        loss = loss_kl  + CFG.tv * loss_tv
                    else:
                        if CFG.POISSION_Loss:
                            y,_ = model.forward(spad)
                        elif CFG.DECONV:
                            _,y = model.forward(spad)
                        poissionloss =  criterion(y,rates.to(DEVICE))
                        loss_tv = criterion_TV(y)
                        loss = poissionloss  + CFG.tv * loss_tv #+ mseloss#+ 0.5 * l1_loss# + CFG.tv * L2_loss# + 1 * l1_loss #+ CFG.tv * loss_tv #+  #+ mseloss
                    writer.add_scalar("Loss/train", loss, epoch * length + batch_idx)
                    lr = optim.param_groups[0]["lr"]
                    train_progress.set_postfix(ploss=loss,lr=lr,tv=loss_tv)#l1=l1_loss)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scheduler.step()
                scaler.update()
           #     lr = scheduler.get_last_lr()[0] if scheduler else CFG.one_cycle_max_lr

                if batch_idx == 500 and batch_idx > 120:
                   torch.save({"model":model.state_dict()},f'./{path}/train_{epoch}_{batch_idx}'+"_model")

                if batch_idx % 500 == 0 and batch_idx <=4000 and batch_idx > 0:
                    gc.collect()
                    if eval_loader:
                        rmseloss = evaluate_model(model,eval_loader)
                        if rmseloss < best_eval_score:
                            torch.save({"model":model.state_dict(),"loss":loss},f'./{path}/{rmseloss}-{epoch}'+"_model")

            if eval_loader:
                rmseloss = evaluate_model(model, eval_loader)
                if rmseloss < best_eval_score:
                    torch.save({"model": model.state_dict(), "loss": criterion},
                               f'./{path}/{rmseloss}-{epoch}' + "_model")

    return model

import gc
for fold in range(CFG.folds):
    gc.collect()
    train_model(train_loader,eval_loader)