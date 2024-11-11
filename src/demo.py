import re
import numpy as np
import torch
import torch.nn as nn
from glob import glob
import pathlib
import scipy
import os
import scipy.io as scio
import time
import h5py
import cv2
dtype = torch.cuda.FloatTensor
from glob import glob
import pathlib
import scipy
import os
import torch
import torch.nn as nn
import scipy.io as scio
import copy
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import gc
norm = Normalize(vmin=0,vmax=1)
color_map = ScalarMappable(norm=norm,cmap="winter")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

from models import Intensity_Branch_Module,Depth_modual

Intensity_branch = Intensity_Branch_Module(1,32)
Depth_branch = Depth_modual(in_channels=1)

data = "../data/LR_Art_2_50.mat"
Intensity_PreTrain = "../models/Intensity_branch.pth"
Depth_PreTrain = "../models/depth_branch.pth"

Intensity_branch.load_state_dict(torch.load(Intensity_PreTrain)["model"],strict=False)
Intensity_branch.eval()
Intensity_branch.cuda()

Depth_branch.load_state_dict(torch.load(Depth_PreTrain)["model"],strict=True)
Depth_branch.cpu()
Depth_branch.eval()

Dep_Target = scio.loadmat(data)["depth"]
Intensity_Target = scio.loadmat(data)["intensity"]
Dep_Target = np.asarray(Dep_Target).astype(np.float32)
Intensity_Target = np.asarray(Intensity_Target).astype(np.float32)
h, w = Dep_Target.shape
###
print(h,w,"shape of data")
M_mea = scio.loadmat(data)["spad"]
M_mea = scipy.sparse.csc_matrix.todense(M_mea)
M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, w, h, -1])  #note that there would be dimension transpose if data is generated with some different settings, so if you find the results are strange or transposed, please check the dimension here and that in test data generating.
M_mea = np.transpose(M_mea, (0, 1, 4, 3, 2))
M_mea = M_mea[:,:,:1024,:,:]


M_meabak = M_mea

### to save time
M_mea = M_mea[:, :, :320, :, :]

### show raw data
img = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        img[i, j] = np.argmax(M_mea[0, 0, :300, i, j])

plt.imshow(img)
plt.title("Raw Depth")
plt.tight_layout()
plt.axis("off")
plt.show()

img = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        img[i, j] = np.max(M_mea[0, 0, :300, i, j])

plt.imshow(img, cmap="gray")
plt.title("Raw Intensity")
plt.tight_layout()
plt.axis("off")
plt.show()
# over

tmp = np.max(M_mea)



bincount = M_mea.shape[2]



M_mea_re = np.zeros([bincount,h,w])
dep_re = np.zeros([h,w])
rates_pred = torch.zeros([1, bincount, h, w])
mid = torch.zeros([1, bincount, h, w])
DEVICE = "cuda"
spad_full = M_mea
# plt.plot(spad_full[0,0,:300,80,80])
# plt.title("Raw - visualization about time dimension")
# plt.tight_layout()
# plt.xticks([])
# plt.yticks([])
# plt.show()
# print(model)
with torch.no_grad():
   for i in range(0, bincount - 32, 10):
      #  print(i)
        if i > 1024 - 48:
            i = 1024 - 48
        spad = spad_full[0,0, i:i + 32, :, :]
        spad = torch.from_numpy(spad).to(DEVICE).type(torch.cuda.FloatTensor)
        spad = torch.unsqueeze(spad, 0).contiguous()
        spad = torch.unsqueeze(spad, 0).contiguous()
        y_pred,_ = Intensity_branch.forward(spad.to(DEVICE))
        y_pred = y_pred.squeeze(1)
        rates_pred[:, i + 10:i + 22, :, :] = y_pred[:, 10:-10, :, :]
        if i == 0:
            print(" i == 0")
            rates_pred[:, i:i + 32, :, :] = y_pred[:, :, :, :]
        if i == 1024 - 32:
            print("i == 1024 - 64")
            rates_pred[:, i+12:i + 64, :, :] = y_pred[:, 12:, :, :]
        gc.collect()
        torch.cuda.empty_cache()
        del spad


print("inten branch module over")
M_mea_re = rates_pred[0,:,:,:]
Inten_bak = copy.deepcopy(M_mea_re)

img = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        img[i, j] = torch.argmax(M_mea_re[115:179, i, j])

plt.imshow(img)
plt.title("Depth from Intensity Branch Module")
plt.tight_layout()
plt.axis("off")
plt.show()

img = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        img[i, j] = torch.max(M_mea_re[80:220, i, j])

plt.imshow(img, cmap="gray")
plt.title("Intensity from Intensity Branch Module")
plt.tight_layout()
plt.axis("off")
plt.show()

intensity = img
Intensity_Target /= np.max(Intensity_Target)


bincount = M_meabak.shape[2]
M_mea_re = np.zeros([bincount,h,w])

with torch.no_grad():
    torch.cuda.empty_cache()
    print(M_meabak.shape,"input shape")
    M_meabak = torch.from_numpy(M_meabak)
    rates = Depth_branch(M_meabak)
    M_mea_re[:, :, :] = rates[0,0,:,:,:].cpu().numpy()
    del rates
    gc.collect()
    torch.cuda.empty_cache()


img = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        img[i, j] = np.max(M_mea_re[80:300, i, j])

plt.imshow(img, cmap="gray")
plt.title("Intensity from Depth Branch Module")
plt.tight_layout()
plt.axis("off")
plt.show()

# plt.plot(M_mea_re[:300, 80, 80])
# plt.title("visualization about time dimension")
# plt.tight_layout()
# plt.xticks([])
# plt.yticks([])
# plt.show()

Inten_bak = np.array(Inten_bak)
Depth_pred = np.zeros([h,w])
#M_mea_re[:115, :, :] = 0
#M_mea_re[179:, :, :] = 0
Intensity_pred = np.zeros([h,w])
intensity = np.array(intensity)
for i in range(h):
    for j in range(w):
        a = int(np.argmax(M_mea_re[:182, i, j]))
    #    print(a)
    #    print(Inten_bak.shape)
     #   if a <= 180:
        Intensity_pred[i, j] = np.max(Inten_bak[a - 10:a + 10, i, j])
        Depth_pred[i,j] = a

plt.imshow(Depth_pred)
plt.title("Depth of output")
plt.show()

plt.imshow(Intensity_pred,cmap="gray")
plt.title("Intensity of output")
plt.show()


Intensity_pred /= np.max(Intensity_pred)
Intensity_Target /= np.max(Intensity_Target)

Intensity_pred += (np.mean(Intensity_Target) - np.mean(Intensity_pred))
inten_rmse = np.sqrt(np.mean((Intensity_Target - Intensity_pred) ** 2))

print(inten_rmse,"Intensity rmse")

C = 3e8
# Tp = 80*1024e-12
#  tmp = 80*1000e-12
bin = 80e-12
Depth_pred = Depth_pred * bin  * C / 2

#Dep_Target = Dep_Target * bin  * C / 2
rmse = np.sqrt(np.mean((Depth_pred - Dep_Target) ** 2))
print(rmse,"Depth rmse")
