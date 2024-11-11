import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import torch.nn.functional as F




FoV = 0.03450/1500
Spod = 0.1075/1500
width = 100e-12
bin = 80e-12
A = np.zeros([15,15,13])
A = np.zeros([3,3,1])
# make PSF
def PSF_make(A):
    Psf_x,Psf_y,Psf_z = A.shape
    for a in range(Psf_x):
        for b in range(Psf_y):
            for c in range(Psf_z):
                theta = np.arccos(np.cos((a - (1 + Psf_x) / 2) * FoV)*np.cos((b - (1 + Psf_y) / 2) * FoV))
                A[a,b,c] = np.exp(-(theta/Spod)**2/2 -((c - (Psf_z+1)/2)*bin/width)**2/2)
    return A

#A = PSF_make(A)

#print(A[7,8,7])

def modify_conv_padding(model,new_padding_model="replicate"):
    for name,module in model.named_children():
        if isinstance(module,nn.Conv2d):
            module.padding_mode = new_padding_model
        else:
            modify_conv_padding(module,new_padding_model)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3,padding_mode="replicate")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3,padding_mode="replicate")

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class ConvAttention(nn.Module):
    def __init__(self,num_channels):
        super(ConvAttention,self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_fc = nn.Sequential(
            nn.Linear(num_channels,num_channels // 16),
            nn.ReLU(),
            nn.Linear(num_channels // 16,num_channels),
            nn.Sigmoid()
        )

    def forward(self,x):
        batch_size,num_channels,_,_=x.shape
        gap = self.global_avg_pool(x).view(batch_size,num_channels)
        attention_weights = self.attention_fc(gap).view(batch_size,num_channels,1,1)
        return x * attention_weights


class SKConv3D(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,G=4,r=16,L=32):
        super(SKConv3D,self).__init__()
        d = max(in_channels // r,L)
        self.M = M
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv3d(in_channels,out_channels,kernel_size=3 + i * 2,stride=stride,padding=1+i,groups=G,padding_mode="replicate"),
                nn.GELU()
            ))
        self.gap = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(out_channels,d,1,bias=False),
            nn.ReLU(),
            nn.Conv3d(d,out_channels * M,1,bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
    #    print(x.size())
        batch_size,_,_,_,_ = x.size()
        output = []
        for conv in self.convs:
            output.append(conv(x))

        U = sum(output)
    #    print(U.shape,"U")
        s = self.gap(U)

    #    print(s.shape,"gap")
        z = self.fc(s)
    #    print(z.shape,"z")
        attention_vectors = self.softmax(z.view(batch_size,self.M,-1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        V = sum([a * (torch.squeeze(b,dim=1)) for a,b in zip(output,attention_vectors.split(1,dim=1))])

        return V

class FoV_Module(nn.Module):
    def __init__(self,A):
        super(FoV_Module,self).__init__()
     #   self.Conv3D = nn.Conv3d(in_channels,out_channels,kernel_size=(5), padding=(2),padding_mode="replicate")
        self.blur = PSF_make(A)
        self.blur = np.reshape(self.blur,[1, 1, 3, 3, 1])
        self.blur = np.transpose(self.blur, [0, 1, 4, 2, 3])
        self.blur = self.blur / np.sum(self.blur)

        self.blur = torch.from_numpy(self.blur).float()

        self.photons_conv = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1,padding_mode="replicate")
        for param in self.photons_conv.parameters():
            param.requires_grad = False
        self.photons_conv.weight.data = self.blur

    def forward(self, x):
      #  x = self.Conv3D(x)
        x = self.photons_conv(x)
      #  x = F.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock,self).__init__()
        self.skconv1 = SKConv3D(in_channels,out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.gelu = nn.GELU()
        self.skconv2 = SKConv3D(out_channels,out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self,x):
        out = self.gelu(self.bn1(self.skconv1(x)))
        out = self.bn2(self.skconv2(out))
        out += x
        out = self.gelu(x)
        return out

class Intensity_Branch_Module(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Intensity_Branch_Module,self).__init__()
        self.Depthconv = nn.Conv3d(in_channels,32,kernel_size=(7,1,1), padding=(3,0,0))
        self.Depthconv1 = nn.Conv3d(32,1,kernel_size=(1), padding=(0,0,0))
        self.Depthconv2 = nn.Conv3d(in_channels,32,kernel_size=(1), padding=(0,0,0))
     #   self.Depthconv2 = nn.Conv3d(32,1,kernel_size=1)
        self.skconv = SKConv3D(32,32)
        self.skconv1 = SKConv3D(32,32)
        self.skconv2 = SKConv3D(32, 32)
        self.skconv3 = SKConv3D(32, 32)
        self.outconv = nn.Conv3d(32,1,kernel_size=(1), padding=(0,0,0))



    #    self.blur = np.array(scio.loadmat(mat)["blur"]).reshape([1, 1, 15, 15, 13])
    #    self.blur = np.transpose(self.blur, [0, 1, 4, 2, 3])
        self.blur = FoV_Module(A)

    def forward(self,x):
        x = self.Depthconv(x)
        x = F.relu(x)
        mid = self.Depthconv1(x)
        mid = F.relu(mid)
        mid = self.Depthconv2(mid)
        mid = F.relu(mid)
        res = x
        x = self.skconv(x)
        x = F.relu(x)
        x = torch.squeeze(x,dim=0)
        x = res + x
        x = F.relu(x)
        res = x
        x = self.skconv1(x)
        x = torch.squeeze(x, dim=0)
        x = res + x
        x = F.relu(x)
        x = x +  mid
        denosieout = x
        res = x
        x = self.skconv2(x)
        x = torch.squeeze(x, dim=0)
        x = res + x
        x = F.relu(x)
        res = x
        x = self.skconv3(x)
        x = torch.squeeze(x, dim=0)
        x = res + x
        x = F.relu(x)
        x = self.outconv(x)
        x = F.relu(x)
        output_from_FoV = self.blur(x)
        return x,output_from_FoV



class Block(nn.Module):
    def __init__(self, in_channels):
        outchannel_block = 16
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_block, 1, padding=0, dilation=1, bias=True),
                                   nn.ReLU(inplace=True))
        self.feat1 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=1, dilation=1, bias=True),
                                   nn.ReLU(inplace=True))
        self.feat15 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        self.feat2 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True))
        self.feat25 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        self.feat = nn.Sequential(nn.Conv3d(24, 8, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        feat1 = self.feat1(conv1)
        feat15 = self.feat15(feat1)
        feat2 = self.feat2(conv1)
        feat25 = self.feat25(feat2)
        feat = self.feat(torch.cat((feat1, feat15, feat2, feat25), 1))
        return torch.cat((inputs, feat), 1)

class Pre(nn.Module):
    def __init__(self, in_channels):
        super(Pre, self).__init__()
        outchannel_MS = 2
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1, 1, 1), padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1, 1, 1), padding=2, dilation=2, bias=True),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=1, dilation=1, bias=True),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True))

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1)
        return torch.cat((conv1, conv2, conv3, conv4), 1)
def Downsampling(in_channels, num_layers=4):
     layers = []
     in_filters = in_channels
     for i in range(num_layers):
         layers.extend(
             [nn.Conv3d(in_filters, 2 * in_filters, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),
              nn.ReLU(inplace=True)])
         in_filters *= 2
     return nn.Sequential(*layers)


def Upsampling(in_channels, num_layers=4):
    layers = []
    in_filters = in_channels
    for i in range(num_layers-1):
        layers.extend([nn.ConvTranspose3d(in_filters, in_filters // 2, kernel_size=(6, 3, 3), stride=(2, 1, 1),
                                          padding=(2, 1, 1), bias=False),
                       nn.ReLU(inplace=True)])
        in_filters //= 2
    layers.extend([nn.ConvTranspose3d(in_filters, in_filters // 2, kernel_size=(6, 3, 3), stride=(2, 1, 1),
                                      padding=(2, 1, 1), bias=False)])
    return nn.Sequential(*layers)


class Depth_modual(nn.Module):
    def __init__(self, in_channels=1):
        super(Depth_modual, self).__init__()
        self.pre = Pre(in_channels)
        self.C1 = nn.Sequential(nn.Conv3d(8, 2, kernel_size=1, stride=(1, 1, 1), bias=True), nn.ReLU(inplace=True))
        self.Downsample = Downsampling(2, num_layers=4)
        self.feature_list = [Block(32 + 8 * c) for c in range(10)]
        self.feature_list = nn.ModuleList(self.feature_list)
        self.Upsample = Upsampling(112, num_layers=4)
        self.C2 = nn.Sequential(nn.Conv3d(7, 1, kernel_size=1, stride=(1, 1, 1), bias=True), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.pre(x)
        x = self.C1(x)
        x = self.Downsample(x)
        for i in range(10):
            x = self.feature_list[i](x)
        x = self.Upsample(x)
        x = self.C2(x)
        return x

