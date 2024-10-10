import argparse
import os
import shutil
import time, math ,datetime, re
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as  F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.autograd.variable import Variable
from scipy.interpolate import interp2d
from BaseModel import BaseModel
from resnet3x3 import resnet18
from lenet5 import LeNet
import data_tools
import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):#产生前半段深度模型
    def __init__(self):
        super(SmallNet,self).__init__()
        #self.features = resnet18(pretrained = False)
        #self.features.fc = nn.Threshold(-1e20,1e20)
        self.features = LeNet(pretrained = False)
        self.features.fc = nn.Threshold(-1e20,1e20)

    def forward(self,pressure):
        out = self.features(pressure)
        return out

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x

        return out


class TouchNet(nn.Module):
    def __init__(self,num_classes = 1000,nFrames =5):
        super(TouchNet,self).__init__()
        self.net = SmallNet()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.combination = nn.Conv2d(128,27,kernel_size=1,padding = 0)
        self.lstm = nn.LSTM(10*15*15,64,1)
        self.classifier = nn.Linear(64,27)
        self.classifier2 = nn.Linear(500,27)
        self.droppout = nn.Dropout2d(p=0.3, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv13d = nn.Conv3d(in_channels=1,
                          out_channels=16,
                          kernel_size=(2, 5, 5),
                          padding=0)
        self.conv03d = nn.Conv3d(in_channels=1,
                          out_channels=16,
                          kernel_size=(2, 9, 9),
                          padding=1,)
        self.conv23d = nn.Conv3d(in_channels=1,
                          out_channels=16,
                          kernel_size=(2, 7, 7),
                          padding=0,)

        self.maxpool3d = nn.MaxPool3d(2,2)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(6272,128)
        self.fc_2 = nn.Linear(4608,128)
        self.fc_0 = nn.Linear(18432,128)
        self.conv2_1 = nn.Conv2d(32,64,kernel_size=5,stride=2)
        self.conv2_2 = nn.Conv2d(32,64,kernel_size=5,stride=2)
        self.conv2_3 = nn.Conv2d(64,128,kernel_size=5,stride= 2)

        self.fc_at = nn.Conv2d(32,8,1)
        self.fc_at2 = nn.Conv2d(8,32,1)

        self.fc_at_2 = nn.Conv2d(256,32,1)
        self.fc_at_22 = nn.Conv2d(32,256,1)

        self.spat = nn.Conv2d(3,3,1)
        self.fc3d = nn.Linear(3136,27)
        self.fc3d2 = nn.Linear(128,27)
        self.nlb = NonLocalBlock(channel = 4)

        self.conv_final = nn.Conv2d(32,128,kernel_size=3,stride = 2)
        self.BN = nn.BatchNorm2d(32)
        self.para_1 = nn.Parameter(torch.full((1,),0.05))
        self.para_2 = nn.Parameter(torch.full((1,),0.15))
        self.para_3 = nn.Parameter(torch.full((1,),0.9))

    def channel_attention(self,x,in_chanel):
        y = self.avg_pool(x)
        y = self.fc_at(y)
        y = self.relu(y)
        y = self.fc_at2(y)
        y = self.sigmoid(y)
        return x * y

    def channel_attention2(self,x):
        y = self.avg_pool(x)
        y = self.fc_at_2(y)
        y = self.relu(y)
        y = self.fc_at_22(y)
        y = self.sigmoid(y)
        return x * y

    def sp_attention(self,x):
        score = self.spat(x)
        score = self.sigmoid(score)
        return x * score

    def downsample(self,data):
        """输入最好是32x32"""
        # 初始化一个16x16的下采样矩阵
        downsampled_data = np.zeros((32, 3, 16, 16))

        for k in range(32):
            for v in range(3):
                single = torch.tensor(data[k, v, :, :]).cpu()
                downsampled_matrix = np.zeros((16, 16))

                for i in range(16):
                    for j in range(16):
                        # 计算4x4子矩阵的平均值
                        submatrix = single[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
                        downsampled_matrix[i, j] = torch.mean(submatrix)
                downsampled_data[k, v, :, :] = downsampled_matrix

        return torch.tensor(downsampled_data)

    def upsample(self,data):
        # 创建一个32x32到64x64的双线性插值函数

        downsampled_data = np.zeros((32, 3, 64, 64))

        for k in range(32):
            for v in range(3):
                single = torch.tensor(data[k, v, :, :]).cpu()
                interp_func = interp2d(np.arange(32), np.arange(32), single, kind='linear')

                # 在64x64的网格上进行插值
                x_new = np.linspace(0, 31, 64)
                y_new = np.linspace(0, 31, 64)
                expanded_matrix = interp_func(x_new, y_new)
                downsampled_data[k, v, :, :] = expanded_matrix

        # expanded_matrix现在包含了插值后的64x64矩阵

        return torch.tensor(downsampled_data)


    def forward(self,x):
        x = self.sp_attention(x)

        piece = x[:,2,:,:].reshape(32,1,32,32)
        x = torch.cat((x,piece),dim=1)
        x = self.nlb(x)
        x = x[:,:3,:,:]
        x = x.reshape(32, 1, 3, 32, 32)
        ori = x
        x = self.conv13d(x)
        x_1 = self.conv03d(ori)
        x_2 = self.conv23d(ori)
        # print(x.shape,x_1.shape,x_2.shape)
        x = x[:,:,:,:26,:26]
        x_1 = x_1[:,:,0:2,:26,:26]
        x = x.reshape(32, 32, 26, 26)
        x_1 = x_1.reshape(32, 32, 26, 26)
        x_2 = x_2.reshape(32, 32, 26, 26)
        x = x_1*self.para_1+x_2*self.para_2+x*self.para_3
        #x = self.BN(x)
        x = self.channel_attention(x,32)
        x = self.conv_final(x)
        # print(x.shape)
        x = x.reshape(32, -1)

        x = self.fc_0(x)

        self.droppout(x)
        x = self.fc3d2(x)
        score = 1
        return(x),score

class ClassificationModel(BaseModel):
    def name(self):
        return 'ClassficationModel'


    def initialize(self,numClasses,sequenceLength=1,baseLr = 0.001):
        BaseModel.initialize(self)
        print("Base Lr = %e"%baseLr)

        self.baseLr = baseLr
        self.sequenceLength = sequenceLength
        self.numclasses = numClasses

        self.model = TouchNet(num_classes=self.numclasses,nFrames=self.sequenceLength)
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        cudnn.benchmark = True

        #设置优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(),'lr_mult':1.0},
        ],self.baseLr)
        self.optimizers = [self.optimizer]

        self.criterion = nn.CrossEntropyLoss().cuda()

        self.epoch = 0

        self.error = 1e20
        self.bestPrec = 1e20

        self.dataProcessor = None

    def step(self,inputs,isTrain = True, params = {}):

        if isTrain:
            self.model.train()
            assert not inputs['objectId'] is None
        else:
            self.model.eval()

        image = torch.autograd.Variable(inputs['image'].cuda(),requires_grad = (isTrain))
        pressure = torch.autograd.Variable(inputs['pressure'].cuda(),requires_grad = (isTrain))
        objectId = torch.autograd.Variable(inputs['objectId'].cuda(),requires_grad = False) if 'objectId' in inputs else None

        if isTrain:
            output,score = self.model(pressure)
        else:
            with torch.no_grad():
                output,score = self.model(pressure)
        #print(output)

        _, pred = output.data.topk(1,1,True,True)

        res = {
            'gt': None if objectId is None else objectId.data,
            'pred':pred,
        }

        if objectId is None:
            return res, {}

        loss = self.criterion(output, objectId.view(-1))

        (prec1, prec3) = self.accuracy(output, objectId, topk=(1,min(3,self.numclasses)))

        if isTrain:

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        losses = OrderedDict([
            ('Loss',loss.data.item()),
            ('Top1',prec1),
            ('Top3',prec3),
        ])



        return res, losses,pred,objectId.view(-1),

    def accuracy(self,output,target,topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _,pred = output.data.topk(maxk,1,True,True)

        pred = pred.t()
        correct = pred.eq(target.data.view(1,-1).expand_as(pred))


        res = []

        for k in topk:#寻找top_k的参数的准确率，top3就是意味着概率最高的前三个有对应的object
            correct_k = correct[:k].contiguous().view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res[0], res[1]

    def importState(self,save):
        params = save['state_dict']
        if hasattr(self.model, 'module'):
            try:
                self.model.load_state_dict(params,strict = True)
            except:
                self.model.module.load_state_dict(params,strict = True)

        else:
            params = self._clearState(params)
            self.model.load_state_dict(params,strict = True)

        self.epoch = save['epoch'] if 'epoch' in save else 0
        self.bestPrec = save['best_prec1'] if 'best_prec1' in save else 1e20
        self.error = save['error'] if 'error' in save else 1e20

        print('Imported checkpoint for epoch %05d with loss = %.3f...' % (self.epoch, self.bestPrec))

    def _clearState(self,params):
        res = dict()
        for k,v in params.item():
            kNew = re.sub('^module\.','', k)
            res[kNew] = v

        return res

    def exportState(self):
        dt = datetime.datetime.now()
        state = self.model.state_dict()
        for k in state.keys():
            state[k] = state[k].cpu()


        return {

            'state_dict': state,
            'epoch': self.epoch,
            'error': self.error,
            'best_prec1': self.bestPrec,
            'datetime': dt.strftime("%Y-%m-%d %H:%M:%S")
            }

    def updataLearningRate(self,epoch):
        self.adjust_learning_rate_new(epoch,self.baseLr)

    def adjust_learning_rate_new(self,epoch,base_Lr, period = 100):
        gamma = 0.1 ** (1.0/period)
        lr_default = base_Lr * ((gamma) ** (epoch))
        print('New lr_default = %f' % lr_default)

        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr_mult'] * lr_default


