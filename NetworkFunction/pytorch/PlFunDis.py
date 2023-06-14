from itertools import permutations, product
import numpy as np
import pywt
import torch
from torch import nn
from scipy import signal

from DataFunction.CoorTrans import LLA_to_XYZ


def IndexShuffle(data1, data2):
    indexNum = data1.shape[0]
    index = np.random.permutation(indexNum)
    data1 = data1[index]

    temp = np.zeros([indexNum, int(data2.size / indexNum)])
    for i in range(indexNum):
        temp[i] = data2[i * 6:(i + 1) * 6]

    temp = temp[index]
    data2 = temp.reshape(temp.size)

    return data1, data2


def Cwt(data, lengthOfF, sampleTime):
    result = np.zeros([data.shape[0], lengthOfF, data.shape[1]])
    for i in range(data.shape[0]):
        IQ = data[i]
        # DPV
        wavename = 'cgau8'
        # 频率划分个数
        totalscal = lengthOfF + 1
        fc = pywt.central_frequency(wavename)
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)
        [cwtmatr1, frequencies1] = pywt.cwt(IQ, scales, wavename, sampleTime / IQ.__len__())
        result[i] = abs(cwtmatr1)

    return result


def STFT(data, lengthOfF, sampleTime):
    result = np.zeros([data.shape[0], lengthOfF, data.shape[1]])
    for i in range(data.shape[0]):
        IQ = data[i]

        f, t, Zxx = signal.stft(x=IQ, fs=IQ.__len__() / sampleTime, nperseg=lengthOfF, noverlap=lengthOfF-1, return_onesided=False)

        if Zxx.shape[1] > result.shape[2]:
            result[i] = abs(Zxx[:, :result.shape[2]])
        else:
            result[i] = abs(Zxx)

    return result


def SITloss(input, target, emitterNum, coordDim, isCoorTrans=False, device='cpu', training=True):

    combResult = [(0, 1, 2)] # SITloss, sort operation is done in DataSet function
    #combResult = list(permutations(np.arange(emitterNum), emitterNum)) # PITloss

    if training:
        reg_loss = nn.MSELoss(reduction='none')
        cla_loss = nn.BCEWithLogitsLoss(reduction='none')

        for com in combResult:
            if com == (0, 1, 2):
                continue
            for i in com:
                input = torch.concat((input, input[:, i * 3:(i + 1) * 3]), 1)
            target = torch.concat((target, target[:, :emitterNum * (coordDim + 1)]), 1)

        input_coord = input[:, :2].reshape([-1, 2])
        input_conf = input[:, 2].reshape([-1, 1])
        target_coord = target[:, :2].reshape([-1, 2])
        target_conf = target[:, 2].reshape([-1, 1])
        for i in range(1, emitterNum * len(combResult)):
            input_coord = torch.concat((input_coord, input[:, i * 3:i * 3 + 2].reshape([-1, 2])), 1)
            target_coord = torch.concat((target_coord, target[:, i * 3:i * 3 + 2].reshape([-1, 2])), 1)
            input_conf = torch.concat((input_conf, input[:, i * 3 + 2].reshape([-1, 1])), 1)
            target_conf = torch.concat((target_conf, target[:, i * 3 + 2].reshape([-1, 1])), 1)

        coordLoss = reg_loss(input_coord, target_coord)
        coordLoss_sum = \
            torch.sqrt(torch.sum(coordLoss[:, :6], 1).reshape([-1, 1]))
        for i in range(1, len(combResult)):
            coordLoss_sum = \
                torch.concat(
                    (coordLoss_sum,
                     torch.sqrt(torch.sum(coordLoss[:, i * 6:i * 6 + 6], 1).reshape([-1, 1]))), 1)

        confLoss = cla_loss(input_conf, target_conf)
        confLoss_sum = torch.mean(confLoss[:, :3], 1).reshape([-1, 1])
        for i in range(1, len(combResult)):
            confLoss_sum = \
                torch.concat(
                    (confLoss_sum, torch.mean(confLoss[:, i * 3:i * 3 + 3], 1).reshape([-1, 1])), 1)

        loss_sum = coordLoss_sum + confLoss_sum
        loss_sum, indeices = torch.sort(loss_sum, -1)

        coordLoss = coordLoss_sum.gather(1, indeices[:, 0].reshape([-1, 1]))
        confLoss = confLoss_sum.gather(1, indeices[:, 0].reshape([-1, 1]))

        loss = torch.mean(loss_sum.gather(1, indeices[:, 0].reshape([-1, 1])))
        coordLoss = torch.mean(coordLoss)
        confLoss = torch.mean(confLoss)

        loss.requires_grad_(True)

        return loss, coordLoss, confLoss
    else:
        tp = fp = tn = fn = 0

        target_confidence = target[:, 2::(2 + 1)]
        target_x = target[:, 0::(2 + 1)]
        target_y = target[:, 1::(2 + 1)]

        input_confidence = input[:, 2::(2 + 1)]
        input_x = input[:, 0::(2 + 1)]
        input_y = input[:, 1::(2 + 1)]

        coordLoss = torch.ones(target_confidence.shape[0]) * torch.inf
        confLoss = torch.ones(target_confidence.shape[0]) * torch.inf
        loss = torch.ones(target_confidence.shape[0]) * torch.inf
        for i in range(input_x.shape[0]):
            best_comb = None
            for com in combResult:
                index = (nn.Sigmoid()(input_confidence[i, com]) > 0.5).nonzero(as_tuple=True)[0]
                # index = (target_confidence[i] > 0.5).nonzero(as_tuple=True)[0]

                temp = torch.mean(torch.sqrt(torch.square(input_x[i, com] - target_x[i])[index] + \
                                            torch.square(input_y[i, com] - target_y[i])[index]))
                temp1 = nn.BCEWithLogitsLoss()(input_confidence[i, com], target_confidence[i])
                if loss[i] > temp + temp1:
                    coordLoss[i] = temp
                    confLoss[i] = temp1
                    loss[i] = coordLoss[i] + confLoss[i]
                    best_comb = com

            input_conf = nn.Sigmoid()(input_confidence[i, best_comb])
            target_conf = target_confidence[i]
            input_conf[input_conf > 0.5] = 1
            input_conf[input_conf <= 0.5] = 0
            input_conf = input_conf.bool()
            target_conf = target_conf.bool()

            cur_tp = torch.sum((input_conf & target_conf).int()).item()
            tp += cur_tp
            fp += torch.sum(input_conf.int()).item() - cur_tp

            cur_tn = torch.sum((~input_conf & ~target_conf).int()).item()
            tn += cur_tn
            fn += torch.sum((~input_conf).int()).item() - cur_tn

        loss = torch.mean(loss)
        coordLoss = torch.mean(coordLoss)
        confLoss = torch.mean(confLoss)

        return coordLoss, coordLoss, confLoss, (tp, fp, tn, fn)


class PlModel(nn.Module):
    def __init__(self, fc_par_num=128, fc_depth=4, emitter_num=1, receiver_num=3, backbone=None, coordDim=3):
        super().__init__()
        self.backbone = backbone
        self.coordDim = coordDim

        self.fc1_input = nn.Sequential(nn.Linear(fc_par_num * 2, fc_par_num), nn.ReLU(True))
        self.fc1 = nn.Sequential(*[nn.Sequential(nn.Linear(fc_par_num, fc_par_num), nn.ReLU(True)) for i in range(fc_depth)])
        self.fc1_output = nn.Linear(fc_par_num, emitter_num * coordDim)

        self.fc2_input = nn.Sequential(nn.Linear(fc_par_num * 2, fc_par_num), nn.ReLU(True))
        self.fc2 = nn.Sequential(*[nn.Sequential(nn.Linear(fc_par_num, fc_par_num), nn.ReLU(True)) for i in range(fc_depth)])
        self.fc2_output = nn.Linear(fc_par_num, emitter_num)

        self.fc_input = nn.Sequential(nn.Linear(receiver_num * 6, fc_par_num), nn.ReLU(True))
        self.fc = nn.Sequential(
            *[nn.Sequential(nn.Linear(fc_par_num, fc_par_num), nn.ReLU(True)) for i in range(fc_depth)])
        self.fc_output = nn.Linear(fc_par_num, fc_par_num)

    def forward(self, inputs):
        # 接收的信号
        x1 = inputs[0]

        # 其他信息
        x2 = inputs[1]

        # # 计算FLOPs
        # x1 = inputs
        # x2 = torch.rand(1, 24).to(torch.device('cuda'))

        x1 = self.backbone(x1)

        x2 = self.fc_input(x2)
        x2 = self.fc(x2)
        x2 = self.fc_output(x2)

        x = torch.cat((x1, x2), -1)

        x1 = self.fc1_input(x)
        x1 = self.fc1(x1)
        x1 = self.fc1_output(x1)

        x2 = self.fc2_input(x)
        x2 = self.fc2(x2)
        x2 = self.fc2_output(x2)

        x = x1[:, :self.coordDim]
        for i in range(x2.shape[1]):
            if i > 0:
                x = torch.cat((x, x1[:, i * self.coordDim:(i + 1) * self.coordDim]), 1)
            x = torch.cat((x, x2[:, i].reshape([x2.shape[0], 1])), 1)

        return x
