import os
import sys

# from matplotlib import pyplot as plt

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DataStructure')
sys.path.append(base_dir)
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
#import dill
import numpy as np
from DataFunction import DataIO


# 纯模拟数据(tf版本)
class DataSetCsv:
    def __init__(self, dataFileRoot):

        # 保存根目录
        self.dataFileRoot = dataFileRoot

        # 加载文件夹名称
        self.receDirs = DataIO.GetDirName(dataFileRoot)
        # 接收机文件夹
        self.receDirs = [n for n in self.receDirs if 'Receiver' in n] if self.receDirs else []
        self.receDataFileNames = []
        for dir in self.receDirs:
            temp = DataIO.GetFileName(dataFileRoot + '/' + dir)
            # 将文件名按指定方式排序
            temp.sort(key=lambda x: int(x[:-4]))
            self.receDataFileNames.append(temp)

        # 加载文件夹名称
        self.emitDirs = DataIO.GetDirName(dataFileRoot)
        # 发射机文件夹
        self.emitDirs = [n for n in self.emitDirs if 'Emitter' in n] if self.emitDirs else []
        self.emitDataFileNames = []
        for dir in self.emitDirs:
            temp = DataIO.GetFileName(dataFileRoot + '/' + dir)
            # 将文件名按指定方式排序
            temp.sort(key=lambda x: int(x[:-4]))
            self.emitDataFileNames.append(temp)

        # 加载标签文件
        inforPath = self.dataFileRoot + '/Information.npy'
        if os.path.exists(inforPath):
            temp = np.load(inforPath)
            self.information = np.zeros(temp.shape)
            for i in range(self.information.shape[0]):
                self.information[i] = np.array([float(x) for x in temp[i]])

        self.receiverNum = self.receDirs.__len__()
        self.emitterNum = self.emitDirs.__len__()

        # 数据个数
        self.dataNum = self.information.shape[0] if os.path.exists(inforPath) else 0
        # 已经使用的数据编号
        self.usedDataNum = 0
        # 数据集打乱的编号
        self.dataPermutation = np.random.permutation(self.dataNum)

        print('find {} data'.format(self.receDataFileNames[0].__len__() if os.path.exists(inforPath) else 0))

    def GetEmitSignalIQData(self, batchSize, singleDataShape, batchIndex, isNormal=False):
        data = np.zeros((batchSize,) + singleDataShape, dtype=complex)
        for k in range(data.shape[0]):
            for m in range(data.shape[1]):
                if m >= self.emitDirs.__len__():
                    continue
                data[k, m, :, 0] = np.load(
                    self.dataFileRoot + '/' + self.emitDirs[m] + '/' + self.emitDataFileNames[m][batchIndex[k]])[
                                   :data.shape[2]]
                if isNormal and np.abs(data[k, :, :, m]).max() != 0:
                    data[k, m, :, 0] /= np.abs(data[k, m, :, 0]).max()
        return data

    def GetSignalIQData(self, batchSize, singleDataShape, batchIndex, isNormal=False):
        data = np.zeros((batchSize,) + singleDataShape, dtype=complex)
        for k in range(data.shape[0]):
            for m in range(data.shape[1]):
                if m >= self.receDirs.__len__():
                    continue
                try:
                    data[k, m, :, 0] = np.load(
                        self.dataFileRoot + '/' + self.receDirs[m] + '/' + self.receDataFileNames[m][batchIndex[k]])[
                                       :data.shape[2]]
                except:
                    print('Waring: the required data is unsupported!')
                    exit(-1)
                if isNormal and np.abs(data[k, m, :, 0]).max() != 0:
                    data[k, m, :, 0] /= np.abs(data[k, m, :, 0]).max()
        return data

    def GetSignalIQCWTData(self, batchSize, singleDataShape, batchIndex, isNormal=False):
        data = np.zeros((batchSize,) + singleDataShape, dtype=complex)
        for k in range(data.shape[0]):
            for m in range(data.shape[3]):
                if m >= self.receDirs.__len__():
                    continue
                data[k, :, :, m] = np.load(
                    self.dataFileRoot + '/' + self.receDirs[m] + '/' + self.receDataFileNames[m][batchIndex[k]])
                if isNormal and np.abs(data[k, :, :, m]).max() != 0:
                    data[k, :, :, m] /= np.abs(data[k, :, :, m]).max()
        return data

    def GetRecPosVel(self, batchSize, singleDataShape, batchIndex, isNormal=False):
        posRange = None
        velRange = None
        if isNormal:
            posRange = np.hstack((self.information[batchIndex, 5 + 2 * self.emitterNum:5 + 2 * self.emitterNum + 6].min(1).reshape([batchSize, 1]),
                                  self.information[batchIndex, 5 + 2 * self.emitterNum:5 + 2 * self.emitterNum + 6].max(1).reshape([batchSize, 1])))
            velRange = np.hstack((self.information[batchIndex, 5 + 2 * self.emitterNum + 6].reshape([batchSize, 1]),
                                  self.information[batchIndex, 5 + 2 * self.emitterNum + 7].reshape([batchSize, 1])))

        data = np.zeros((batchSize,) + singleDataShape)
        for k in range(data.shape[1]):
            data[:, k] = self.information[batchIndex, -(self.receDirs.__len__() * 6 + self.emitDirs.__len__() * 6) + k]

            if k % 6 < 3 and isNormal:
                data[:, k] /= posRange[:, 1]
            if k % 6 >= 3 and isNormal:
                data[:, k] /= velRange[:, 1]

        return data

    def GetDelayOfTimeFre(self, batchSize, singleData2Shape, batchIndex):
        data = np.zeros((batchSize,) + singleData2Shape)
        for k in range(data.shape[1]):
            data[:, k] = self.information[batchIndex, -(
                    self.receDirs.__len__() * 6 + self.emitDirs.__len__() * 6 + 2 * self.receDirs.__len__() * self.emitDirs.__len__()) + k]
        return data

    def GetEmiPosVel(self, batchSize, labelShape, batchIndex, isNormal=False):

        posRange = None
        velRange = None
        if isNormal:
            posRange = np.hstack((
                self.information[batchIndex, 5 + 2 * self.emitterNum + 8:5 + 2 * self.emitterNum + 8 + 6].min(1).reshape([batchSize, 1]),
                self.information[batchIndex, 5 + 2 * self.emitterNum + 8:5 + 2 * self.emitterNum + 8 + 6].max(1).reshape([batchSize, 1])))
            velRange = np.hstack((self.information[batchIndex, 5 + 2 * self.emitterNum + 8 + 6].reshape([batchSize, 1]),
                                  self.information[batchIndex, 5 + 2 * self.emitterNum + 8 + 7].reshape([batchSize, 1])))

        data = np.zeros((batchSize,) + labelShape)
        for k in range(data.shape[1]):
            data[:, k] = self.information[batchIndex, -(self.emitDirs.__len__() * 6) + k]

            if k % 6 < 3 and isNormal:
                data[:, k] /= posRange[:, 1]
            if k % 6 >= 3 and isNormal:
                data[:, k] /= velRange[:, 1]

        return data

    def GetBatchIndex(self, batchSize, isTrainData):
        if isTrainData:
            batchIndex = self.dataPermutation[self.usedDataNum:self.usedDataNum + batchSize]

            self.usedDataNum += batchSize
            if self.usedDataNum == self.dataNum:
                self.usedDataNum = 0
                # 所有样本循环一次后再次打乱数据集
                self.dataPermutation = np.random.permutation(self.dataNum)
        else:
            batchIndex = [x for x in
                          range(self.usedDataNum, self.usedDataNum + batchSize)]

            self.usedDataNum += batchSize
            if self.usedDataNum == self.dataNum:
                self.usedDataNum = 0
        return batchIndex

    def GetNextBatch(self, batchSize, singleData1Shape, singleData2Shape, labelShape, isTrainData=True, isNormal=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        # 加载每个batch数据
        data = []
        data.append(np.abs(self.GetSignalIQCWTData(batchSize, singleData1Shape, batchIndex, isNormal)))
        data.append(self.GetRecPosVel(batchSize, singleData2Shape, batchIndex, isNormal))
        label = self.GetEmiPosVel(batchSize, labelShape, batchIndex, isNormal)

        return data, label

    def GetNextBatch1(self, batchSize, singleDataShape, labelShape, isTrainData=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        data1 = self.GetDelayOfTimeFre(batchSize, (6,), batchIndex)
        data2 = self.GetRecPosVel(batchSize, (18,), batchIndex)
        data = np.zeros((batchSize,) + singleDataShape)

        # delta t
        data[:, 0] = (data1[:, 2] - data1[:, 0])
        data[:, 1] = (data1[:, 4] - data1[:, 0])
        # delta f
        data[:, 2] = (data1[:, 3] - data1[:, 1])
        data[:, 3] = (data1[:, 5] - data1[:, 1])

        # delta px
        data[:, 4] = data2[:, 6] - data2[:, 0]
        data[:, 5] = data2[:, 12] - data2[:, 0]
        # delta py
        data[:, 6] = data2[:, 7] - data2[:, 1]
        data[:, 7] = data2[:, 13] - data2[:, 1]
        # delta pz
        data[:, 8] = data2[:, 8] - data2[:, 2]
        data[:, 9] = data2[:, 14] - data2[:, 2]

        # delta vx
        data[:, 10] = data2[:, 9] - data2[:, 3]
        data[:, 11] = data2[:, 15] - data2[:, 3]
        # delta vy
        data[:, 12] = data2[:, 10] - data2[:, 4]
        data[:, 13] = data2[:, 16] - data2[:, 4]
        # delta vz
        data[:, 14] = data2[:, 11] - data2[:, 5]
        data[:, 15] = data2[:, 17] - data2[:, 5]

        label = self.GetEmiPosVel(batchSize, labelShape, batchIndex)

        return data, label

    def GetNextBatch2(self, batchSize, singleData1Shape, singleData2Shape, labelShape, isTrainData=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        # 加载每个batch数据
        data = []
        data.append(self.GetSignalIQData(batchSize, singleData1Shape, batchIndex))
        data.append(self.GetRecPosVel(batchSize, singleData2Shape, batchIndex))
        label = self.GetEmiPosVel(batchSize, labelShape, batchIndex)

        return data, label

    def GetNextBatch3(self, batchSize, singleData1Shape, singleData2Shape, labelShape, isTrainData=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        # 加载每个batch数据
        data = []
        temp = np.zeros((batchSize,) + singleData1Shape)
        temp1 = self.GetSignalIQData(batchSize, (3, 800, 1), batchIndex)
        temp[:, 0] = temp1[:, 0].real
        temp[:, 1] = temp1[:, 0].imag
        temp[:, 2] = temp1[:, 1].real
        temp[:, 3] = temp1[:, 1].imag
        temp[:, 4] = temp1[:, 2].real
        temp[:, 5] = temp1[:, 2].imag

        data.append(temp)
        data.append(self.GetRecPosVel(batchSize, singleData2Shape, batchIndex))

        label = np.zeros((batchSize,) + labelShape)
        data2 = self.GetDelayOfTimeFre(batchSize, (6,), batchIndex)
        # delta t
        label[:, 0] = (data2[:, 2] - data2[:, 0])
        label[:, 1] = (data2[:, 4] - data2[:, 0])
        # delta f
        label[:, 2] = (data2[:, 3] - data2[:, 1])
        label[:, 3] = (data2[:, 5] - data2[:, 1])

        return data, label

    def GetNextBatch4(self, batchSize, singleData1Shape, singleData2Shape, labelShape, isTrainData=True):

        batchIndex = self.GetBatchIndex(batchSize, isTrainData)

        # 加载每个batch数据
        data = []
        data.append(np.abs(self.GetSignalIQCWTData(batchSize, singleData1Shape, batchIndex)))
        data.append(self.GetRecPosVel(batchSize, singleData2Shape, batchIndex))

        label = np.zeros((batchSize,) + labelShape)
        data2 = self.GetDelayOfTimeFre(batchSize, (6,), batchIndex)
        # delta t
        label[:, 0] = (data2[:, 2] - data2[:, 0])
        label[:, 1] = (data2[:, 4] - data2[:, 0])
        # delta f
        label[:, 2] = (data2[:, 3] - data2[:, 1])
        label[:, 3] = (data2[:, 5] - data2[:, 1])

        return data, label


# 纯模拟数据
class DataSetCsv1:
    def __init__(self, root_path: str, receiverNum=3, emitterNum=1, time_scale=None, transform=None,
                 time_fre_trans=None, fre_scale=None, isNormal=False, coordDim=3, dataType='IQ'):

        root_path += "/Npy/" + dataType
        self.dataType = dataType
        self.dataset = DataSetCsv(root_path)
        self.root_path = root_path
        self.transform = transform
        self.time_fre_trans = time_fre_trans
        self.fre_scale = fre_scale
        self.time_scale = time_scale
        self.isNormal = isNormal
        self.coordDim = coordDim

    def getDataNum(self):
        return self.dataset.dataNum

    def getData(self, item, isGetFsFc=False):

        data1 = None
        if self.dataType == 'IQ':
            data1 = self.dataset.GetSignalIQData(1, (self.dataset.receiverNum, self.time_scale, 1), [item], self.isNormal)
        elif self.dataType == 'IQCWT':
            data1 = self.dataset.GetSignalIQCWTData(1, (self.fre_scale, self.time_scale, self.dataset.receiverNum), [item], self.isNormal)
            data1 = np.abs(data1.reshape([1, self.dataset.receiverNum, self.fre_scale, self.time_scale]))
        data1 = np.squeeze(data1)

        if isGetFsFc:
            data2 = np.zeros(6 * self.dataset.receiverNum + 2)
            data2[:-2] = self.dataset.GetRecPosVel(1, (6 * self.dataset.receiverNum,), [item], self.isNormal)
            data2[-2] = self.dataset.information[item, 2 * self.dataset.emitterNum + 4]
            data2[-1] = self.dataset.information[item, 4]
        else:
            data2 = self.dataset.GetRecPosVel(1, (6 * self.dataset.receiverNum,), [item], self.isNormal)
        data2 = np.squeeze(data2)

        if self.time_fre_trans is not None and self.dataType == 'IQ':
            sampleTime = self.dataset.information[item, 2 * self.dataset.emitterNum + 3]
            data1 = self.time_fre_trans(data1, self.fre_scale, sampleTime)
            if self.isNormal:
                for i in range(data1.shape[0]):
                    if data1[i].max() != 0:
                        data1[i] = data1[i] / data1[i].max()

        if self.transform is not None:
            data1, data2[:6 * self.dataset.receiverNum] = self.transform(data1, data2[:6 * self.dataset.receiverNum])

        temp = self.dataset.GetEmiPosVel(1, (6 * self.dataset.emitterNum,), [item], isNormal=self.isNormal)
        label = np.zeros([self.dataset.emitterNum, self.coordDim + 1])
        for i in range(self.dataset.emitterNum):
            label[i, :self.coordDim] = temp[0, 6 * i: 6 * i + self.coordDim]
            # confidence
            if np.sum(label[i]) == 0:
                label[i, -1] = 0
            else:
                label[i, -1] = 1

        # sort
        distence = np.sum(np.square(label), 1)
        sort_index = np.argsort(distence)
        label = label[sort_index]

        if self.isNormal:
            if isGetFsFc:
                data2[-1] /= 161975000
                data2[-2] /= 19200

        return [data1, data2], label.flatten()
