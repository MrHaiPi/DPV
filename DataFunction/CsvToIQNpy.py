import os

import numpy as np
import DataIO

# SNR用于确定目标文件夹
SnrRange = [25, 25]

# 数据集根目录
dataFileRoot = r"E:\资料\研究生\课题\射频定位\code\Dataset/test/SNR" + str(SnrRange)

for n in ['train', 'valid']:
    print('\n>processing {} data'.format(n))
    dataPath = dataFileRoot + "/" + n

    # 如果没有文件夹就创建
    DataIO.MakeDir(dataPath + '/Npy/IQ')

    # 删除先前文件
    DataIO.DelFile(dataPath + '/Npy/IQ')

    # 读取标签文件
    if not os.path.exists(dataPath + '/Csv/IQ/Information.csv'):
        continue

    labels = np.array(DataIO.ReadCSV(dataPath + '/Csv/IQ/Information.csv')[1:])#不要表头
    np.save(dataPath + '/Npy/IQ/Information.npy', labels)

    # 加载文件夹名称
    dirs = DataIO.GetDirName(dataPath + '/Csv/IQ')

    num = 0
    for dir in dirs:
        # 生成文件夹
        DataIO.MakeDir(dataPath + '/Npy/IQ/' + dir)
        # 加载文件夹中的文件名称
        dataFileNames = DataIO.GetFileName(dataPath + '/Csv/IQ/' + dir)
        for dataFileName in dataFileNames:
            # 获取IQ数据
            data = np.array(DataIO.ReadCSV(dataPath + '/Csv/IQ/' + dir + '/' + dataFileName)[1:])  # 不要表头
            I = np.array([float(x) for x in data[:, 0]])
            Q = np.array([float(x) for x in data[:, 1]])
            IQ = I + Q * 1j

            np.save(dataPath + '/Npy/IQ/' + dir + '/' + dataFileName[:-4] + '.npy', IQ)

            num += 1
            print('\r' + "MakingData:", format(num / dirs.__len__(), '.1f'), end="", flush=True)
