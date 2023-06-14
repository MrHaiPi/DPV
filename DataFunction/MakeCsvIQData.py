import os.path

from ReceSignalSim import ReceSignalSim, ReceSignalSim1
import numpy as np
import DataIO
import pylab as plt
np.random.seed(1)
# 一般最低使用8000个数据点
# 采样时间
samplingTime = (64 + 32) / 9600
# 采样频率
samplingRate = 9600 * 2

# 信噪比范围，单位dB
SnrRange = [25, 25]

# 数据保存根目录
dataFileRoot = r"E:\资料\研究生\课题\射频定位\code\Dataset\SignalDataset\SNR" + str(SnrRange)

# 生成数据个数
dataNum = 1000

# 验证集所占的比例
validRate = 1

# 接收机数量范围
receiverNumRange = [3, 3]
# 发射机数量范围
emitterNumRange = [1, 1]

# 卫星位置范围，单位m
receiverPosRange = 1e6 * np.array([[0, 1], [0, 1], [0.6, 1]])

# 卫星速度范围，单位m/s
# 物理常数G，单位(N·m^2 /kg^2)；地球质量M，单位KG，地球半径R，单位m
R = 6371000
GM = 6.67 * 5.965 * 1e13
receiverVelRange = [np.sqrt(GM / (receiverPosRange[2][1] + R)),
                    np.sqrt(GM / (receiverPosRange[2][0] + R))]  # [7346.9 7554.6]

# 发射器位置范围，单位m
emitterPosRange = 1e6 * np.array([[0, 1], [0, 1], [0, 0]])
# 发射器速度范围，单位m/s
emitterVelRange = [0, 30]

# 信号衰减系数范围
attenRange = [0.5, 1]

# 接收机不同步造成的误差
# 时间不同步的标准差/s
sigma_t = 0#1e-5
# 频率不同步的标准差/Hz
sigma_f = 0
# 不同步的参数
asynchrony_par_sigma = [sigma_t, sigma_f]

# 保存信号数据的标签
signalLabelsHeaders = ['num', 'receiverNum', 'emitterNum']
for i in range(emitterNumRange[1]):
    signalLabelsHeaders.append('snr' + str(i))
    signalLabelsHeaders.append('fc' + str(i))

signalLabelsHeaders.extend(
    ['samplingTime', 'samplingRate',
     'receiverPosRange_x_min', 'receiverPosRange_x_max',
     'receiverPosRange_y_min', 'receiverPosRange_y_max',
     'receiverPosRange_z_min', 'receiverPosRange_z_max',
     'receiverVelRange_min', 'receiverVelRange_max',
     'emitterPosRange_x_min', 'emitterPosRange_x_max',
     'emitterPosRange_y_min', 'emitterPosRange_y_max',
     'emitterPosRange_z_min', 'emitterPosRange_z_max',
     'emitterVelRange_min', 'emitterVelRange_max'])

for i in range(receiverNumRange[1]):
    for j in range(emitterNumRange[1]):
        signalLabelsHeaders.append('receiver' + str(i) + 'DelyT' + str(j))
        signalLabelsHeaders.append('receiver' + str(i) + 'DelyF' + str(j))

for j in range(receiverNumRange[1]):
    signalLabelsHeaders.append('receiver' + str(j) + 'Posx')
    signalLabelsHeaders.append('receiver' + str(j) + 'Posy')
    signalLabelsHeaders.append('receiver' + str(j) + 'Posz')
    signalLabelsHeaders.append('receiver' + str(j) + 'Velx')
    signalLabelsHeaders.append('receiver' + str(j) + 'Vely')
    signalLabelsHeaders.append('receiver' + str(j) + 'Velz')
for j in range(emitterNumRange[1]):
    signalLabelsHeaders.append('emitterPos' + str(j) + 'x')
    signalLabelsHeaders.append('emitterPos' + str(j) + 'y')
    signalLabelsHeaders.append('emitterPos' + str(j) + 'z')
    signalLabelsHeaders.append('emitterVel' + str(j) + 'x')
    signalLabelsHeaders.append('emitterVel' + str(j) + 'y')
    signalLabelsHeaders.append('emitterVel' + str(j) + 'z')

for n in ['train', 'valid']:
    print('\n>processing {} data'.format(n))
    dataPath = dataFileRoot + "/" + n

    # 如果没有文件夹就创建
    DataIO.MakeDir(dataPath + "/Csv" + '/IQ')

    # 删除先前文件夹中的所有文件
    DataIO.DelFile(dataPath + "/Csv" + '/IQ')

    if n == 'train':
        genDataNum = int(dataNum * (1 - validRate))
    else:
        genDataNum = int(dataNum * validRate)

    for i in range(genDataNum):

        # 接收器数量
        receiverNum = int(np.random.random() * (receiverNumRange[1] + 1 - receiverNumRange[0]) + receiverNumRange[0])
        # 发射器的数量
        emitterNum = int(np.random.random() * (emitterNumRange[1] + 1 - emitterNumRange[0]) + emitterNumRange[0])

        # 生成接收器位置与速度
        receiverPos = np.zeros([receiverNum, 3])
        # 每次数据3个接收器的高度是一样的
        recePosZ = (receiverPosRange[2][1] - receiverPosRange[2][0]) * np.random.random() + receiverPosRange[2][0]
        receiverVel = np.zeros(receiverPos.shape)
        for j in range(receiverNum):
            recePosX = (receiverPosRange[0][1] - receiverPosRange[0][0]) * np.random.random() + receiverPosRange[0][0]
            recePosY = (receiverPosRange[1][1] - receiverPosRange[1][0]) * np.random.random() + receiverPosRange[1][0]
            receiverPos[j][0] = recePosX
            receiverPos[j][1] = recePosY
            receiverPos[j][2] = recePosZ

            receVel = (receiverVelRange[1] - receiverVelRange[0]) * np.random.random() + receiverVelRange[0]
            theta = np.random.random() * np.pi / 2
            fa = np.random.random() * np.pi * 2
            receVelZ = receVel * np.cos(theta)
            receVelX = receVel * np.sin(theta) * np.cos(fa)
            receVelY = receVel * np.sin(theta) * np.sin(fa)
            receiverVel[j][0] = receVelX
            receiverVel[j][1] = receVelY
            receiverVel[j][2] = receVelZ

        # 生成发射器位置与速度
        emitterPos = np.zeros([emitterNum, 3])
        emitterVel = np.zeros(emitterPos.shape)
        for j in range(emitterNum):
            emitPosX = (emitterPosRange[0][1] - emitterPosRange[0][0]) * np.random.random() + emitterPosRange[0][0]
            emitPosY = (emitterPosRange[1][1] - emitterPosRange[1][0]) * np.random.random() + emitterPosRange[1][0]
            emitPosZ = (emitterPosRange[2][1] - emitterPosRange[2][0]) * np.random.random() + emitterPosRange[2][0]
            emitterPos[j][0] = emitPosX
            emitterPos[j][1] = emitPosY
            emitterPos[j][2] = emitPosZ

            emitVel = (emitterVelRange[1] - emitterVelRange[0]) * np.random.random() + emitterVelRange[0]
            theta = np.pi / 2
            fa = np.random.random() * np.pi * 2
            emitVelZ = emitVel * np.cos(theta)
            emitVelX = emitVel * np.sin(theta) * np.cos(fa)
            emitVelY = emitVel * np.sin(theta) * np.sin(fa)
            emitterVel[j][0] = emitVelX
            emitterVel[j][1] = emitVelY
            emitterVel[j][2] = emitVelZ

        # emitterPos = 1e6 * np.array([[0.3, 0.5, 0], [0.3, 0.503, 0]])
        # emitterVel = np.array([[5, 50, 0]])
        # receiverPos = 1e6 * np.array([[0.5, 0.25, 0.6], [0.05, 0.95, 0.6], [0.95, 0.95, 0.6]])
        if receiverPos.shape[0] == 3:
            receiverPos = 1e6 * np.array([[0.25, 0.75, 0.6], [0.5, 0.25, 0.6], [0.75, 0.75, 0.6]])
            receiverVel = np.array([[5500, 5500, 0], [5500, 5500, 0], [5500, 5500, 0]])
        elif receiverPos.shape[0] == 4:
            receiverPos = 1e6 * np.array([[0.3, 0.3, 0.6], [0.6, 0.3, 0.6], [0.6, 0.6, 0.6], [0.3, 0.6, 0.6]])
            receiverVel = np.array([[5500, 5500, 0], [5500, 5500, 0], [5500, 5500, 0], [5500, 5500, 0]])
        # receiverVel = np.array([[5500, 5500, 0], [5500, 5500, 0], [5500, 5500, 0]])
        # receiverPos = 1e6 * np.array([[0.5, 0.25, 0.], [0.05, 0.95, 0.], [0.95, 0.95, 0.]])
        # receiverVel = np.array([[0, 7500, 0], [0, 7500, 0], [5500, 5500, 0]])

        # 卫星和发射器位置图绘制
        '''
        x = receiverPos[:, 0];#x = np.append(x, [emitterPos[0, 0]])
        y = receiverPos[:, 1];#y = np.append(y, [emitterPos[0, 1]])
        z = receiverPos[:, 2];#z = np.append(z, [emitterPos[0, 2]])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, c='black', marker='1', s=300)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
        direction = receiverVel
        ax.quiver(x, y, z,
                  direction[:, 0], direction[:, 1], direction[:, 2],
                  length=20, normalize=False, color='r')

        ax.set_xlabel('x/m')
        ax.set_ylabel('y/m')
        ax.set_zlabel('z/m')
        ax.set_xlim(0, 1e6)
        ax.set_ylim(0, 1e6)
        ax.set_zlim(0, 1e6)
        plt.grid(True)
        # plt.legend()
        plt.show()
        '''

        # 信噪比
        Snrs = []
        for num in range(emitterNum):
            Snrs.append((SnrRange[1] - SnrRange[0]) * np.random.random() + SnrRange[0])

        # 生成模拟信号
        emitSignals, receSignal, deltaTs, deltaFs, fcs = ReceSignalSim(samplingTime, samplingRate, emitterPos, emitterVel,
                                                                        receiverPos,
                                                                        receiverVel, Snrs, attenRange, asynchrony_par_sigma)
        # 绘制信号的时频图
        isPlot = False
        if isPlot:
            for num in range(emitterNum):
                plt.figure()
                t = np.arange(0, emitSignals[num].shape[1]) / samplingRate
                plt.plot(t, abs(emitSignals[num][0]))
                center = int(np.ceil(deltaTs[num].mean() * samplingRate))

                # t = np.arange(center - emitSignals[num].shape[1] / 2, center + emitSignals[num].shape[1] / 2) / samplingRate
                plt.plot(t, abs(receSignal[0]), label='receiver1Data')
                plt.plot(t, abs(receSignal[1]), label='receiver2Data')
                plt.plot(t, abs(receSignal[2]), label='receiver3Data')
                plt.xlabel('t/s', fontsize=15)
                plt.ylabel('Amplitude', fontsize=15)
                plt.tick_params(labelsize=15)
                # plt.legend()
                # 绘制信号的频域图
                plt.figure()
                plt.magnitude_spectrum(emitSignals[num][0], Fs=samplingRate)
                plt.magnitude_spectrum(receSignal[0], Fs=samplingRate, label='receiver1Data')
                plt.magnitude_spectrum(receSignal[1], Fs=samplingRate, label='receiver2Data')
                plt.magnitude_spectrum(receSignal[2], Fs=samplingRate, label='receiver3Data')
                plt.xlabel('Frequency/Hz', fontsize=15)
                plt.ylabel('Magnitude(energy)', fontsize=15)
                print("TimeDelay：", deltaTs[num])
                print("FreDelay：", deltaFs[num])
                # plt.legend()
                plt.tick_params(labelsize=15)
                plt.show()

        # 保存数据标签
        fileName = dataPath + "/Csv" + '/IQ' + '/Information.csv'
        signalLabel = [i, receiverNum, emitterNum]
        for j in range(emitterNumRange[1]):
            if j < emitterNum:
                signalLabel.append(Snrs[j])
                signalLabel.append(fcs[j])
            else:
                signalLabel.append(0)
                signalLabel.append(0)
        signalLabel.extend([samplingTime, samplingRate,
                            receiverPosRange[0][0], receiverPosRange[0][1],
                            receiverPosRange[1][0], receiverPosRange[1][1],
                            receiverPosRange[2][0], receiverPosRange[2][1],
                            receiverVelRange[0], receiverVelRange[1],
                            emitterPosRange[0][0], emitterPosRange[0][1],
                            emitterPosRange[1][0], emitterPosRange[1][1],
                            emitterPosRange[2][0], emitterPosRange[2][1],
                            emitterVelRange[0], emitterVelRange[1]])

        for j in range(receiverNumRange[1]):
            for k in range(emitterNumRange[1]):
                if j < receiverNum and k < emitterNum:
                    #signalLabel = signalLabel + [deltaTs[k][j] / samplingTime, deltaFs[k][j] / (samplingRate / 2)]
                    signalLabel = signalLabel + [deltaTs[k][j], deltaFs[k][j]]
                else:
                    signalLabel = signalLabel + [0, 0]

        for j in range(receiverNumRange[1]):
            if j < receiverNum:
                signalLabel = signalLabel + [receiverPos[j][0], receiverPos[j][1], receiverPos[j][2], receiverVel[j][0], receiverVel[j][1], receiverVel[j][2]]
                # signalLabel = signalLabel + [receiverPos[j][0] / receiverPosRange[0][1],
                #                              receiverPos[j][1] / receiverPosRange[1][1],
                #                              receiverPos[j][2] / receiverPosRange[2][1],
                #                              receiverVel[j][0] / receiverVelRange[1], receiverVel[j][1] / receiverVelRange[1],
                #                              receiverVel[j][2] / receiverVelRange[1]]
            else:
                signalLabel = signalLabel + [0, 0, 0, 0, 0, 0]
        for j in range(emitterNumRange[1]):
            if j < emitterNum:
                signalLabel = signalLabel + [emitterPos[j][0], emitterPos[j][1], emitterPos[j][2], emitterVel[j][0] , emitterVel[j][1], emitterVel[j][2]]
                # signalLabel = signalLabel + [emitterPos[j][0] / emitterPosRange[0][1], emitterPos[j][1] / emitterPosRange[1][1],
                #                              emitterPos[j][2], emitterVel[j][0] / emitterVelRange[1],
                #                              emitterVel[j][1] / emitterVelRange[1], emitterVel[j][2]]
            else:
                signalLabel = signalLabel + [0, 0, 0, 0, 0, 0]
        if i == 0:
            DataIO.WriteCSV([signalLabel], signalLabelsHeaders, fileName)
        else:
            DataIO.WriteCSV([signalLabel], [], fileName, True)

        # 保存发射器的信号数据
        for k in range(emitterNumRange[1]):
            # 保存接收器的信号数据
            path = dataPath + "/Csv" + '/IQ' + '/Emitter' + str(k)
            DataIO.MakeDir(path)
            fileName = path + '/' + str(i) + '.csv'
            if k < emitterNum:
                I = emitSignals[k][0].real
                I = I.reshape([I.__len__(), 1])
                Q = emitSignals[k][0].imag
                Q = Q.reshape([Q.__len__(), 1])
            else:
                I = [[0]]
                Q = [[0]]
            DataIO.WriteCSV(np.append(I, Q, 1), ['I', 'Q'], fileName)

        # 保存接收信号的数据
        for k in range(receiverNumRange[1]):
            # 保存接收器的信号数据
            path = dataPath + "/Csv" + '/IQ' + '/Receiver' + str(k)
            DataIO.MakeDir(path)
            fileName = path + '/' + str(i) + '.csv'
            if k < receiverNum:
                I = receSignal[k].real
                I = I.reshape([I.__len__(), 1])
                Q = receSignal[k].imag
                Q = Q.reshape([Q.__len__(), 1])
            else:
                I = [[0]]
                Q = [[0]]
            DataIO.WriteCSV(np.append(I, Q, 1), ['I', 'Q'], fileName)

        print('\r' + "MakingData:{}/{}".format(i + 1, genDataNum), end="", flush=True)
