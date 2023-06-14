import numpy as np
import math
import AISSignal

def Random(time, samplingRate):
    # 载波频率
    fc = 161.975 * 1e6

    r = np.random.random(math.floor(time * samplingRate))
    angle = np.random.random(math.floor(time * samplingRate)) * 2 * np.pi / 4
    signal = r * np.cos(angle) + 1j * r * np.sin(angle)

    return signal, fc

def Ais(time, samplingRate):
    # 载波频率
    fc = 161.975 * 1e6

    # AIS帧格式，共256个bit，码元速率为9600，即一个AIS信号的时间为256/9600秒，采样点数必须为256的整数倍
    # 上升沿|对准序列|开始标志|数据   |帧校验序列|结束标志|缓冲  |
    # 8bits|24bits |8bits  |168bits|16bits   |8bits  |24bits|
    data = AISSignal.genAISData()
    signal = None
    if time < 256 / 9600:
        nrzi, gmsk = AISSignal.genAISModul(data, math.floor(256 / 9600 * samplingRate / 256))
        signal = gmsk[:int(time * samplingRate)]
    else:
        nrzi, gmsk = AISSignal.genAISModul(data, math.floor(time * samplingRate / 256))
        signal = gmsk

    return signal, fc

# 接收信号仿真
def ReceSignalSim(time, samplingRate, emitterPos, emitterVel, receiverPos, receiverVel, SNR, attenRange, asynchrony_par_sigma):
    '''
    :param time:采样时间，单位s
    :param samplingRate:采样频率，单位Hz
    :param emitterPos: 发射器位置，单位m，内容为[发射器编号，x坐标，y坐标，z坐标]
    :param emitterVel: 发射器速度，单位m/s，内容为[x分量，y分量，z分量]
    :param receiverPos: 接收器位置，单位m，内容为[x坐标，y坐标，z坐标]
    :param receiverVel: 接收器速度，单位m/s，内容为[x分量，y分量，z分量]
    :param SNR:信噪比，单位dB
    :param attenRange:多目标混叠时候的衰减系数范围
    :param asynchrony_par_sigma:多接收机之间不同步参数的标准差
    :return:oriSignal原始信号，内容为[发射器编号，数据]；simSignal模拟接收的信号，内容为[发射器编号，数据]；t延时，每一行信号的延时；f延频，每一行信号的延频
    '''

    recSignals = []
    emitSignals = []
    deltaTs = []
    deltaFs = []
    fcs = []

    for emitter_num in range(emitterPos.shape[0]):

        # 光速，单位m/s
        c = 299792458

        # 加载信号
        data, fc = Ais(time, samplingRate)
        a = np.zeros(data.shape, dtype=complex)
        a[:] = data[:]

        # 载波
        #data = data * np.exp(1j * 2 * np.pi * fc * np.array(range(data.shape[1])) / samplingRate)

        signal = a.copy()
        recSignal = np.zeros([receiverPos.shape[0], signal.shape[0]], dtype='complex')
        emitSignal = np.zeros([receiverPos.shape[0], signal.shape[0]], dtype='complex')

        for i in range(emitSignal.shape[0]):
            emitSignal[i, :] = signal.copy()

        # 发射器与接收器相对位置与速度
        p = emitterPos[emitter_num] - receiverPos
        v = receiverVel - emitterVel[emitter_num]

        deltaT = np.sqrt(np.sum(p * p, 1)) / c
        deltaF = fc / c * np.sum(p * v, 1) / np.sqrt(np.sum(p * p, 1))

        # 生成时频不同步的参数
        deltaT_error = np.random.normal(loc=0, scale=asynchrony_par_sigma[0], size=(deltaT.shape[0]))
        deltaF_error = np.random.normal(loc=0, scale=asynchrony_par_sigma[1], size=(deltaF.shape[0]))

        # 考虑时频不同步，默认第一个接收机的时间频率为标准
        deltaT[1:] = deltaT[1:] + deltaT_error[1:]
        deltaF[1:] = deltaF[1:] + deltaF_error[1:]

        # 先模拟频延
        for i in range(recSignal.shape[0]):
            fDely = np.exp(1j * 2 * np.pi * deltaF[i] * np.array(range(recSignal.shape[1])) / samplingRate)
            recSignal[i, :] = emitSignal[i, :] * fDely

        # 后模拟时延
        for i in range(recSignal.shape[0]):
            if round(deltaT[i] * samplingRate) == 0:
                continue
            else:
                recSignal[i, round(deltaT[i] * samplingRate):] = recSignal[i, :-round(deltaT[i] * samplingRate)]
                recSignal[i, :round(deltaT[i] * samplingRate)] = 0

        # 添加噪声
        for i in range(recSignal.shape[0]):

            # 计算信号的功率(归一化)
            #signalPower = np.linalg.norm(recSignal[i] - recSignal[i].mean()) ** 2 / recSignal[i].shape[0]
            signalPower = recSignal[i].std() ** 2 # np.mean(np.abs(recSignal[i]) ** 2)
            # 计算噪声的功率
            noisePower = signalPower / (np.power(10, SNR[emitter_num] / 10))

            # 生成噪声
            aa = np.random.random() * 10000
            noiseI = np.random.randn(recSignal[i].shape[0]) * aa
            noiseQ = np.random.randn(recSignal[i].shape[0]) * aa
            noise = noiseI + 1j * noiseQ
            noise = noise - np.mean(noise)
            noise = (np.sqrt(noisePower) / np.std(noise)) * noise

            # Ps = (np.linalg.norm(recSignal[i] - recSignal[i].mean())) ** 2  # signal power
            # Pn = (np.linalg.norm(noise - noise.mean())) ** 2  # noise power
            # snr = 10 * np.log10(Ps / Pn)

            recSignal[i] += noise

        emitSignals.append(emitSignal)
        recSignals.append(recSignal)
        deltaTs.append(deltaT)
        deltaFs.append(deltaF)
        fcs.append(fc)

    recSignal = None
    for i in range(recSignals.__len__()):
        if recSignal is None:
            recSignal = recSignals[i]
        else:
            real = np.random.random() * (attenRange[1] - attenRange[0]) + attenRange[0]
            img = 1j * np.random.random() * (attenRange[1] - attenRange[0]) + attenRange[0]
            recSignal += (real + img) * recSignals[i]

    return emitSignals, recSignal, deltaTs, deltaFs, fcs



# 接收信号仿真
def ReceSignalSim1(time, samplingRate, emitterPos, emitterVel, receiverPos, receiverVel, Snr):
    '''
    :param time:采样时间，单位s
    :param samplingRate:采样频率，单位Hz
    :param emitterPos: 发射器位置，单位m，内容为[发射器编号，x坐标，y坐标，z坐标]
    :param emitterVel: 发射器速度，单位m/s，内容为[x分量，y分量，z分量]
    :param receiverPos: 接收器位置，单位m，内容为[x坐标，y坐标，z坐标]
    :param receiverVel: 接收器速度，单位m/s，内容为[x分量，y分量，z分量]
    :param SNR:信噪比，单位dB
    :return:oriSignal原始信号，内容为[发射器编号，数据]；simSignal模拟接收的信号，内容为[发射器编号，数据]；t延时，每一行信号的延时；f延频，每一行信号的延频

    # 将接收机的信号移至中央, 这样可以除去信号的起始时间信息
    '''

    recSignals = []
    emitSignals = []
    deltaTs = []
    deltaFs = []
    fcs = []

    for emitter_num in range(emitterPos.shape[0]):
        # 光速，单位m/s
        c = 299792458

        # 加载信号
        data, fc = Ais(time, samplingRate)

        # 信号左右添加0
        paddLength = int(0.1 * samplingRate)
        data = np.append(np.zeros(paddLength, dtype=complex), data)
        data = np.append(data, np.zeros(paddLength, dtype=complex))

        # 载波
        # data = data * np.exp(1j * 2 * np.pi * fc * np.array(range(data.shape[0])) / samplingRate)

        signal = data.copy()
        recSignal = np.zeros([receiverPos.shape[0], signal.shape[0]], dtype='complex')
        emitSignal = np.zeros([receiverPos.shape[0], signal.shape[0]], dtype='complex')

        for i in range(emitSignal.shape[0]):
            emitSignal[i, :] = signal.copy()

        # 发射器与接收器相对位置与速度
        p = emitterPos[emitter_num] - receiverPos
        v = receiverVel - emitterVel[emitter_num]

        deltaT = np.sqrt(np.sum(p * p, 1)) / c

        if deltaT.max() - deltaT.min() >= time:
            print('采样时间过短!')
            exit()

        deltaF = fc / c * np.sum(p * v, 1) / np.sqrt(np.sum(p * p, 1))

        # 先模拟频延
        for i in range(recSignal.shape[0]):
            fDely = np.exp(1j * 2 * np.pi * deltaF[i] * np.array(range(recSignal.shape[1])) / samplingRate)
            recSignal[i, :] = emitSignal[i, :] * fDely

        # 后模拟时延
        for i in range(recSignal.shape[0]):
            if math.floor(deltaT[i] * samplingRate) == 0:
                continue
            else:
                recSignal[i, math.floor(deltaT[i] * samplingRate):] = recSignal[i, :-math.floor(deltaT[i] * samplingRate)]
                recSignal[i, :math.floor(deltaT[i] * samplingRate)] = recSignal[i, -math.floor(deltaT[i] * samplingRate):]

        # 将接收机的信号移至中央, 这样可以除去信号的起始时间信息
        endIndex = int(np.ceil(time * samplingRate))
        left = int(np.ceil(deltaT.min() * samplingRate))
        emitSignal = emitSignal[:, paddLength:paddLength + endIndex]
        recSignal = recSignal[:, int(paddLength + left):int(paddLength + left + endIndex)]

        # 添加噪声
        for i in range(recSignal.shape[0]):
            # 计算信号的功率(归一化)
            signalPower = np.linalg.norm(recSignal[i] - recSignal[i].mean()) ** 2 / recSignal[i].shape[0]
            # 计算噪声的功率
            noisePower = signalPower / (np.power(10, Snr[emitter_num] / 10))

            # 生成噪声
            noiseI = np.random.randn(recSignal[i].shape[0])
            noiseQ = np.random.randn(recSignal[i].shape[0])
            noise = noiseI + 1j * noiseQ
            noise = noise - np.mean(noise)
            noise = (np.sqrt(noisePower) / np.std(noise)) * noise

            # Ps = (np.linalg.norm(recSignal[i] - recSignal[i].mean())) ** 2  # signal power
            # Pn = (np.linalg.norm(noise - noise.mean())) ** 2  # noise power
            # snr = 10 * np.log10(Ps / Pn)

            recSignal[i] += noise

        emitSignals.append(emitSignal)
        recSignals.append(recSignal)
        deltaTs.append(deltaT)
        deltaFs.append(deltaF)
        fcs.append(fc)

    recSignal = None
    for i in range(recSignals.__len__()):
        if recSignal is None:
            recSignal = recSignals[i]
        else:
            recSignal += recSignals[i]

    return emitSignals, recSignal, deltaTs, deltaFs, fcs


