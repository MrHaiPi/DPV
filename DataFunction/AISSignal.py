# 生成AIS信号模块
# 可以处理连续时间

import numpy as np
from math import sqrt, pi, log, erfc
from scipy import interpolate
import pylab as plt

TRANING_LENGTH = 24
RISING_LENGTH = 8
BUFFER_LENGTH = 24

AIS_BAUD_RATE = 9600

def genAISData():
    dataLen = 184 # data length, including CRC bits
    crcLen = 16
    infoLen = dataLen - crcLen
    zeroNum = 0

    # 随机生成AIS数据bit
    info = np.round(np.random.rand(infoLen)).astype(np.uint8)
    #info = np.ones(infoLen).astype(np.uint8)  # for test

    # 对数据bit进行插0操作
    i = 0
    while i < len(info):
        if int(np.sum(info[i : i + 5])) == 5:
            if zeroNum < 4:  # 最多补入4个0，多余5连1需要修改随机生成的数据
                info = np.insert(info, i + 5, 0)
                zeroNum += 1
            else:
                info[i + 4] = 0
            i = i + 5
        i += 1
    # print("origin data: ", info)

    # 增加CRC校验bit
    data = np.zeros(dataLen + 4).astype(np.uint8)
    data[0: len(info)] = info
    data[len(info): len(info) + 16] = crc16(info)
    # print("crc: ", crc16(info))
    return data


def crc16(data):
    crcNo = 16
    crcGenerator = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], np.uint8)  # D^16+D^12+D^5+1
    dataLen = data.shape[0]

    output = np.zeros(dataLen + crcNo).astype(np.uint8)
    output[0: crcNo] = np.mod(data[0: crcNo] + 1, 2)  # x^16*G(x)+x^k(x^15+x^14+...+x+1)
    output[crcNo: dataLen] = data[crcNo: dataLen]

    for _ in range(dataLen):
        if (int(output[0]) == 1):
            output[0: crcNo+1] = np.mod(output[0: crcNo+1] + crcGenerator, 2)
        output = np.hstack((output[1: ], output[0]))
    output[0: crcNo] = np.mod(output[0: crcNo] + 1, 2)  # FCS是余数的反码
    return output[0: crcNo]


def gmskMod(data, os, Rb, BT, L, Kf):
    # data: 原信号，numpy向量
    # os: oversample 抽样倍率
    # Rb: 码元速率
    # BT: 带宽时延积
    # L: 高斯滤波器点数, 一般取3即可
    # # Kf: 调制系数, 一般取0.5 

    Tb = 1.0 / Rb
    B = BT / Tb
    Fs = Rb * os
    dataLen = data.shape[0]

    t = np.arange(-L*os/2, L*os/2, 1) / Fs
    coef = sqrt(2 / log(2)) * pi * B
    tempNeg = np.vectorize(erfc)(coef * (t-Tb/2))
    tempPos = np.vectorize(erfc)(coef * (t+Tb/2))
    gt = (tempNeg - tempPos) / (4*os)

    # data_high = reshape([data; zeros(os-1, len)], 1, len*os);
	# phase = 2*pi*Kf*cumsum(conv(gt, data_high(1:end-os+1)));
    overSampleData = np.vstack((data, np.zeros((os-1, dataLen))))
    dataHigh = np.reshape(overSampleData, (1, dataLen*os), order='F')
    dataConv = np.convolve(gt, dataHigh[0, 0:-os+1])
    phase = 2 * pi * Kf * np.cumsum(dataConv)

    # truncate
    phase = phase[(L-1)//2*os : (dataLen+(L-1)//2)*os]
    return np.exp(1j * phase)


def genAISModul(data, os):
    # 采样率 = os * 9600
    def NRZIEncode(data):
        encoded = np.array([], np.uint8)
        current_element = 0
        for element in data:
            if element == 1:
                current_element = (current_element + 1) % 2
            encoded = np.append(encoded, current_element)
        return encoded

    training = np.fromfunction(lambda i: i % 2, (24,)).astype(np.uint8)
    startFlag = np.array([0, 1, 1, 1, 1, 1, 1, 0], np.uint8)
    endFlag = np.array([0, 1, 1, 1, 1, 1, 1, 0], np.uint8)
    data = np.hstack((training, startFlag, data, endFlag))

    # NRZI编码
    NRZIData = 2 * NRZIEncode(data) - 1

    # GMSK调制
    gsmkData = gmskMod(NRZIData, os, AIS_BAUD_RATE, 0.4, 3, 0.5)
    gsmkData = np.hstack((np.zeros(RISING_LENGTH * os), gsmkData, np.zeros((BUFFER_LENGTH - 4) * os)))

    return NRZIData, gsmkData


def getAISModulSig(time):
    _, gmsk = genAISModul(genAISData())
    f_abs = interpolate.interp1d(np.arange(len(gmsk)), np.absolute(gmsk))
    f_ang = interpolate.interp1d(np.arange(len(gmsk)), np.angle(gmsk))
    # return gmsk[time]
    return f_abs(time) * np.exp(1j * f_ang(time))


if __name__ == "__main__":
    data = genAISData()
    nrzi, gmsk = genAISModul(data, 3)
    plt.figure()
    plt.plot(nrzi)

    plt.figure()
    plt.plot(gmsk)
    plt.show()

    # print(getAISModulSig(100.1))