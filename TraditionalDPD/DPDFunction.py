import numpy as np
import matplotlib.pyplot as plt

def MLDPD(fc, samplingRate, emitterPos, emitterVel, receiverPos, receiverVel, receiveSignal, emitterSignal=None):
    '''
    :param fc: 载波频率
    :param samplingRate: 采样频率
    :param emitterPos: 发射机位置[px,py,pz]
    :param emitterVel: 发射机速度[vx,vy,vz]
    :param receiverPos: 接收机位置[[p0x,p0y,p0z],[p1x,p1y,p1z],...]
    :param receiverVel: 接收机速度[[v0x,v0y,v0z],[v1x,v1y,v1z],...]
    :param receiveSignal: 接收的信号[[s1(t1),s1(t2),... ],[s2(t1),s2(t2),... ],...]
    :param emitterSignal: 发射机的信号[[s1(t1),s1(t2),... ]]
    :return:
    '''
    # 光速，单位m/s
    c = 299792458

    # 发射器与接收器相对位置与速度
    p = emitterPos - receiverPos
    v = receiverVel - emitterVel

    # 计算时延与频延
    t = np.sqrt(np.sum(p * p, 1)) / c
    if emitterSignal is None:
        t = t - t.min()  # 防止数据点过少导致根据时差平移时超过数据长度
    f = fc / c * np.sum(p * v, 1) / np.sqrt(np.sum(p * p, 1))

    ## V矩阵计算
    objV = receiveSignal.copy()
    V = None
    # 计算方式
    ways = 0
    if ways == 0:
        # 方式一,此方式实际是方式二的物理意义
        # 先补偿时延
        for i in range(objV.shape[0]):
            if round(t[i] * samplingRate) != 0:
                objV[i] = np.hstack((objV[i, round(t[i] * samplingRate):], objV[i, :round(t[i] * samplingRate)]))

        # 再补偿频延
        for i in range(objV.shape[0]):
            fDely = np.exp(-1j * 2 * np.pi * f[i] * np.arange(objV.shape[1]) / samplingRate)
            objV[i] = objV[i] * fDely

        V = (objV.T).copy()
    else:
        # 方式二, 此方式与方式一原理上没有任何区别，仅仅是为了更符合数学表达式，计算速度缓慢，实际建议使用方式一
        V = np.zeros([objV.shape[1], objV.shape[0]], dtype=complex)
        for i in range(objV.shape[0]):
            # F = np.diag(np.exp(1j * 2 * np.pi * f[i] * np.arange(objV.shape[1]) / samplingRate))
            # T = np.diag(np.ones(objV.shape[1] - round(t[i] * samplingRate)), - round(t[i] * samplingRate))
            # V[:, i] = np.dot(np.dot(T.T.conjugate(), F.T.conjugate()), objV.T[:, i])

            # 直接写出FT的结果，减少计算量，与上式等价
            FT = np.diag(np.exp(1j * 2 * np.pi * f[i] * np.arange(round(t[i] * samplingRate), objV.shape[1]) / samplingRate), - round(t[i] * samplingRate))
            V[:, i] = np.dot(FT.T.conjugate(), objV.T[:, i])

    VH = (V.T.conjugate()).copy()
    Q = np.dot(VH, V)

    if emitterSignal is not None:
        emitterSignalCopy = emitterSignal.T.copy()
        maxLamda = np.abs(np.dot(np.dot(np.dot(emitterSignalCopy.T.conjugate(), V), VH), emitterSignalCopy))  # 信号已知
    else:
        maxLamda = np.abs(np.linalg.eig(Q)[0].max())

    return maxLamda


def MVDRDPD(fc, samplingRate, emitterPos, emitterVel, receiverPos, receiverVel, receiveSignal, emitterSignal=None):
    '''
    :param fc: 载波频率
    :param samplingRate: 采样频率
    :param emitterPos: 发射机位置[px,py,pz]
    :param emitterVel: 发射机速度[vx,vy,vz]
    :param receiverPos: 接收机位置[[p0x,p0y,p0z],[p1x,p1y,p1z],...]
    :param receiverVel: 接收机速度[[v0x,v0y,v0z],[v1x,v1y,v1z],...]
    :param receiveSignal: 接收的信号[[s1(t1),s1(t2),... ],[s2(t1),s2(t2),... ],...]
    :param emitterSignal: 发射机的信号[[s1(t1),s1(t2),... ]]
    :return:
    '''
    # 光速，单位m/s
    c = 299792458

    # 发射器与接收器相对位置与速度
    p = emitterPos - receiverPos
    v = receiverVel - emitterVel

    # 先补偿时延
    t = np.sqrt(np.sum(p * p, 1)) / c
    if emitterSignal is None:
        t = t - t.min()  # 防止数据点过少导致根据时差平移时超过数据长度

    # 再补偿频延
    f = fc / c * np.sum(p * v, 1) / np.sqrt(np.sum(p * p, 1))

    Qs = None
    maxLamda = None
    objV = receiveSignal.copy()
    if emitterSignal is not None:
        for i in range(objV.shape[0]):
            # F = np.diag(np.exp(1j * 2 * np.pi * f[i] * np.arange(objV.shape[1]) / samplingRate))
            # T = np.diag(np.ones(objV.shape[1] - round(t[i] * samplingRate)), - round(t[i] * samplingRate))
            # FT = np.dot(F, T)

            # 直接写出FT的结果，减少计算量，与上式等价
            FT = np.diag(
                np.exp(1j * 2 * np.pi * f[i] * np.arange(round(t[i] * samplingRate), objV.shape[1]) / samplingRate),
                - round(t[i] * samplingRate))

            FTH = FT.T.conjugate()
            R = np.dot(objV.T[:, i].reshape([objV.shape[1], 1]),
                        objV.T[:, i].T.conjugate().reshape([1, objV.shape[1]]))  # 不能直接取逆
            R = R + 0.001 * np.diag(np.ones(R.shape[0]))  # R不满秩，不可逆，所以可以加上很小的对角矩阵使其满秩
            R = np.linalg.inv(R)  # 原文有快速算法

            Q = np.dot(np.dot(FTH, R), FT)

            Qs = Q if Qs is None else Qs + Q

        emitterSignalCopy = emitterSignal.T.copy()
        maxLamda = 1 / np.abs(np.dot(np.dot(emitterSignalCopy.T.conjugate(), Qs), emitterSignalCopy))
    else:
        FTs = []
        for i in range(objV.shape[0]):
            # F = np.diag(np.exp(1j * 2 * np.pi * f[i] * np.arange(objV.shape[1]) / samplingRate))
            # T = np.diag(np.ones(objV.shape[1] - round(t[i] * samplingRate)), - round(t[i] * samplingRate))
            # FT = np.dot(F, T)

            # 直接写出FT的结果，减少计算量，与上式等价
            FT = np.diag(
                np.exp(1j * 2 * np.pi * f[i] * np.arange(round(t[i] * samplingRate), objV.shape[1]) / samplingRate),
                - round(t[i] * samplingRate))

            FTs.append(FT)

        for i in range(objV.shape[1]):
            Gama = np.zeros([objV.shape[0], 1], dtype=complex)
            r = np.zeros([objV.shape[0], 1], dtype=complex)
            for j in range(objV.shape[0]):
                FTobj = np.sum(FTs[j], 1)
                if FTobj[i] == 0:
                    Gama[j] = 0
                    r[j] = 0
                else:
                    Gama[j] = FTobj[i]
                    r[j] = objV[j, i - round(t[j] * samplingRate)]

            Gama /= np.sqrt(np.linalg.norm(Gama))
            Gama = np.diag(Gama[:, 0])
            R = np.dot(r, r.T.conjugate())
            R = R + 0.001 * np.diag(np.ones(R.shape[0]))  # R不满秩，不可逆，所以可以加上很小的对角矩阵使其满秩
            R = np.linalg.inv(R)

            Q = np.dot(np.dot(Gama.T.conjugate(), R), Gama)

            Qs = Q if Qs is None else Qs + Q

        maxLamda = np.abs(1 / np.linalg.eig(Qs)[0].min())
        # maxLamda = np.abs(np.linalg.eig(Qs)[0].max())

    return maxLamda
