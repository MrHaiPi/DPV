import numpy as np

# 单个发射机、两个坐标维度计算CRLB
def CRLB(emitterPos, emitterVel, emitterSignal, receiverPos, receiverVel, receivedSignal, snr, fc, samplingRate, a):
    # 光速，单位m/s
    c = 299792458

    dmdx = [0] * receiverPos.shape[0]
    dmdy = [0] * receiverPos.shape[0]
    sigma2 = (emitterSignal[0].std() ** 2) / (10 ** (snr / 10))
    for i in range(receiverPos.shape[0]):

        dtdx = 1 / c * (emitterPos[0][0] - receiverPos[i][0]) / np.linalg.norm(emitterPos[0] - receiverPos[i])
        dtdy = 1 / c * (emitterPos[0][1] - receiverPos[i][1]) / np.linalg.norm(emitterPos[0] - receiverPos[i])

        dfdx = fc / c * (receiverVel[i][0] - emitterVel[0][0]) / np.linalg.norm(emitterPos[0] - receiverPos[i]) - \
               fc / c * (emitterPos[0][0] - receiverPos[i][0]) * \
               np.sum((receiverVel[i] - emitterVel[0]) * (emitterPos[0] - receiverPos[i])) / \
               np.linalg.norm(emitterPos[0] - receiverPos[i]) ** 3
        dfdy = fc / c * (receiverVel[i][1] - emitterVel[0][1]) / np.linalg.norm(emitterPos[0] - receiverPos[i]) - \
               fc / c * (emitterPos[0][1] - receiverPos[i][1]) * \
               np.sum((receiverVel[i] - emitterVel[0]) * (emitterPos[0] - receiverPos[i])) / \
               np.linalg.norm(emitterPos[0] - receiverPos[i]) ** 3

        t = np.linalg.norm(emitterPos[0] - receiverPos[i]) / c
        f = fc / c * np.sum((receiverVel[i] - emitterVel[0]) * (emitterPos[0] - receiverPos[i])) / np.linalg.norm(emitterPos[0] - receiverPos[i])

        ts = np.arange(emitterSignal.shape[1]) / samplingRate
        F = np.diag(np.exp(1j * 2 * np.pi * f * ts))
        dFdf = 1j * 2 * np.pi * np.diag(ts * np.exp(1j * 2 * np.pi * f * ts))

        emitter_signal_with_time_delay = emitterSignal.copy()
        emitter_signal_with_time_delay[0, round(t * samplingRate):] = emitter_signal_with_time_delay[0, :-round(t * samplingRate)]
        emitter_signal_with_time_delay[0, :round(t * samplingRate)] = 0
        dsdt = np.zeros(emitter_signal_with_time_delay.shape, dtype=complex)
        dsdt[0, :-1] = emitter_signal_with_time_delay[0, 1:] - emitter_signal_with_time_delay[0, :-1]
        dsdt[0, -1] = emitter_signal_with_time_delay[0, -1]
        dsdt = -dsdt / (1 / samplingRate)

        dmdx[i] = a[i] * dfdx * np.dot(dFdf, emitter_signal_with_time_delay.T) + \
                  a[i] * dtdx * np.dot(F, dsdt.T)

        dmdy[i] = a[i] * dfdy * np.dot(dFdf, emitter_signal_with_time_delay.T) + \
                  a[i] * dtdy * np.dot(F, dsdt.T)

    dmdx = np.expand_dims(np.array(dmdx).flatten(), 1)
    dmdy = np.expand_dims(np.array(dmdy).flatten(), 1)
    J11 = 2 / sigma2 * np.real(np.dot(dmdx.T.conjugate(), dmdx))
    J12 = 2 / sigma2 * np.real(np.dot(dmdx.T.conjugate(), dmdy))
    J21 = 2 / sigma2 * np.real(np.dot(dmdy.T.conjugate(), dmdx))
    J22 = 2 / sigma2 * np.real(np.dot(dmdy.T.conjugate(), dmdy))

    J = np.zeros([2, 2])
    J[0, 0] = J11
    J[0, 1] = J12
    J[1, 0] = J21
    J[1, 1] = J22

    Jinv = np.linalg.inv(J)
    crlb = np.sqrt(Jinv[0, 0] + Jinv[1, 1]) # Jinv[0, 0]为参数x的方差估计，Jinv[1, 1]为参数y的方差估计，求和开根号则为发射机位置估计的方差

    return crlb

# 多个发射机、多个坐标维度计算CRLB，更全面的版本，兼容上面的函数
def CRLB1(emitterPos, emitterVel, emitterSignal, receiverPos, receiverVel, receivedSignal, snr, fc, samplingRate, a):
    # 光速，单位m/s
    c = 299792458

    dmdx = [[0] * receiverPos.shape[0] for _ in range(emitterPos.shape[0])]
    dmdy = [[0] * receiverPos.shape[0] for _ in range(emitterPos.shape[0])]
    sigma2 = (emitterSignal[0].std() ** 2) / (10 ** (snr / 10))
    for i in range(receiverPos.shape[0]):
        for j in range(emitterPos.shape[0]):
            dtdx = 1 / c * (emitterPos[j][0] - receiverPos[i][0]) / np.linalg.norm(emitterPos[j] - receiverPos[i])
            dtdy = 1 / c * (emitterPos[j][1] - receiverPos[i][1]) / np.linalg.norm(emitterPos[j] - receiverPos[i])

            dfdx = fc / c * (receiverVel[i][0] - emitterVel[j][0]) / np.linalg.norm(emitterPos[j] - receiverPos[i]) - \
                   fc / c * (emitterPos[j][0] - receiverPos[i][0]) * \
                   np.sum((receiverVel[i] - emitterVel[j]) * (emitterPos[j] - receiverPos[i])) / \
                   np.linalg.norm(emitterPos[j] - receiverPos[i]) ** 3
            dfdy = fc / c * (receiverVel[i][1] - emitterVel[j][1]) / np.linalg.norm(emitterPos[j] - receiverPos[i]) - \
                   fc / c * (emitterPos[j][1] - receiverPos[i][1]) * \
                   np.sum((receiverVel[i] - emitterVel[j]) * (emitterPos[j] - receiverPos[i])) / \
                   np.linalg.norm(emitterPos[j] - receiverPos[i]) ** 3

            t = np.linalg.norm(emitterPos[j] - receiverPos[i]) / c
            f = fc / c * np.sum((receiverVel[i] - emitterVel[j]) * (emitterPos[j] - receiverPos[i])) / np.linalg.norm(
                emitterPos[j] - receiverPos[i])

            ts = np.arange(emitterSignal.shape[1]) / samplingRate
            F = np.diag(np.exp(1j * 2 * np.pi * f * ts))
            dFdf = 1j * 2 * np.pi * np.diag(ts * np.exp(1j * 2 * np.pi * f * ts))

            emitter_signal_with_time_delay = emitterSignal.copy()
            emitter_signal_with_time_delay[0, round(t * samplingRate):] = emitter_signal_with_time_delay[0,
                                                                          :-round(t * samplingRate)]
            emitter_signal_with_time_delay[0, :round(t * samplingRate)] = 0
            dsdt = np.zeros(emitter_signal_with_time_delay.shape, dtype=complex)
            dsdt[0, :-1] = emitter_signal_with_time_delay[0, 1:] - emitter_signal_with_time_delay[0, :-1]
            dsdt[0, -1] = emitter_signal_with_time_delay[0, -1]
            dsdt = -dsdt / (1 / samplingRate)

            # 衰减系数
            attenRange = [0.5, 1]
            real = np.random.random() * (attenRange[1] - attenRange[0]) + attenRange[0]
            img = 1j * (np.random.random() * (attenRange[1] - attenRange[0]) + attenRange[0])
            a[i] = real + img

            dmdx[j][i] = a[i] * dfdx * np.dot(dFdf, emitter_signal_with_time_delay.T) + \
                      a[i] * dtdx * np.dot(F, dsdt.T)

            dmdy[j][i] = a[i] * dfdy * np.dot(dFdf, emitter_signal_with_time_delay.T) + \
                      a[i] * dtdy * np.dot(F, dsdt.T)

    temp = list(zip(dmdx, dmdy))
    dmdxy = []
    for item in temp:
        dmdxy.extend(list(item))
    J = np.zeros([len(dmdxy), len(dmdxy)])
    for i in range(len(dmdxy)):
        for j in range(len(dmdxy)):
            dmdxy_flatten1 = np.expand_dims(np.array(dmdxy[i]).flatten(), 1)
            dmdxy_flatten2 = np.expand_dims(np.array(dmdxy[j]).flatten(), 1)
            J[i][j] = 2 / sigma2 * np.real(np.dot(dmdxy_flatten1.T.conjugate(), dmdxy_flatten2))

    # rank = np.linalg.matrix_rank(J)
    # det = np.linalg.det(J)
    # print("Rank:", rank, " Determinant:", det)

    Jinv = np.linalg.inv(J)

    crlb = []
    arguments_num = 2 # x,y两个维度
    for i in range(emitterPos.shape[0]):
        crlb.append(np.sqrt(  # Jinv[arguments_num * i, arguments_num * i]为参数x的方差估计，
            # Jinv[arguments_num * i + 1, arguments_num * i + 1]为参数y的方差估计，求和开根号则为发射机位置估计的方差
            Jinv[arguments_num * i, arguments_num * i] + Jinv[arguments_num * i + 1, arguments_num * i + 1])
        )

    crlb = np.mean(crlb)

    return crlb


