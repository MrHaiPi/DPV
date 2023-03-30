from itertools import permutations
import scipy as sp
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans

from DataFunction import DataIO
from DataFunction.DataSetDis import DataSetCsv1
import matplotlib.pyplot as plt
import time
import numpy as np
from DPDFunction import MLDPD, MVDRDPD

if __name__ == "__main__":
    # 用于确定数据文件位置
    SnrRange = [-5, -5]
    dataFileRoot = r"E:\资料\研究生\课题\射频定位\code\Dataset\test\SNR" + str(SnrRange)

    errorUnit = '1e6m'
    dataLength = 256

    # 接收机个数
    receiverNum = 4
    # 发射机个数
    emitterNum = 3

    dataSet = DataSetCsv1(root_path=dataFileRoot + "/valid", receiverNum=receiverNum,
                          emitterNum=emitterNum, transform=None,
                          time_fre_trans=None, fre_scale=None, time_scale=dataLength, isNormal=False)

    # 用于定位的接收机编号:0,1,2,...
    locationReceiverNum = [0, 1, 2]

    # 计算精度
    delta = 10000/2
    delta *= 2
    testErrors = []
    useTime = 0
    allUseTime = 0
    for num in range(dataSet.getDataNum()):

        data, label = dataSet.getData(num, True)
        data1, data2 = data[0], data[1]

        fc = data2[-1]
        samplingRate = data2[-2]

        emitterPos = np.zeros([emitterNum, 3])
        emitterVel = np.zeros([emitterNum, 3])
        receiverPos = np.zeros([receiverNum, 3])
        receiverVel = np.zeros([receiverNum, 3])

        label = label.reshape([emitterNum, 4])
        label = label[:, :-1] # remove confidence label
        emitterPos = label.copy()

        for i in range(receiverNum):
            receiverPos[i] = data2[i * 6:i * 6 + 3]
            receiverVel[i] = data2[(i + 1) * 6 - 3:(i + 1) * 6]

        receiveSignal = data1
        emitterSignal = None


        # 去除零值
        def clearZero(data):
            length = data.shape[1]
            sumData = np.sum(data, 1)
            index = np.where(sumData != 0)[0]
            temp = data[index]
            return temp


        # 确定真正用于定位的数据
        emitterPos = clearZero(emitterPos)
        emitterNumReal = emitterPos.shape[0]

        receiverPos = clearZero(receiverPos)
        receiverVel = clearZero(receiverVel)
        receiveSignal = clearZero(receiveSignal)
        receiverNumReal = receiverPos.shape[0]

        if emitterNumReal == 0 or receiverNumReal < 3:
            continue

        startTime = time.time()

        # 计算范围m

        # calRange = float(errorUnit[:-1])
        # loss = np.zeros([int(2 * calRange / delta + 1), int(2 * calRange / delta + 1)])
        # center = np.mean(emitterPos, 0)
        # for i in range(loss.shape[1]):
        #     for j in range(loss.shape[0]):
        #         p = np.zeros([1, 3])
        #         p[0, 0] = center[0] + i * delta - calRange
        #         p[0, 1] = center[1] + j * delta - calRange
        #         loss[j, i] = MLDPD(fc, samplingRate, p, emitterVel[0], receiverPos[locationReceiverNum],
        #                              receiverVel[locationReceiverNum], receiveSignal[locationReceiverNum])


        # 计算范围m

        calRange = float(errorUnit[:-1])
        loss = np.zeros([int(calRange / delta + 1), int(calRange / delta + 1)])
        for i in range(loss.shape[1]):
            for j in range(loss.shape[0]):
                if i == 16 and j == 28:
                    aa=0
                p = np.zeros([1, 3])
                p[0, 0] = i * delta
                p[0, 1] = j * delta
                loss[j, i] = MLDPD(fc, samplingRate, p, emitterVel[0], receiverPos[locationReceiverNum],
                                         receiverVel[locationReceiverNum], receiveSignal[locationReceiverNum], emitterSignal=None)


        # sample = np.zeros([loss.size, 3])
        # for i in range(loss.shape[0]):
        #     for j in range(loss.shape[1]):
        #         sample[i * loss.shape[1] + j, 0] = i /loss.shape[0]+ loss.min()
        #         sample[i * loss.shape[1] + j, 1] = j /loss.shape[0]+ loss.min()
        #         sample[i * loss.shape[1] + j, 2] = loss[i, j] / (loss.max() - loss.min()) + loss.min()
        #
        # model = KMeans(n_clusters=emitterNumReal + 4)
        # model.fit(sample)
        # # 为每个示例分配一个集群
        # yhat = model.predict(sample)
        # # 检索唯一群集
        # clusters = np.unique(yhat)
        # # 为每个群集的样本创建散点图
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for cluster in clusters:
        #     # 获取此群集的示例的行索引
        #     row_ix = np.where(yhat == cluster)
        #     # 创建这些样本的散布
        #     ax.scatter(sample[row_ix, 0], sample[row_ix, 1], sample[row_ix, 2])
        # # 绘制散点图
        # plt.show()

        # 找出loss中的前emitterNum个最大值对应的下标
        # lossObj = loss.copy()
        # predict = np.zeros([emitterNumReal, 3])
        # for i in range(emitterNumReal):
        #     index = np.array(np.where(lossObj == np.max(lossObj))).reshape([1, 2])
        #     predict[i, :2] = index * delta
        #     lossObj[index[0][0], index[0][1]] = -np.inf

        # 找出loss中的前emitterNum个最大值对应的下标
        lossObj = loss.copy()
        # plt.figure()
        # plt.imshow(lossObj)

        #sigma = [0.5, 0.5]
        # # 平滑损失函数
        # lossObj = sp.ndimage.filters.gaussian_filter(lossObj, sigma, mode='constant')
        # # 锐化损失函数
        # kernel = np.array([[-1, -1, -1],
        #                [-1, 10, -1],
        #                [-1, -1, -1]])
        # lossObj = cv2.filter2D(lossObj, -1, kernel)

        # plt.figure()
        # plt.imshow(lossObj)

        predict = np.zeros([emitterNumReal, 3])
        maxAllIndex = argrelextrema(lossObj, np.greater)
        maxValue = lossObj[maxAllIndex]
        maxValue = np.sort(maxValue)
        # 先确定最好的
        # 在确定其他的，排除与最好的比较近的
        ensuredCount = 0
        offset = 0
        for i in range(emitterNumReal):
            if i == 0:
                temp = np.where(lossObj == maxValue[- i - 1])
                predict[i, 0] = temp[0][0].item() * delta
                predict[i, 1] = temp[1][0].item() * delta
                ensuredCount += 1
            else:
                while True:
                    temp = np.where(lossObj == maxValue[- i - 1 - offset])
                    pre = np.zeros([1, 3])
                    pre[0, 0] = temp[0][0] * delta
                    pre[0, 1] = temp[1][0] * delta
                    canEnsure = True
                    for j in range(ensuredCount):
                        if np.sqrt(np.sum(np.square(pre - predict[j, :]))) < 10 * delta:
                            canEnsure = False
                            break
                    if canEnsure:
                        predict[i] = pre
                        ensuredCount += 1
                        break
                    else:
                        offset += 1


        # 将x, y数据互换，这样才能与发射机的坐标对齐
        temp = predict.copy()
        predict[:, 0] = temp[:, 1]
        predict[:, 1] = temp[:, 0]

        # 将估计的位置与真实位置一一对应
        combResult = list(permutations(np.arange(predict.shape[0]), predict.shape[0]))
        minDistance = np.inf
        minErroIndex = None
        for com in combResult:
            error = np.mean(np.sqrt(np.sum(np.square(predict[list(com)] - emitterPos), 1)))
            if minDistance > error:
                minDistance = error
                minErroIndex = list(com)
        predict = predict[minErroIndex]

        error = np.sqrt(np.sum(np.square(predict - emitterPos), 1)) / float(errorUnit[:-1])
        useTime = time.time() - startTime
        print('num:{}/{},time:{},receiverNum:{},error({}):{}'.format(num + 1, dataSet.getDataNum(), useTime, receiverNumReal, errorUnit, error))
        # for e in error:
        #     testErrors.append(e)
        testErrors.append(error.mean())
        allUseTime += useTime

        # loss绘图
        isPlot = False
        plotType = 0
        if isPlot:
            predict /= float(errorUnit[:-1])
            emitterPos /= float(errorUnit[:-1])
            receiverPos /= float(errorUnit[:-1])
            # 绘图
            plt.figure(figsize=(4/1.25, 3/1.25))
            for i in range(receiverNumReal):
                plt.plot(abs(receiveSignal[i]), label='receiver{}'.format(i + 1))
            plt.legend()

            plt.figure(figsize=(4/1.25, 3/1.25))
            x = np.arange(loss.shape[0]) * delta / float(errorUnit[:-1])
            y = np.arange(loss.shape[1]) * delta / float(errorUnit[:-1])
            X, Y = np.meshgrid(x, y)

            if plotType == 0:
                ## 2D绘图
                plt.contourf(X, Y, loss, cmap='rainbow')
                #plt.colorbar()
                for i in range(emitterNumReal):
                    plt.scatter(predict[i, 0], predict[i, 1], s=100, marker='8',  c='fuchsia', alpha=1,
                               label='predict' if i == 0 else None)
                    plt.scatter(emitterPos[i, 0], emitterPos[i, 1], s=100, marker='*', c='black', alpha=1,
                               label='real' if i == 0 else None)
                for i in range(receiverNumReal):
                    plt.scatter(receiverPos[i, 0], receiverPos[i, 1], s=100, marker='>', c='k', alpha=1,
                                label='receiver' if i == 0 else None)
            else:
                ## 3D绘图
                ax = plt.figure().gca(projection="3d")

                # 幅度
                #ax.plot_surface(X, Y, loss, rstride=1, cstride=1, cmap='rainbow', linewidth=0.9, antialiased=True)

                # 柱状图
                # xpos = X.flatten('F')
                # ypos = Y.flatten('F')
                # zpos = np.zeros_like(xpos)
                ##设置柱形图大小
                # dx = delta * np.ones_like(zpos)
                # dy = dx.copy()
                # dz = loss.flatten()
                # colors = plt.cm.jet(loss.flatten()/float(loss.max()))
                # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average', shade=True)

                # 等高线
                # 进行颜色填充
                ax.contourf(X, Y, loss, 10, cmap=plt.cm.rainbow)
                # 进行等高线绘制
                c = ax.contour(X, Y, loss, 10)
                # 线条标注的绘制
                plt.clabel(c, inline=True, fontsize=10)
                plt.xticks(())
                plt.yticks(())

                for i in range(emitterNumReal):
                    ax.scatter(predict[i, 0], predict[i, 1], np.max(loss), s=100, marker='8',  c='fuchsia', alpha=1,
                               label='predict' if i == 0 else None)
                    ax.scatter(emitterPos[i, 0], emitterPos[i, 1], np.max(loss), s=100, marker='*', c='black', alpha=1,
                               label='real' if i == 0 else None)

            plt.xlabel('x/1e6m')
            plt.ylabel('y/1e6m')
            #plt.title('emitterPos:{}\npredict:{}\nerror:{}'.format(emitterPos, predict, error))

            # 卫星位置绘制
            # x = receiverPos[locationReceiverNum, 0]
            # y = receiverPos[locationReceiverNum, 1]
            # z = np.ones([locationReceiverNum.__len__(), 1]) * np.max(loss)
            # ax.scatter(x, y, z, marker='1', s=300, c='black')

            plt.legend(loc='lower left')
            plt.show()

    print('SNR:', SnrRange)
    print('average time:', allUseTime / dataSet.getDataNum())
    testErrors = np.array(testErrors)
    colors1 = '#00CED1'  # 点的颜色
    area = np.pi * 4 ** 2  # 点面积
    plt.figure(figsize=(4/1.25, 3/1.25))
    plt.scatter(range(testErrors.shape[0]), testErrors, s=area, c=colors1, alpha=0.4)
    print('Average RMSE({}):{}'.format(errorUnit, testErrors.mean()))
    #plt.title('Average RMSE({}):{}'.format(errorUnit, testErrors.mean()))
    plt.ylim([1e-4, 1.5])
    plt.xlabel('sample number')
    plt.ylabel('RMSE/{}'.format(errorUnit))
    plt.yscale('log')
    plt.grid(axis='y', ls='--')
    plt.gca().yaxis.grid(True, which='minor', ls='--')  # minor grid on too
    plt.show()

    # path = r"E:\资料\研究生\课题\射频定位\code\DeepPL\CompareExperRes" + "/" + errorUnit + '/SNR' + str(SnrRange)
    # DataIO.MakeDir(path)
    # path = path + '/' + '(time{})ESDPD average error({})'.format(round(allUseTime / dataSet.getDataNum(), 3), dataLength)
    # plt.savefig(path + '.png')
    # np.save(path + '.npy', testErrors)
