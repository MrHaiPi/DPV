from torch.utils.data import Dataset
from DataFunction.DataSetDis import DataSetCsv1


# 加载仿真数据
class MyDataSetCsv1(Dataset):
    """自定义数据集"""

    def __init__(self, root_path: str, receiverNum=3, emitterNum=1, time_scale=None, transform=None, time_fre_trans=None, fre_scale=None, isNormal=False, coordDim=3, dataType='IQ'):

        self.dataset = DataSetCsv1(root_path, receiverNum, emitterNum, time_scale, transform, time_fre_trans, fre_scale, isNormal, coordDim, dataType)

    def __len__(self):
        return self.dataset.getDataNum()

    def __getitem__(self, item):
        return self.dataset.getData(item, False)
