# 深度学习构建DataLoader
# 先创建自己数据的数组，然后下面的GetLoader封装，传入数据和标签矩阵，再用torch的DataLoader封装
import numpy as np
from torch.utils.data import DataLoader

transforms = None

# 创建自定义数据集
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label, transforms=None):
        self.data = data_root
        self.label = data_label
        self.transforms = transforms

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        if self.transforms:
          data = self.transform(data)
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

X_train = np.random.rand(100, 50)
y_train = np.random.choice([0, 1], size=100)

# 对于一维数据需要在中间扩充一个维度，(100, 50)->(100, 1, 50)
X_train = torch.from_numpy(X_train[:, np.newaxis, :])

train_data = DataLoader(GetLoader(X_train, y_train), batch_size=batch_size, shuffle=False)

