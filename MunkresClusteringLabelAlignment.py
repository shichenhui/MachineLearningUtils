from munkres import Munkres


def best_map(L1,L2):
  Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
	nClass1 = len(Label1)        # 标签的大小
	Label2 = np.unique(L2)       
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2


# 下面这个函数也可以
import numpy as np
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
        new_y_pred: new predicted labels after reassignment
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    accuracy = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    # 创建一个字典，用于将旧标签映射到新标签
    mapping = {old_label: new_label for old_label, new_label in ind}
    
    # 重新分配y_pred标签
    new_y_pred = np.array([mapping[label] for label in y_pred])
    
    return accuracy, new_y_pred

# 示例代码
y_true = np.array([0, 1, 2, 1, 0])
y_pred = np.array([2, 0, 1, 0, 2])

accuracy, new_y_pred = clustering_accuracy(y_true, y_pred)
print("Accuracy:", accuracy)
print("New y_pred:", new_y_pred)
