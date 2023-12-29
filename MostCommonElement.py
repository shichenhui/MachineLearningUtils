# 统计一个数组中出现最多的元素和出现的次数
def most_common_label(labels):
    labels = np.array(labels)
    m = np.bincount(labels).argmax()
    # 计算m出现的次数
    return m, Counter(labels)[m]
