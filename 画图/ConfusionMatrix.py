# -*- coding: utf-8 -*-
import itertools
from matplotlib import pyplot as plt
import numpy as np 
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Greys, position=111):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    cm = np.array(cm)
    #f = plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm[::-1]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    xtick_marks = np.arange(len(cm[1]))
    print(classes, len(cm[1]))
    plt.xticks(xtick_marks, classes[:len(cm[1])] )
    plt.yticks(tick_marks, classes[:len(cm)][::-1])
    plt.xticks(rotation=70)
    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=8,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
if __name__ == "__main__":
    cnf_matrix = np.loadtxt('10error_matrix.csv', delimiter=',')
    cifar10_types = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(cnf_matrix, normalize=False, classes=cifar10_types, title='Normalized confusion matrix')
    #plt.show()

