import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# 相关库

def plot_matrix(y_true, y_pred, labels_name, savepath, axis_labels=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    matplotlib.use('agg')
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('summer'))
    plt.colorbar()  # 绘制图例

# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            # if int(cm[i][j] * 100 + 0.5) > 0:
            plt.text(j, i, cm[i][j],
                        ha="center", va="center",
                        color="black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
#     plt.show()
    fig.savefig(savepath)
