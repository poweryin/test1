# coding=UTF-8
from sklearn.utils.multiclass import type_of_target


from sklearn import metrics
import matplotlib.pylab as plt
import scipy.io as scio
dataFile = '/home/z840/Downloads/mat/label_true.mat'
data = scio.loadmat(dataFile)
idx=data['id']
dataFile1 = '/home/z840/Downloads/mat/scores.mat'
data1 = scio.loadmat(dataFile1)
score=data1['id']
print(type_of_target(idx))


# 真实值
# idx= [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
#           0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# # 模型预测值
# score = [0.99, 0.98, 0.11, 0.93, 0.30, 0.80, 0.20, 0.75, 0.23, 0.95,
#             0.12, 0.87, 0.33, 0.14, 0.95, 0.29, 0.18, 0.2, 0.9, 0.09]

idx=idx.tolist()
score=score.tolist()
print(type_of_target(idx))
print(idx[0])
print(score[0])
print(len(idx[0]))
print(len(score[0][:8841]))


fpr, tpr, thresholds = metrics.roc_curve(idx[0], score[0][:8841], pos_label=0)
roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
print(roc_auc)

plt.plot(fpr, tpr, 'gold',linestyle='--',linewidth=3.5)
plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')  # 横坐标是fpr
plt.ylabel('True Positive Rate')  # 纵坐标是tpr
plt.title('Receiver operating characteristic example')
plt.show()

