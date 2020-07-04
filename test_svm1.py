import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import array
import os
import re
feature_path="/home/z840/dataset/UCF_Crimes/crop_out_30/add/bg_feature/test"
All_Folder = os.listdir(feature_path)
All_Folder.sort()
for i in All_Folder:
    li = []
    filepath = os.path.join(feature_path, i)
    # read binary data
    feature_folder=os.listdir(filepath)
    lenth=len(feature_folder)
    feature_folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    for id in feature_folder:
        filepath_ = os.path.join(filepath, id)
        f = open(filepath_, "rb")
        # read all bytes into a string
        s = f.read()
        f.close()
        (n, c, l, h, w) = array.array("i", s[:20])
        feature_vec = np.array(array.array("f", s[20:]))
        li.append(feature_vec)
X_train = np.array(li)
# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
# 预测的结果为-1 abnormal 或 1 normal ,在这个群落中为1,不在为-1.
y_pred_train = clf.predict(X_train)
print(y_pred_train)
normal = X_train[y_pred_train == 1]
abnormal = X_train[y_pred_train == -1]
print(normal)
print(abnormal)
print(normal.shape)
print(abnormal.shape)


print("labels_true")
print(labels_true)

plt.plot(normal[:, 0], normal[:, 1], 'bx')
plt.plot(abnormal[:, 0], abnormal[:, 1], 'ro')
plt.show()





plt.title("Novelty Detection")

s = 20
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='black', s=s)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.show()