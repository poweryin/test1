import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def accuracy(frameNum=0, start=0, end=-1, GT=None, acc=1, cls=0):
    GT = [0] * frameNum
    GT[GT[0]:GT[1]] = [cls]*(GT[1]-GT[0])
    GT = np.asarray(GT)
    n = frameNum-frameNum*acc
    label = GT[:]
    if start-n:
        label[start-n:start] = [cls] * n
    else:
        label[:start] = [cls] * n
        label[end:end+n-start] =  [cls] * n
    acc = accuracy_score(GT, label)
    pre = precision_score(GT, label, average='macro')
    recall = recall_score(GT, label, average='macro')
    f1 = f1_score(GT, label, average='macro')
    print("acc:{}   pre:{}, recall:{}, f1:{}".format(acc, pre, recall, f1))
    pass

if __name__ == '__main__':
    accuracy()
