import numpy as np

# M=[[6172,100,2886,4258],
#  [5743,32857,12809,12811],
#  [294,24,38093,1534],
#  [40179,22011,46604,64926]]

M=[[ 6172,100,886,258],
 [ 5743,32857,12809,12811],
 [  294,24, 38093, 1534],
 [40179,22011,46604,64926]]


# M = np.zeros((4, 4), dtype=np.int64)

n = len(M)
for i in range(len(M[0])):
    rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
    try:
        P=M[i][i]/float(colsum)
        R=M[i][i] / float(rowsum)
        F1=2*P*R/(P+R)
        print ('precision: %s' % (M[i][i]/float(colsum)), 'recall: %s' % (M[i][i]/float(rowsum)),'F1:%s'%(F1))
    except ZeroDivisionError:
        print ('precision: %s' % 0, 'recall: %s' %0)
M = np.array(M)
total = np.sum(M)
TP = 0
for i in range(n):
    TP += M[i,i]
final_acc = TP/total
print(final_acc)
