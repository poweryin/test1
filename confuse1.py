import numpy as np
# MRes35-3D:0.52
# M=[[ 90 ,19,  36,  2],
#  [ 40, 388, 64, 139],
#  [  8,   3, 375,  18],
#  [80, 205, 760, 1220]]

# MSC8-3D:0.49
# M=[[ 80 ,33,  36,  2],
#  [ 50, 380, 66, 139],
#  [  28, 3, 365,  28],
#  [80, 205, 799, 1181]]

# M=[[ 0 ,52,  0,  95],
#  [ 0, 19, 0, 542],
#  [  0,   17, 0,  387],
#  [0, 387, 0, 1278]]
# Res34
# M=[[ 40 ,34,  29,  44],
#  [ 114, 245, 168, 84],
#  [  29,   0, 335,  40],
#  [244, 103, 832, 486]]
# res34-p
# M=[[ 86 ,15,  38,  8],
#  [ 80, 388, 93, 50],
#  [  0,   2, 372,  30],
#  [81, 164, 620, 800]]
# C3D
# M=[[ 55 ,59,  26,  7],
#  [ 25, 280, 60,207],
#  [  42,   50, 149,  163],
#  [0, 637, 0, 1128]]

# M=[[ 65 ,24,  56,  2],
#  [ 90, 368, 114, 59],
#  [  8,   3, 329,  64],
#  [80, 305, 799, 1081]]

M=[[ 77 ,40,  36,  4],
 [ 60, 367, 112, 166],
 [  28, 3, 362,  31],
 [100, 215, 799, 1151]]

# M=[[ 80 ,33,  36,  2],
#  [ 50, 380, 66, 139],
#  [  28, 3, 365,  28],
#  [80, 205, 799, 1181]]


# M = np.zeros((4, 4), dtype=np.int64)
precision=[]
recall=[]
F1=[]
n = len(M)
for i in range(len(M[0])):
    rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
    try:
        P=M[i][i]/float(colsum)
        R=M[i][i] / float(rowsum)
        F1_score=2*P*R/(P+R)
        print ('precision: %s' % (M[i][i]/float(colsum)), 'recall: %s' % (M[i][i]/float(rowsum)),'F1:%s'%(F1_score))
        precision.append(P)
        recall.append(R)
        F1.append(F1_score)
    except ZeroDivisionError:
        print ('precision: %s' % 0, 'recall: %s' %0)
M = np.array(M)
total = np.sum(M)
TP = 0
for i in range(n):
    TP += M[i,i]
final_acc = TP/total

print('pre_av:')
print(precision)
print(sum(precision)/4)
print('recall_av:')
print(recall)
print(recall[:3])
print(sum(recall[:3])/3)
print('F1_av:')
print(F1)
print(sum(F1)/4)
print('acc')
print(final_acc)