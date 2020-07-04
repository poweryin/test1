# def confusion_metrics(tp, fp, fn, tn, classes):
#     print("{:15s} {:15s} {:15s} {:15s}".format("abnormalEvent", "TPR", "FPR", "FNR"))
#     tpr = float(tp) / (float(tp) + float(fn))
#     fpr = float(fp) / (float(fp) + float(tn))
#     fnr = float(fn) / (float(tp) + float(fn))
#     accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
#     recall = tpr
#     precision = float(tp) / (float(tp) + float(fp))
#     f1_score = (2 * (precision * recall)) / (precision + recall)
#
#     print("{}   {}   {}  {}".format(classes, tpr, fpr, fnr))

def confusion_metrics(tp, fp, fn, tn, classes):
    acc=[]

    tpr = float(tp) / (float(tp) + float(fn))
    fpr = float(fp) / (float(fp) + float(tn))
    fnr = float(fn) / (float(tp) + float(fn))
    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
    recall = tpr
    precision = float(tp) / (float(tp) + float(fp))
    f1_score = (2 * (precision * recall)) / (precision + recall)
    acc.append(accuracy)
    print("{}   {}   {}  {}  {}".format(classes, precision, tpr, f1_score,accuracy))
    return acc
if __name__ == '__main__':
    # confusion_metrics(2273,159,210,5279,classes="Arson")
    # confusion_metrics(11350, 1071, 2097, 11968, classes="Explosion")
    # confusion_metrics(4831, 0, 294, 3274, classes="RoadAccident")
    # confusion_metrics(4831, 0, 294, 3274, classes="Normal")
    print("{:15s} {:15s} {:15s} {:15s} {:15s} ".format("abnormalEvent", "precision", "tpr", "f1_score", "accuracy"))
    acc1=confusion_metrics(6172,100,2886,4258,classes="Arson")
    acc2=confusion_metrics(5743,32857,12809,12811, classes="Explosion")
    acc3=confusion_metrics(294,24,38093,534, classes="RoadAccident")
    acc4=confusion_metrics(40179,22011,46604,64926, classes="Normal")
    print(acc1)
    print(acc2)
    print(acc3)
    print(acc4)
    avg_acc=sum(acc1+acc2+acc3+acc4)/4
    print(avg_acc)





