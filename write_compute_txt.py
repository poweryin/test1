
import re
def ratio(string):
    video =string.split(".",1)[0]
    # a,b=string.split(":", maxsplit=1)
    _,ind, gt = string.split(":", maxsplit=2)
    ind=ind[:-3]
    gt=gt[:-1]

    ind = list(map(int, ind.split()))

    gt = list(map(int, gt.split()))
    tmp, tmpGt, tmpLen, gtLen = [], [], 0, 0
    gtLen += gt[1]-gt[0]

    try:
        gtLen += gt[3]-gt[2]
    except:
        pass
    if len(ind)==1:
        tmpLen=gtLen
        return "{}:{}\n".format(video, str(tmpLen / gtLen))
    if len(ind)==0:
        tmpLen=0
        return "{}:{}\n".format(video, str(tmpLen / gtLen))
    if len(ind)>1 and len(ind)%2!=0:
        ind.pop()

    for i in range(0, len(ind), 2):
        tmp.append((ind[i], ind[i+1]))
    tmpGt.append((gt[0], gt[1]))

    if len(gt)!=2:
        tmpGt.append((gt[2], gt[3]))

    for a in tmp:
        if a[0]>=tmpGt[0][0]:
            if a[1] <= tmpGt[0][1]:
                tmpLen += a[1]-a[0]
            else:
                # tmpLen += tmpGt[0][1]
                tmpLen += 0
        else:
            try:
                if a[0]>=tmpGt[1][0]:
                    if a[1] <= tmpGt[0][1]:
                        tmpLen += a[1]-a[0]
                    else:
                        tmpLen += 0
            except:
                pass
    return "{}:{}\n".format(video, str(tmpLen/gtLen))

if __name__ == '__main__':
    res = []
    n=0
    m=0
    with open("/data/UCF_Crimes/c3d_features/curve_tmp/ind.txt", 'r') as f:
        for line in f.readlines():
            res.append(ratio(line))
            _,rate=ratio(line).split(":", maxsplit=2)
            rate=rate[:-1]
            if rate=='0.0':
                n=n+1
            if rate=='1.0':
                m=m+1
        print(res)
        print(n)
        print(m)

    with open("/data/UCF_Crimes/c3d_features/cu./train.txtrve_tmp/ratio.txt", 'w') as g:
        g.writelines(res)


