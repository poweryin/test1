
# coding=utf-8
def Trans(dic):
    rt=[]
    try:
        fp = open("dict.txt",'w')
        for (k,v) in dic.items():
            for (ik,iv) in v.items():
                fp.write('%-10s %-10s %-10s %-10s\n' %(k,ik,iv[0],iv[1]))
        return rt
    finally:
        fp.close()

if __name__=="__main__":
    dic={'11542': {'68784': [5.0,4.0], '43485': [5.0,3.0], '83646': [5.0,2.0], '109754': [5.0,8.0], '119735': [3.0,1.0], '42640': [2.0,5.0], '69983': [5.0,7.0], '119736': [5.0,1.0]}}
    rt=Trans(dic)