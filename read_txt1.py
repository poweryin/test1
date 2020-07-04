# -*- coding: UTF8 -*-




import cPickle as pickle
# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
f=open('/private/poweryin/MCPRL-master/data/models/ResNet50_similar.pkl','rb')
data=pickle.load(f)
print(data)

#
# with open('/private/poweryin/MCPRL-master/py-faster-rcnn/txtfile/result_8_3_3_100_all_clas_re.txt','r') as f:
#     lines = f.readlines()
# for line in lines:
#     im_name1 = line.strip('\n').split(' ')[0]
#     im_name2 = line.split('/')[0]
#     print(im_name2)

