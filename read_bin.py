# coding:utf-8
import struct
import os
if __name__ == '__main__':
    filepath='/home/z840/Downloads/C3D/C3D-v1.0/examples/c3d_feature_extraction/output/c3d/v_ApplyEyeMakeup_g01_c01/000000.fc6-1'
    binfile = open(filepath, 'rb') #打开二进制文件
    size = os.path.getsize(filepath) #获得文件大小
    for i in range(size):
        data = binfile.read(1)
        #每次输出一个字节
        print(data)
    binfile.close()
