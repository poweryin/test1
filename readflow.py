import os
color_path ='/home/z840/poweryin/flow-code/color_flow'
# path = '/home/z840/poweryin/dataset/flow/'
path = '/home/z840/poweryin/dataset/FlyingChairs_release/data/'
png_path = '/home/z840/poweryin/dataset/picture/'
length = len(os.listdir(path))
l = length +1
num = 1
while(num<l):
    ml = color_path+'\t' + path + str(num) +'_flow' + '.flo\t' + png_path + str(num)  + '.png'
    os.system(ml)
    num += 1
