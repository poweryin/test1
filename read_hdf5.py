# coding:utf-8
import h5py
# f = h5py.File('/data/UCF_Crimes/Videos/output_test_frm/Abuse028_x264','r')
# f.keys()
# c3d_features = f["/"]["/Abuse028_x264.mp4"]['c3d_features'].value
# # c3d_features = f["/"]["/RoadAccidents002_x264.mp4"]['c3d_features'].value
# out_file = "/media/z840/2A73-C58D/yin/anomal/output/"
# for i in range(len(c3d_features)):
#     o_file = out_file+"%d.fc6-1"%(i+1)
#     with open(o_file,'w') as f:
#         f.write(" ".join([str(l) for l in list(c3d_features[i])]))
data=h5py.File('/home/z840/disktend/oldhome/poweryin/AnomalyDetection_MIL/weightsAnomalyL1L2_20000.mat','r')
print(data)