import os,sys
import scipy.io as sio
import matplotlib.pyplot as plt


def mat2txt(mat_path):
    # matlab文件名
    matfn = mat_path
    data = sio.loadmat(matfn)
    boundbox = data['bounding_boxes']

    H,W = boundbox.shape


    f = open(matfn+'.txt', 'w')

    for i in range(W):
        elem = boundbox[0][i]

        (name, bb_det, bb_GT) = elem[0][0]

        str_name = ''
        for e in name[0]:
            str_name += e
        f.write(str_name + ' ')

        str_bb_det = ''
        for e in bb_det[0]:
            str_bb_det += str(e) + ' '
        f.write(str_bb_det + ' ')

        str_bb_GT = ''
        for e in bb_GT[0]:
            str_bb_GT += str(e) + ' '
        f.write(str_bb_GT + '\n')



    f.close()


#matfn = 'D:/FaceLandmark/300W_Dataset/Bounding Boxes/bounding_boxes_afw'
#matfn = 'D:/FaceLandmark/300W_Dataset/Bounding Boxes/bounding_boxes_afw'
#mat2txt(matfn)

mat_lst = os.listdir('D:/FaceLandmark/300W_Dataset/Bounding Boxes/')
for e in mat_lst:
    prex = 'D:/FaceLandmark/300W_Dataset/Bounding Boxes/'
    mat2txt(prex+e)