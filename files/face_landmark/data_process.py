import os,sys
import numpy as np
import cv2


def add_prex(txt_path, prex):
    assert os.path.exists(txt_path)
    #assert os.path.exists(prex)
    lst = []
    for line in open(txt_path):
        line1 = prex + line
        lst.append(line1)

    return lst

def merge_txt_train():
    lst1 = add_prex('./train/bounding_boxes_afw.mat.txt', 'afw/')
    lst2 = add_prex('./train/bounding_boxes_helen_trainset.mat.txt', 'helen/trainset/')
    lst3 = add_prex('./train/bounding_boxes_lfpw_trainset.mat.txt', 'lfpw/trainset/')
    lst4 = add_prex('./train/bounding_boxes_xm2vts.mat.txt', 'xm2vts/')

    lst = lst1+lst2+lst3+lst4

    f = open('300W_trainset.txt', 'w')
    for e in lst:
        f.write(e)
    f.close()


def merge_txt_test():
    lst1 = add_prex('./test/bounding_boxes_ibug.mat.txt', 'ibug/')

    lst2 = add_prex('./test/bounding_boxes_helen_testset.mat.txt', 'helen/testset/')
    lst3 = add_prex('./test/bounding_boxes_lfpw_testset.mat.txt', 'lfpw/testset/')

    lst = lst1#+lst2+lst3

    f = open('300W_testset_challenge.txt', 'w')
    for e in lst:
        f.write(e)
    f.close()



def generate_list(list_path):
    assert os.path.exists(list_path)

    f_OD = open(list_path[0:-4] + '_OD_list' + list_path[-4:], 'w')
    f_GT = open(list_path[0:-4]+ '_GT_list' +list_path[-4:], 'w')
    for line in open(list_path):
        line = line.strip()
        if len(line) <= 0:
            break
        elem = line.split(' ')
        name = elem[0]
        OD = elem[1:5]
        GT = elem[5:10][1:5]

        img_prex = 'D:/FaceLandmark/300W_Dataset/orgin_dataset/'
        pts_path = img_prex+name[0:-4]+'.pts'
        print(pts_path)
        print(elem)
        assert os.path.exists(pts_path)
        pts_lst = []
        f_pts = open(pts_path)
        line_tmp = f_pts.readline()
        line_tmp = f_pts.readline()
        line_tmp = f_pts.readline()
        for i in range(68):
            ptxy = f_pts.readline()
            xy = ptxy.strip().split(' ')
            print(ptxy)
            print(line)
            print(i)
            print('***************')
            assert len(xy)==2
            pts_lst.append(xy[0])
            pts_lst.append(xy[1])

        OD_list = [name] + OD + pts_lst
        for e in OD_list:
            f_OD.write(e+' ')
        f_OD.write('\n')

        GT_list = [name] + GT + pts_lst

        debug_str = ''
        print(GT)
        for e in GT_list:
            debug_str+= e + ' '
        #print(debug_str)
        #input('pause')

        for e in GT_list:
            f_GT.write(e + ' ')
        f_GT.write('\n')

    f_OD.close()
    f_GT.close()


if __name__ == "__main__":
    print('main')
    #merge_txt_test()
    generate_list('./300W_testset_challenge.txt')
    generate_list('./300W_testset_common.txt')
    generate_list('./300W_testset_full.txt')
    generate_list('./300W_trainset.txt')