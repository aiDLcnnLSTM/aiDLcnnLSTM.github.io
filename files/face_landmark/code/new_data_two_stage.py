import os,sys
import numpy as np
import cv2
import random

DEBUG = False


def gene_data(txt_path):

    fname_lst = open('name_lst.txt', 'w')

    for e in open(txt_path):
        print(e)
        elems = e.split('\t')
        name = elems[0]
        bbox = elems[1:5]
        pts = elems[5:]

        fname = open(name[0:-4]+'.txt', 'w')
        fname.write(name+'\n')
        fname.close()

        frect = open(name[0:-4]+'.rct', 'w')
        for i in bbox:
            frect.write(i+ ' ')
        frect.close()

        fpts = open(name[0:-4]+'.pts', 'w')
        for pt_i in range(len(pts)//2):
            pt_x = pts[2*pt_i]
            pt_y = pts[2 * pt_i+1]
            fpts.write(pt_x + ' ' +pt_y + '\n')
        fpts.close()


gene_data('300W_GT_FD_testList.txt')