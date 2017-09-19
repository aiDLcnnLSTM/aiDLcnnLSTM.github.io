#Import helper functions
from DataRow import ErrorAcum

import os,sys
import numpy as np




def test_error(txt_path):
    testError = ErrorAcum()
    fname_lst = open('name_lst.txt', 'w')

    cnt = 0
    for e in open(txt_path):
        e=e.strip()
        if len(e) < 5:
            continue
        elems = e.split('\t')
        #print(e)
        assert len(elems) == 68 * 2+4+1
        name = elems[0]
        bbox = elems[1:5]
        pts = elems[5:]
        #print(len(pts))
        #print(e)
        assert len(pts) == 68*2
        f_pts = []
        for e in pts:
            #print(e)
            f_e = float(e)
            f_pts.append(f_e)

        pred_txt = 'D:/FaceLandmark/dataset/Data/20170904/' + name[0:-4]+'_res.txt'
        pred_txt = pred_txt.replace('\\', '/')
        pred_txt = pred_txt.replace('\\', '/')
        #print(pred_txt)
        assert os.path.exists(pred_txt)

        f_pred = open(pred_txt, 'r')
        pt_num = f_pred.readline()
        pred_pts = []
        for i in range(int(pt_num)):
            line = f_pred.readline()
            pt = line.split(' ')
            pred_pts.append(float(pt[0]))
            pred_pts.append(float(pt[1]))

        testError.add_68(groundTruth=np.array(f_pts, np.double), pred= np.array(pred_pts, np.double))
        print("image Error: %f" % (testError.meanError_68().mean() * 100))
        cnt += 1




    print('\n\nresult\n')
    print(cnt)

    print("failed: %d" % (testError.failureCounter))
    print("image Error: %f" % (testError.meanError_68().mean() * 100))
    f = open('errors_68.txt', 'w')
    res = testError.meanError_68().mean() * 100
    res = res.astype(np.float32)
    f.write( str(res) )
    f.close()

    print(testError.itemsCounter)


test_error('./300W_GT_FD_testList-common.txt')