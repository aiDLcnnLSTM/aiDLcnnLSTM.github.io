
import numpy as np
#from mercurial.templater import if_
_A = np.array  # A shortcut to creating arrays in command line 
import os
import cv2
import sys
import struct
import math
from pickle import load, dump
from zipfile import ZipFile
from urllib import urlretrieve
import time

sys.path.append("/usr/lib/python2.7")
import struct

#Import helper functions
from DataRow import DataRow, ErrorAcum, Predictor, getGitRepFolder, createDataRowsFromCSV, getValidWithBBox, getValidWithBBox_68, writeHD5, GLOBAL_IMAGE_SIZE, mergeResult, BBox

###########################    PATHS TO SET   ####################
# Either define CAFFE_ROOT in your enviroment variables or set it here
CAFFE_ROOT = os.environ.get('CAFFE_ROOT','~/caffe/caffe_root')
print "CAFFE_ROOT: %s\n" % CAFFE_ROOT
tmpPath = CAFFE_ROOT+'/python/caffe';
sys.path.append("~/caffe/caffe_root")
import caffe

# Make sure dlib python path exists on PYTHONPATH else "pip install dlib" if needed.
# sys.path.append(CAFFE_ROOT+'/models/LD_USC/VanillaCNN-master/python/src/dlib-18.18/python_examples/')
# import dlib
# detector=dlib.get_frontal_face_detector() # Load dlib's face detector

postfix = '.feat'
featPath = '../caffeData/feat/';
if not os.path.isdir(featPath):
    os.makedirs(featPath)

ROOT = getGitRepFolder()  # ROOT is the git root folder .
print "ROOT = %s" % ROOT
sys.path.append(os.path.join(ROOT, 'python'))  # Assume git root directory
# DATA_PATH = os.path.join(ROOT, 'data')
# CSV_TEST  = os.path.join(ROOT, 'data', 'testImageList_debug.txt')

DATA_PATH = os.path.join('/data/b0216.yu/300W_Test')
CSV_TEST  = os.path.join('/data/b0216.yu/300W_Test/', 'list_common_caffe_68.txt')

#DATA_PATH = os.path.join('/data/b0216.yu/30000/30000_dataset_rotate/')
#CSV_TEST  = os.path.join('/data/b0216.yu/30000/', 'listCaffe_68.txt')

# PATH_TO_WEIGHTS  = os.path.join(ROOT, 'ZOO', 'Original', 'vanillaCNN.caffemodel')
# PATH_TO_DEPLOY_TXT = os.path.join(ROOT, 'ZOO', 'Original', 'vanilla_deploy.prototxt')

MEAN_TRAIN_SET = os.path.join('/data/b0216.yu/LD_Train/68Data/NewData/', 'MeanFace.png')
STD_TRAIN_SET  = os.path.join('/data/b0216.yu/LD_Train/68Data/NewData/', 'StdVar.png')

MEAN_TRAIN_LE = os.path.join('/data/b0216.yu/LD_Train/Parts/', 'MeanFace.png')
STD_TRAIN_LE  = os.path.join('/data/b0216.yu/LD_Train/Parts/', 'StdVar.png')

MEAN_TRAIN_RE = os.path.join('/data/b0216.yu/LD_Train/Parts/EyeRight/', 'MeanFace.png')
STD_TRAIN_RE  = os.path.join('/data/b0216.yu/LD_Train/Parts/EyeRight/', 'StdVar.png')

MEAN_TRAIN_Mo = os.path.join('/data/b0216.yu/LD_Train/Parts/Mouth/', 'MeanFace.png')
STD_TRAIN_Mo  = os.path.join('/data/b0216.yu/LD_Train/Parts/Mouth/', 'StdVar.png')

MEAN_TRAIN_No = os.path.join('/data/b0216.yu/LD_Train/Parts/Nose/', 'MeanFace.png')
STD_TRAIN_No  = os.path.join('/data/b0216.yu/LD_Train/Parts/Nose/', 'StdVar.png')

MEAN_TRAIN_Co = os.path.join('/data/b0216.yu/LD_Train/Parts/Contour/', 'MeanFace.png')
STD_TRAIN_Co  = os.path.join('/data/b0216.yu/LD_Train/Parts/Contour/', 'StdVar.png')

MEAN_TRAIN_Ch = os.path.join('/data/b0216.yu/LD_Train/Parts/Chin/', 'MeanFace.png')
STD_TRAIN_Ch  = os.path.join('/data/b0216.yu/LD_Train/Parts/Chin/', 'StdVar.png')

MEAN_TRAIN_LC = os.path.join('/data/b0216.yu/LD_Train/Parts/LeftContour/', 'MeanFace.png')
STD_TRAIN_LC  = os.path.join('/data/b0216.yu/LD_Train/Parts/LeftContour/', 'StdVar.png')

MEAN_TRAIN_RC = os.path.join('/data/b0216.yu/LD_Train/Parts/RightContour/', 'MeanFace.png')
STD_TRAIN_RC  = os.path.join('/data/b0216.yu/LD_Train/Parts/RightContour/', 'StdVar.png')

MEAN_TRAIN_UM = os.path.join('/data/b0216.yu/LD_Train/Parts/UpMouth/', 'MeanFace.png')
STD_TRAIN_UM  = os.path.join('/data/b0216.yu/LD_Train/Parts/UpMouth/', 'StdVar.png')

MEAN_TRAIN_DM = os.path.join('/data/b0216.yu/LD_Train/Parts/DownMouth/', 'MeanFace.png')
STD_TRAIN_DM  = os.path.join('/data/b0216.yu/LD_Train/Parts/DownMouth/', 'StdVar.png')


# detector=dlib.get_frontal_face_detector()

###########################    STEPS TO RUN       ####################
STEPS = ['testErro']

##########################################    SCRIPT STEPS       ##################################################

if 'testErro' in STEPS:
    print "Loading image set....."
    dataRowsTest_CSV  = createDataRowsFromCSV(CSV_TEST , DataRow.DataRowFromNameBoxInterlaved_68, DATA_PATH)
    print "Finished reading %d rows from image" % len(dataRowsTest_CSV)
    dataRowsTestValid,R = getValidWithBBox_68(dataRowsTest_CSV)
    print "Original image:",len(dataRowsTest_CSV), "Valid Rows:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch

    # Best models - yubing 20160718
    ROOT_DEPLOY_TXT = '/data/b0216.yu/LD_Train/68Data/Models60/vanilla_deploy_60x60.prototxt'
    ROOT_MODEL = '/data/b0216.yu/LD_Train/68Data/Models60/LD_usc_iter_90000.caffemodel'
    root_predictor = Predictor(protoTXTPath=ROOT_DEPLOY_TXT, weightsPath=ROOT_MODEL, meanPath=MEAN_TRAIN_SET, stdPath=STD_TRAIN_SET) 
    
    MEAN_TRAIN_LE_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/LeftEye/', 'MeanFace.png')
    STD_TRAIN_LE_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/LeftEye/', 'StdVar.png')

    MEAN_TRAIN_RE_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/RightEye/', 'MeanFace.png')
    STD_TRAIN_RE_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/RightEye/', 'StdVar.png')

    MEAN_TRAIN_Mo_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Mouth/', 'MeanFace.png')
    STD_TRAIN_Mo_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Mouth/', 'StdVar.png')

    MEAN_TRAIN_No_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Nose/', 'MeanFace.png')
    STD_TRAIN_No_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Nose/', 'StdVar.png') 
    
    MEAN_TRAIN_Co_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Contour/', 'MeanFace.png')
    STD_TRAIN_Co_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Contour/', 'StdVar.png')
    
    MEAN_TRAIN_Ch_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Chin/', 'MeanFace.png')
    STD_TRAIN_Ch_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Chin/', 'StdVar.png')
    
    MEAN_TRAIN_LC_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/LeftContour/', 'MeanFace.png')
    STD_TRAIN_LC_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/LeftContour/', 'StdVar.png')  
    
    MEAN_TRAIN_RC_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/RightContour/', 'MeanFace.png')
    STD_TRAIN_RC_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/RightContour/', 'StdVar.png')
    
    MEAN_TRAIN_UM_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/UpMouth/', 'MeanFace.png')
    STD_TRAIN_UM_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/UpMouth/', 'StdVar.png')
    
    MEAN_TRAIN_DM_BEST = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/DownMouth/', 'MeanFace.png')
    STD_TRAIN_DM_BEST  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/DownMouth/', 'StdVar.png')
    
    PATH_TO_WEIGHTS_LE  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/LeftEye/', 'LD_usc_iter_100000.caffemodel')
    PATH_TO_DEPLOY_TXT_LE = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/LeftEye/', 'parts_40x40.prototxt')
    predictorLE = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_LE, weightsPath=PATH_TO_WEIGHTS_LE, meanPath=MEAN_TRAIN_LE_BEST, stdPath=STD_TRAIN_LE_BEST)
                  
    PATH_TO_WEIGHTS_RE  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/RightEye/', 'LD_usc_iter_98000.caffemodel')
    PATH_TO_DEPLOY_TXT_RE = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/RightEye/', 'parts_40x40.prototxt')
    predictorRE = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_RE, weightsPath=PATH_TO_WEIGHTS_RE, meanPath=MEAN_TRAIN_RE_BEST, stdPath=STD_TRAIN_RE_BEST)
    
    #PATH_TO_WEIGHTS_Mo  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Mouth/', 'LD_usc_iter_315.caffemodel')
    #PATH_TO_DEPLOY_TXT_Mo = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Mouth/', 'parts_40x40.prototxt') 
    #predictorMo = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_Mo, weightsPath=PATH_TO_WEIGHTS_Mo, meanPath=MEAN_TRAIN_Mo_BEST, stdPath=STD_TRAIN_Mo_BEST)
          
    PATH_TO_WEIGHTS_No  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Nose/', 'LD_usc_iter_100000.caffemodel')
    PATH_TO_DEPLOY_TXT_No = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Nose/', 'parts_40x40.prototxt') 
    predictorNo = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_No, weightsPath=PATH_TO_WEIGHTS_No, meanPath=MEAN_TRAIN_No_BEST, stdPath=STD_TRAIN_No_BEST)
    
    #PATH_TO_WEIGHTS_Co  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Contour/', 'LD_usc_iter_161000.caffemodel')
    #PATH_TO_DEPLOY_TXT_Co = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Contour/', 'parts_40x40.prototxt') 
    #predictorCo = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_Co, weightsPath=PATH_TO_WEIGHTS_Co, meanPath=MEAN_TRAIN_Co_BEST, stdPath=STD_TRAIN_Co_BEST)  
    
    PATH_TO_WEIGHTS_Ch  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Chin/', 'LD_usc_iter_126000.caffemodel')
    PATH_TO_DEPLOY_TXT_Ch = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/Chin/', 'parts_40x40.prototxt') 
    predictorCh = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_Ch, weightsPath=PATH_TO_WEIGHTS_Ch, meanPath=MEAN_TRAIN_Ch_BEST, stdPath=STD_TRAIN_Ch_BEST)
    
    PATH_TO_WEIGHTS_LC  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/LeftContour/', 'LD_usc_iter_59000.caffemodel')
    PATH_TO_DEPLOY_TXT_LC = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/LeftContour/', 'parts_40x40.prototxt') 
    predictorLC = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_LC, weightsPath=PATH_TO_WEIGHTS_LC, meanPath=MEAN_TRAIN_LC_BEST, stdPath=STD_TRAIN_LC_BEST)
    
    PATH_TO_WEIGHTS_RC  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/RightContour/', 'LD_usc_iter_67000.caffemodel')
    PATH_TO_DEPLOY_TXT_RC = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/RightContour/', 'parts_40x40.prototxt') 
    predictorRC = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_RC, weightsPath=PATH_TO_WEIGHTS_RC, meanPath=MEAN_TRAIN_RC_BEST, stdPath=STD_TRAIN_RC_BEST)   
    
    PATH_TO_WEIGHTS_UM  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/UpMouth/', 'LD_usc_iter_202000.caffemodel')
    PATH_TO_DEPLOY_TXT_UM = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/UpMouth/', 'parts_40x40.prototxt') 
    predictorUM = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_UM, weightsPath=PATH_TO_WEIGHTS_UM, meanPath=MEAN_TRAIN_UM_BEST, stdPath=STD_TRAIN_UM_BEST)
    
    PATH_TO_WEIGHTS_DM  = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/DownMouth/', 'LD_usc_iter_117000.caffemodel')
    PATH_TO_DEPLOY_TXT_DM = os.path.join('/data/b0216.yu/LD_Train/Parts/Models/DownMouth/', 'parts_40x40.prototxt') 
    predictorDM = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_DM, weightsPath=PATH_TO_WEIGHTS_DM, meanPath=MEAN_TRAIN_DM_BEST, stdPath=STD_TRAIN_DM_BEST)   
    
    for k in range(399,400):  
      '''
      PATH_TO_WEIGHTS_LE  = os.path.join('/data/b0216.yu/LD_Train/Parts/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_LE = os.path.join('/data/b0216.yu/LD_Train/Parts/', 'parts_40x40.prototxt')
      #predictorLE = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_LE, weightsPath=PATH_TO_WEIGHTS_LE, meanPath=MEAN_TRAIN_LE, stdPath=STD_TRAIN_LE)
                  
      PATH_TO_WEIGHTS_RE  = os.path.join('/data/b0216.yu/LD_Train/Parts/EyeRight/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_RE = os.path.join('/data/b0216.yu/LD_Train/Parts/EyeRight/', 'parts_40x40.prototxt')
      #predictorRE = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_RE, weightsPath=PATH_TO_WEIGHTS_RE, meanPath=MEAN_TRAIN_RE, stdPath=STD_TRAIN_RE)
      
      PATH_TO_WEIGHTS_No  = os.path.join('/data/b0216.yu/LD_Train/Parts/Nose/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_No = os.path.join('/data/b0216.yu/LD_Train/Parts/Nose/', 'parts_40x40.prototxt') 
      #predictorNo = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_No, weightsPath=PATH_TO_WEIGHTS_No, meanPath=MEAN_TRAIN_No, stdPath=STD_TRAIN_No) 
      
      PATH_TO_WEIGHTS_Co  = os.path.join('/data/b0216.yu/LD_Train/Parts/Contour/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_Co = os.path.join('/data/b0216.yu/LD_Train/Parts/Contour/', 'parts_40x40.prototxt') 
      #predictorCo = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_Co, weightsPath=PATH_TO_WEIGHTS_Co, meanPath=MEAN_TRAIN_Co, stdPath=STD_TRAIN_Co) 
      
      PATH_TO_WEIGHTS_Mo  = os.path.join('/data/b0216.yu/LD_Train/Parts/Mouth/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_Mo = os.path.join('/data/b0216.yu/LD_Train/Parts/Mouth/', 'parts_40x40.prototxt') 
      #predictorMo = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_Mo, weightsPath=PATH_TO_WEIGHTS_Mo, meanPath=MEAN_TRAIN_Mo, stdPath=STD_TRAIN_Mo)
      
      PATH_TO_WEIGHTS_Ch  = os.path.join('/data/b0216.yu/LD_Train/Parts/Chin/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_Ch = os.path.join('/data/b0216.yu/LD_Train/Parts/Chin/', 'parts_40x40.prototxt') 
      #predictorCh = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_Ch, weightsPath=PATH_TO_WEIGHTS_Ch, meanPath=MEAN_TRAIN_Ch, stdPath=STD_TRAIN_Ch) 
      
      PATH_TO_WEIGHTS_LC  = os.path.join('/data/b0216.yu/LD_Train/Parts/LeftContour/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_LC = os.path.join('/data/b0216.yu/LD_Train/Parts/LeftContour/', 'parts_40x40.prototxt') 
      #predictorLC = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_LC, weightsPath=PATH_TO_WEIGHTS_LC, meanPath=MEAN_TRAIN_LC, stdPath=STD_TRAIN_LC) 
      
      PATH_TO_WEIGHTS_RC  = os.path.join('/data/b0216.yu/LD_Train/Parts/RightContour/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_RC = os.path.join('/data/b0216.yu/LD_Train/Parts/RightContour/', 'parts_40x40.prototxt') 
      #predictorRC = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_RC, weightsPath=PATH_TO_WEIGHTS_RC, meanPath=MEAN_TRAIN_RC, stdPath=STD_TRAIN_RC) 
      
      PATH_TO_WEIGHTS_UM  = os.path.join('/data/b0216.yu/LD_Train/Parts/UpMouth/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_UM = os.path.join('/data/b0216.yu/LD_Train/Parts/UpMouth/', 'parts_40x40.prototxt') 
      #predictorUM = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_UM, weightsPath=PATH_TO_WEIGHTS_UM, meanPath=MEAN_TRAIN_UM, stdPath=STD_TRAIN_UM) 
      
      PATH_TO_WEIGHTS_DM  = os.path.join('/data/b0216.yu/LD_Train/Parts/DownMouth/', 'LD_usc_iter_%d.caffemodel' % ((k + 1) * 1000))
      PATH_TO_DEPLOY_TXT_DM = os.path.join('/data/b0216.yu/LD_Train/Parts/DownMouth/', 'parts_40x40.prototxt') 
      predictorDM = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT_DM, weightsPath=PATH_TO_WEIGHTS_DM, meanPath=MEAN_TRAIN_DM, stdPath=STD_TRAIN_DM) 
      '''
      testError=ErrorAcum()
      
      ttime = 0.0
      
      for i, dataRow in enumerate(dataRowsTestValid):
      
        start_time = time.time()
        
        fbbox = BBox(dataRow.fbbox.left,dataRow.fbbox.top,dataRow.fbbox.right,dataRow.fbbox.bottom)      
        
        dataRow40 = dataRow.copyCroppedByBBox_68(fbbox)
        image, lm40 = root_predictor.preprocess(dataRow40.image, dataRow40.landmarks_68())
        prediction = root_predictor.predict(image)
        prediction = dataRow40.inverseScaleAndOffset((prediction + 0.5) * GLOBAL_IMAGE_SIZE)
        
        # Rotate & refine - yubing 20160727
        dataRowRot, rotangleT = dataRow.copyCropped_68_rot(prediction,fbbox)
        imageRot, lmRot = root_predictor.preprocess(dataRowRot.image, dataRow40.landmarks_68())
        predictionRot = root_predictor.predict(imageRot)
        predictionRot = dataRowRot.inverseScaleAndOffset((predictionRot + 0.5) * GLOBAL_IMAGE_SIZE)
        prediction = dataRow.reverseRotate(predictionRot, -rotangleT)
        
        # Local part refine - yubing 20160701
        # Left Eye
        dataRowLE, rotangle = dataRow.copyCropped_patch(prediction,'LE',rotangleT)
        imageLE, lm40 = predictorLE.preprocess(dataRowLE.image, dataRow40.landmarks_68())
        predictionLE = predictorLE.predict(imageLE)
        predictionLE = dataRowLE.inverseScaleAndOffset((predictionLE + 0.5) * 40)
        predictionLE = dataRow.reverseRotate(predictionLE, -rotangle)
        
        # Right Eye
        dataRowRE, rotangle = dataRow.copyCropped_patch(prediction,'RE',rotangleT)
        imageRE, lm40 = predictorRE.preprocess(dataRowRE.image, dataRow40.landmarks_68())
        predictionRE = predictorRE.predict(imageRE)
        predictionRE = dataRowRE.inverseScaleAndOffset((predictionRE + 0.5) * 40)
        predictionRE = dataRow.reverseRotate(predictionRE, -rotangle)
        
        # Mouth
        #dataRowMo, rotangle = dataRow.copyCropped_patch(prediction,'Mo',rotangleT)
        #imageMo, lm40 = predictorMo.preprocess(dataRowMo.image, dataRow40.landmarks_68())
        #predictionMo = predictorMo.predict(imageMo)
        #predictionMo = dataRowMo.inverseScaleAndOffset((predictionMo + 0.5) * 40)
        #predictionMo = dataRow.reverseRotate(predictionMo, -rotangle)
        
        # Nose
        dataRowNo, rotangle = dataRow.copyCropped_patch(prediction,'No',rotangleT)
        imageNo, lm40 = predictorNo.preprocess(dataRowNo.image, dataRow40.landmarks_68())
        predictionNo = predictorNo.predict(imageNo)
        predictionNo = dataRowNo.inverseScaleAndOffset((predictionNo + 0.5) * 40)
        predictionNo = dataRow.reverseRotate(predictionNo, -rotangle)
        
        # Contour
        #dataRowCo, rotangle = dataRow.copyCropped_patch(prediction,'Co')
        #imageCo, lm40 = predictorCo.preprocess(dataRowCo.image, dataRow40.landmarks_68())
        #predictionCo = predictorCo.predict(imageCo)
        #predictionCo = dataRowCo.inverseScaleAndOffset((predictionCo + 0.5) * 40)
        #predictionCo = dataRow.reverseRotate(predictionCo, -rotangle)     
        
        # Chin
        dataRowCh, rotangle = dataRow.copyCropped_patch(prediction,'Ch',rotangleT)
        imageCh, lm40 = predictorCh.preprocess(dataRowCh.image, dataRow40.landmarks_68())
        predictionCh = predictorCh.predict(imageCh)
        predictionCh = dataRowCh.inverseScaleAndOffset((predictionCh + 0.5) * 40)
        predictionCh = dataRow.reverseRotate(predictionCh, -rotangle)
        
        # Left Contour
        dataRowLC, rotangle = dataRow.copyCropped_patch(prediction,'LC',rotangleT)
        imageLC, lm40 = predictorLC.preprocess(dataRowLC.image, dataRow40.landmarks_68())
        predictionLC = predictorLC.predict(imageLC)
        predictionLC = dataRowLC.inverseScaleAndOffset((predictionLC + 0.5) * 40)
        predictionLC = dataRow.reverseRotate(predictionLC, -rotangle)
        
        # Right Contour
        dataRowRC, rotangle = dataRow.copyCropped_patch(prediction,'RC',rotangleT)
        imageRC, lm40 = predictorRC.preprocess(dataRowRC.image, dataRow40.landmarks_68())
        predictionRC = predictorRC.predict(imageRC)
        predictionRC = dataRowRC.inverseScaleAndOffset((predictionRC + 0.5) * 40)
        predictionRC = dataRow.reverseRotate(predictionRC, -rotangle)
        
        # Up Mouth
        dataRowUM, rotangle = dataRow.copyCropped_patch(prediction,'UM',rotangleT)
        imageUM, lm40 = predictorUM.preprocess(dataRowUM.image, dataRow40.landmarks_68())
        predictionUM = predictorUM.predict(imageUM)
        predictionUM = dataRowUM.inverseScaleAndOffset((predictionUM + 0.5) * 40)
        predictionUM = dataRow.reverseRotate(predictionUM, -rotangle)
        
        # Down Mouth
        dataRowDM, rotangle = dataRow.copyCropped_patch(prediction,'DM',rotangleT)
        imageDM, lm40 = predictorDM.preprocess(dataRowDM.image, dataRow40.landmarks_68())
        predictionDM = predictorDM.predict(imageDM)
        predictionDM = dataRowDM.inverseScaleAndOffset((predictionDM + 0.5) * 40)
        predictionDM = dataRow.reverseRotate(predictionDM, -rotangle)
        
        # Merge results
        prediction = mergeResult(prediction, predictionLE, 'LE')
        prediction = mergeResult(prediction, predictionRE, 'RE')
        #prediction = mergeResult(prediction, predictionMo, 'Mo')
        prediction = mergeResult(prediction, predictionNo, 'No')
        #prediction = mergeResult(prediction, predictionCo, 'Co')
        prediction = mergeResult(prediction, predictionCh, 'Ch')
        prediction = mergeResult(prediction, predictionLC, 'LC')
        prediction = mergeResult(prediction, predictionRC, 'RC')
        prediction = mergeResult(prediction, predictionUM, 'UM')
        prediction = mergeResult(prediction, predictionDM, 'DM')
        
        prediction = dataRow.fitMargin(prediction)
        
        testError.add_68(dataRow.rawData, prediction.ravel())
        #testError.add_part(dataRow.rawData, prediction.ravel(), 'No')
        #dataRow40.prediction = (prediction + 0.5) * GLOBAL_IMAGE_SIZE

        end_time = time.time()
        total_time = end_time - start_time
        
        print "Time: %f s" % (total_time)
        ttime = ttime + total_time
        
        DEBUG = 0
     
        if DEBUG:
            
            #dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction)
            f = open('/data/b0216.yu/caffe/shapes_whole_1.txt','at')
            #f.write(dataRow.name + ' ')
            for x,y in prediction.reshape(-1,2):
              f.write('%f %f ' % (x,y))
            f.write('\n')
            f.close()
            '''
            f = open('/data/b0216.yu/caffe/gt_all.txt','at')
            #f.write(dataRow.name + ' ')
            for x,y in dataRow.rawData.reshape(-1,2):
              f.write('%f %f ' % (x,y))
            f.write('\n')
            f.close()
            
            f = open('/data/b0216.yu/caffe/68pts.txt','at')
            f.write(dataRow.name + ' ')
            for x,y in prediction.reshape(-1,2):
              f.write('%f %f ' % (x,y))
            f.write('\n')
            f.close()
            '''
            dataRow.prediction = prediction
            #dataRow.show()
            
            # feat = predictor.net.blobs['abs4'].data[0]
            # channels = predictor.net.blobs['abs4'].channels
            # height = predictor.net.blobs['abs4'].height
            # width = predictor.net.blobs['abs4'].width
            
            # shapeT = feat.shape
            # channels = shapeT[0]
            # height = shapeT[1]
            # width = shapeT[2]
            
            # print channels, height, width
            # f = open('/data/b0216.yu/caffe/feat.txt','at')
            # f.write(dataRow.name + ',')
            # for c in range(channels):
            #   for h in range(height):
            #     for w  in range(width):
            #       f.write('%f,' % feat[c,h,w])
            # f.write('\n')
            # f.close()
            
            print "Image Error: %f"  % (testError.meanError_68().mean()*100)
            # break
       
      print "Iter: %d" % (k + 1) 
      print "image Error: %f" % (testError.meanError_68().mean()*100)
      f = open('/data/b0216.yu/caffe/errors_68.txt','at')
      f.write('%d %f\n' % (k + 1, testError.meanError_68().mean()*100))
      f.close()

      print "Avg. time: %f s" % (ttime / 554)
	
if 'ExtraFeat' in STEPS:
	print "Loading image set....."
	dataRowsTest_CSV  = createDataRowsFromCSV(CSV_TEST , DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
	print "Finished reading %d rows from image" % len(dataRowsTest_CSV)
	dataRowsTestValid,R = getValidWithBBox(dataRowsTest_CSV)
	print "Original image:",len(dataRowsTest_CSV), "Valid Rows:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch

# Run the same caffe image set using python
	layerName = 'data'
	testError=ErrorAcum()
	predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
	c = predictor.net.blobs[layerName].channels
	num = predictor.net.blobs[layerName].num		
	h = predictor.net.blobs[layerName].height
	w = predictor.net.blobs[layerName].width		
	count = predictor.net.blobs[layerName].count
	
	for i, dataRow in enumerate(dataRowsTestValid):
		#print "name: ", dataRow.fbbox
		dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
		orgImgpath = featPath + dataRow.name + 'org'
		with  open(orgImgpath,'w') as fi:
			fi.write(dataRow40.image);
		
		image, lm40 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
		prediction = predictor.predict(image)
		
		
		predictor.net.blobs['data'].data[...] = cv2.split(image)
		prediction = predictor.net.forward()['Dense2'][0]
		
		
		feat = predictor.net.blobs[layerName].data[0] 
		
		# print num, c, h, w, count, feat[0,0,0], feat[0,0,1],feat[0,0,2],feat[1,0,0]
		# print feat
		feat = feat.reshape(1, count)		
		
		testError.add(lm40, prediction)
		dataRow40.prediction = (prediction+0.5)*40.		
		fea_name = dataRow.name.replace('.jpg','.'+layerName)	
		fea_file = featPath + fea_name
		
		# with  open(fea_file,'w') as f:
			# f.write(str(num));		
			# f.write(str(feat));
		with  open(fea_file,'w') as f:
			f.write(struct.pack('f',num))
			f.write(struct.pack('f',c))
			f.write(struct.pack('f',h))
			f.write(struct.pack('f',w))
			f.write(struct.pack('f',count))
			f.write(feat);

	print "image features have been extracted to %s\n" % featPath
	print "image Error:", testError