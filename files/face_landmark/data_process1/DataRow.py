# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:30:29 2015

@author: Ishay Tubi
"""

import os
import cv2
import numpy as np
import sys
import csv
import math

global GLOBAL_IMAGE_SIZE
GLOBAL_IMAGE_SIZE = 60.

def getGitRepFolder():
    import subprocess
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip()

def mse_normlized(groundTruth, pred):
    delX = groundTruth[0]-groundTruth[2] 
    delY = groundTruth[1]-groundTruth[3] 
    interOc = (1e-6+(delX*delX + delY*delY))**0.5  # Euclidain distance
    diff = (pred-groundTruth)**2
    sumPairs = (diff[0::2]+diff[1::2])**0.5  # Euclidian distance 
    return (sumPairs / interOc)  # normlized 
    
def mse_normlized_68(groundTruth, pred):
    #delX = (groundTruth[90] + groundTruth[84]) / 2 - (groundTruth[72] + groundTruth[78]) / 2 
    #delY = (groundTruth[91] + groundTruth[85]) / 2 - (groundTruth[73] + groundTruth[79]) / 2 
    delX = groundTruth[90] - groundTruth[72]
    delY = groundTruth[91] - groundTruth[73] 
    interOc = (1e-6+(delX*delX + delY*delY))**0.5  # Euclidain distance
    #pred_t = np.array([(pred[72] + pred[78]) / 2, (pred[73] + pred[79]) / 2, (pred[84] + pred[90]) / 2, (pred[85] + pred[91]) / 2, pred[60], pred[61], pred[96], pred[97], pred[108], pred[109]])
    #grounf_t = np.array([(groundTruth[72] + groundTruth[78]) / 2, (groundTruth[73] + groundTruth[79]) / 2, (groundTruth[84] + groundTruth[90]) / 2, (groundTruth[85] + groundTruth[91]) / 2, groundTruth[60], groundTruth[61], groundTruth[96], groundTruth[97], groundTruth[108], groundTruth[109]])
    #diff = (pred_t - grounf_t)**2
    diff = (pred - groundTruth)**2
    sumPairs = (diff[0::2]+diff[1::2])**0.5  # Euclidian distance 
    return (sumPairs / interOc)  # normlized 

def mse_normlized_part(groundTruth, pred, partName):
    delX = groundTruth[90] - groundTruth[72]
    delY = groundTruth[91] - groundTruth[73] 
    interOc = (1e-6+(delX*delX + delY*delY))**0.5  # Euclidain distance
    
    if cmp(partName,'LE') == 0:
      pred_t = np.array(np.hstack((pred[34:44],pred[72:84])))
      grounf_t = np.array(np.hstack((groundTruth[34:44],groundTruth[72:84])))
    elif cmp(partName,'Mo')== 0:
      pred_t = np.array(pred[96:136])
      grounf_t = np.array(groundTruth[96:136])
    elif cmp(partName,'No')== 0:
      pred_t = np.array(pred[54:72])
      grounf_t = np.array(groundTruth[54:72])
    elif cmp(partName,'RE') == 0:
      pred_t = np.array(np.hstack((pred[44:54],pred[84:96])))
      grounf_t = np.array(np.hstack((groundTruth[44:54],groundTruth[84:96])))
    elif cmp(partName,'Co')== 0:
      pred_t = np.array(pred[0:34])
      grounf_t = np.array(groundTruth[0:34])
    elif cmp(partName,'Ch')== 0:
      pred_t = np.array(pred[12:22])
      grounf_t = np.array(groundTruth[12:22])
    elif cmp(partName,'LC')== 0:
      pred_t = np.array(pred[0:12])
      grounf_t = np.array(groundTruth[0:12])
    elif cmp(partName,'RC')== 0:
      pred_t = np.array(pred[22:34])
      grounf_t = np.array(groundTruth[22:34])
    elif cmp(partName,'UM')== 0:
      pred_t = np.array(np.hstack((pred[96:110],pred[122:128])))
      grounf_t = np.array(np.hstack((groundTruth[96:110],groundTruth[122:128])))
    elif cmp(partName,'DM')== 0:
      pred_t = np.array(np.hstack((pred[110:122],pred[128:136])))
      grounf_t = np.array(np.hstack((groundTruth[110:122],groundTruth[128:136])))

    diff = (pred_t - grounf_t)**2
    sumPairs = (diff[0::2]+diff[1::2])**0.5  # Euclidian distance 
    return (sumPairs / interOc)  # normlized 

def showImage(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
    return 0
    
def mergeResult(prediction, predictionPatch, patchName):
    if cmp(patchName, 'LE') == 0:
      prediction[17:22,:] = predictionPatch[0:5,:].copy()
      prediction[36:42,:] = predictionPatch[5:11,:].copy()
    elif cmp(patchName, 'Mo') == 0:
      prediction[48:68,:] = predictionPatch.copy()
    elif cmp(patchName, 'No') == 0:
      prediction[27:36,:] = predictionPatch.copy()
    elif cmp(patchName, 'RE') == 0:
      prediction[22:27,:] = predictionPatch[0:5,:].copy()
      prediction[42:48,:] = predictionPatch[5:11,:].copy()
    elif cmp(patchName, 'Co') == 0:
      prediction[0:17,:] = predictionPatch.copy()
    elif cmp(patchName, 'Ch') == 0:
      prediction[6:11,:] = predictionPatch.copy()
    elif cmp(patchName, 'LC') == 0:
      prediction[0:6,:] = predictionPatch.copy()
    elif cmp(patchName, 'RC') == 0:
      prediction[11:17,:] = predictionPatch.copy()
    elif cmp(patchName, 'UM') == 0:
      prediction[48:55,:] = predictionPatch[0:7,:].copy()
      prediction[61:64,:] = predictionPatch[7:10,:].copy()
    elif cmp(patchName, 'DM') == 0:
      prediction[55:61,:] = predictionPatch[0:6,:].copy()
      prediction[64:68,:] = predictionPatch[6:10,:].copy()

    return prediction

class RetVal:
    pass  ## A generic class to return multiple values without a need for a dictionary.

def createDataRowsFromCSV(csvFilePath, csvParseFunc, DATA_PATH, limit = sys.maxint):
    ''' Returns a list of DataRow from CSV files parsed by csvParseFunc, 
        DATA_PATH is the prefix to add to the csv file names,
        limit can be used to parse only partial file rows.
    ''' 
    data = []  # the array we build
    validObjectsCounter = 0 
    
    with open(csvFilePath, 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            d = csvParseFunc(row, DATA_PATH)
            if d is not None:
                data.append(d)
                validObjectsCounter += 1
                if (validObjectsCounter > limit ):  # Stop if reached to limit
                    return data 
    return data
    
def getValidWithBBox(dataRows):
    ''' Returns a list of valid DataRow of a given list of dataRows 
    '''
    # import dlib
    R=RetVal()
    
    R.outsideLandmarks = 0 
    R.noImages = 0 
    R.noFacesAtAll = 0 
    R.couldNotMatch = 0
    # detector=dlib.get_frontal_face_detector()

    validRow=[]
    for dataRow in dataRows:        
        if dataRow.image is None or len(dataRow.image)==0:
            R.noImages += 1
        lmd_xy = dataRow.landmarks().reshape([-1,2])
        left,  top = lmd_xy.min( axis=0 )
        right, bot = lmd_xy.max( axis=0 )
                
        # dets = detector( np.array(dataRow.image, dtype = 'uint8' ) );
        
        
        det_bbox = None  # the valid bbox if found
        det_box = BBox.BBoxFromLTRB(dataRow.left, dataRow.top, dataRow.right, dataRow.bottom)
             
        # for det in dets:
        #  det_box = BBox.BBoxFromLTRB(det.left(), det.top(), det.right(), det.bottom())
            
            # Does all landmarks fit into this box?
        # print det_box.top,det_box.bottom,det_box.left,det_box.right,top,bot,left,right
        if top >= det_box.top and bot<= det_box.bottom and left>=det_box.left and right<=det_box.right:
          det_bbox = det_box  
                    
        if det_bbox is None:
            # if len(dets)>0:
            R.couldNotMatch += 1  # For statistics, dlib found faces but they did not match our landmarks.
            # else:
            #    R.noFacesAtAll += 1  # dlib found 0 faces.
        else:
            dataRow.fbbox = det_bbox  # Save the bbox to the data row
            if det_bbox.left<0 or det_bbox.top<0 or det_bbox.right>dataRow.image.shape[1] or det_bbox.bottom>dataRow.image.shape[0]:
                # print det_bbox.left, det_bbox.top, det_bbox.right, det_bbox.bottom, dataRow.image.shape[0], dataRow.image.shape[1]
                R.outsideLandmarks += 1  # Saftey check, make sure nothing goes out of bound.
            else:
                validRow.append(dataRow)  
    
    
    return validRow,R     
    

def getValidWithBBox_68(dataRows):
    ''' Returns a list of valid DataRow of a given list of dataRows 
    '''
    # import dlib
    R=RetVal()
    
    R.outsideLandmarks = 0 
    R.noImages = 0 
    R.noFacesAtAll = 0 
    R.couldNotMatch = 0
    # detector=dlib.get_frontal_face_detector()

    validRow=[]
    for dataRow in dataRows:        
        if dataRow.image is None or len(dataRow.image)==0:
            R.noImages += 1
        lmd_xy = dataRow.landmarks().reshape([-1,2])
        left,  top = lmd_xy.min( axis=0 )
        right, bot = lmd_xy.max( axis=0 )
                
        # dets = detector( np.array(dataRow.image, dtype = 'uint8' ) );
        
        
        det_bbox = None  # the valid bbox if found
        det_box = BBox.BBoxFromLTRB(dataRow.left, dataRow.top, dataRow.right, dataRow.bottom)
        
        ########## yubing 20160408 ##############
        wTemp = det_box.right - det_box.left + 1
        hTemp = det_box.bottom - det_box.top + 1

        ratioTemp = 0.2
        det_box.top -= hTemp * ratioTemp
        det_box.left -= wTemp * ratioTemp
        det_box.bottom += hTemp * ratioTemp
        det_box.right += wTemp * ratioTemp
        
        # det_box.top = max(0, det_box.top)
        # det_box.left = max(0, det_box.left)
        # det_box.bottom = min(dataRow.image.shape[0], det_box.bottom)
        # det_box.right = min(dataRow.image.shape[1], det_box.right)
        
        ########################################
        
        
        # for det in dets:
        #  det_box = BBox.BBoxFromLTRB(det.left(), det.top(), det.right(), det.bottom())
            
            # Does all landmarks fit into this box?
        # print det_box.top,det_box.bottom,det_box.left,det_box.right,top,bot,left,right
        # if top >= det_box.top and bot<= det_box.bottom and left>=det_box.left and right<=det_box.right:
        det_bbox = det_box  
                    
        if det_bbox is None:
            # if len(dets)>0:
            R.couldNotMatch += 1  # For statistics, dlib found faces but they did not match our landmarks.
            # else:
            #    R.noFacesAtAll += 1  # dlib found 0 faces.
        else:
            dataRow.fbbox = det_bbox  # Save the bbox to the data row
            # if det_bbox.left<0 or det_bbox.top<0 or det_bbox.right>dataRow.image.shape[1] or det_bbox.bottom>dataRow.image.shape[0]:
                # print det_bbox.left, det_bbox.top, det_bbox.right, det_bbox.bottom, dataRow.image.shape[0], dataRow.image.shape[1]
            #     R.outsideLandmarks += 1  # Saftey check, make sure nothing goes out of bound.
            # else:
            validRow.append(dataRow)  
    
    
    return validRow,R 
        
def writeHD5(dataRows, outputPath, setTxtFilePATH, meanTrainSet, stdTrainSet , IMAGE_SIZE=GLOBAL_IMAGE_SIZE, mirror=False):
    ''' Create HD5 data set for caffe from given valid data rows.
    if mirror is True, duplicate data by mirroring. 
    ''' 
    from numpy import zeros
    import h5py
    
    if mirror:
        BATCH_SIZE = len(dataRows) *2
    else:
        BATCH_SIZE = len(dataRows) 

    HD5Images = zeros([BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
    HD5Landmarks = zeros([BATCH_SIZE, 10], dtype='float32')
    #prefix  = os.path.join(ROOT, 'caffeData', 'hd5', 'train')
    setTxtFile = open(setTxtFilePATH, 'w')

        
    i = 0 
    
    for dataRowOrig in dataRows:
        if i % 1000 == 0 or i >= BATCH_SIZE-1:
            print "Processing row %d " % (i+1) 
            
        if not hasattr(dataRowOrig, 'fbbox'):
            print "Warning, no fbbox"
            continue
        
        dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox)  # Get a cropped scale copy of the data row
        scaledLM = dataRow.landmarksScaledMinus05_plus05() 
        image = dataRow.image.astype('f4')
        image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)
        
        #HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
        HD5Images[i, :] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   # gray image
        HD5Landmarks[i,:] = scaledLM
        i+=1
        
        if mirror:
            dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox).copyMirrored()  # Get a cropped scale copy of the data row
            scaledLM = dataRow.landmarksScaledMinus05_plus05() 
            image = dataRow.image.astype('f4')
            image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)
            
            #HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
            HD5Images[i, :] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   # gray image
            HD5Landmarks[i,:] = scaledLM
            i+=1
        
        
    with h5py.File(outputPath, 'w') as T:
        T.create_dataset("X", data=HD5Images)
        T.create_dataset("landmarks", data=HD5Landmarks)

    setTxtFile.write(outputPath+"\n")
    setTxtFile.flush()
    setTxtFile.close()
    
    
    
  



class ErrorAcum:  # Used to count error per landmark
    def __init__(self):
        self.errorPerLandmark = np.zeros(5, dtype ='f4')
        self.errorPerLandmark_68 = np.zeros(68, dtype ='f4')
        self.errorPerLandmark_part = np.zeros(9, dtype ='f4')
        self.itemsCounter = 0
        self.failureCounter = 0
        self.curMeanError = 0.
        
    def __repr__(self):
        return '%f mean error, %d items, %d failures  %f precent' % (self.meanError().mean()*100, self.itemsCounter, self.failureCounter, float(self.failureCounter)/self.itemsCounter if self.itemsCounter>0 else 0)
        
        
    def add(self, groundTruth, pred):
        normlized = mse_normlized(groundTruth, pred)
        self.errorPerLandmark += normlized
        self.itemsCounter +=1
        
        if normlized.mean() > 0.05: # yubing 20160408
            # Count error above 5% as failure
            self.failureCounter +=1
            
    def add_68(self, groundTruth, pred):
        normlized = mse_normlized_68(groundTruth, pred)
        self.errorPerLandmark_68 += normlized
        #self.errorPerLandmark += normlized
        
        self.curMeanError = normlized.mean()*100
        #print "Error: ", normlized.mean()*100
        
        self.itemsCounter +=1
        # print "Error: ", normlized.mean()
        if normlized.mean() > 0.05: # yubing 20160408
            # Count error above 5% as failure
            self.failureCounter +=1
            
    def add_part(self, groundTruth, pred, partName):
        normlized = mse_normlized_part(groundTruth, pred, partName)
        self.errorPerLandmark_part += normlized
        #self.errorPerLandmark += normlized
        self.itemsCounter +=1
        # print "Error: ", normlized.mean()
        if normlized.mean() > 0.05: # yubing 20160408
            # Count error above 5% as failure
            self.failureCounter +=1

    def meanError(self):
        if self.itemsCounter > 0:
            return self.errorPerLandmark/self.itemsCounter
        else:
            return self.errorPerLandmark
            
    def meanError_68(self):
        print self.itemsCounter
        if self.itemsCounter > 0:
            return self.errorPerLandmark_68/self.itemsCounter
        else:
            return self.errorPerLandmark_68
            
    def meanError_part(self):
        print self.itemsCounter
        if self.itemsCounter > 0:
            return self.errorPerLandmark_part/self.itemsCounter
        else:
            return self.errorPerLandmark_part

    def __add__(self, x):
        ret = ErrorAcum()
        ret.errorPerLandmark = self.errorPerLandmark + x.errorPerLandmark
        ret.itemsCounter    = self.itemsCounter + x.itemsCounter
        ret.failureCounter  = self.failureCounter + x.failureCounter        
        return ret
        
    def plot(self):
        from matplotlib.pylab import show, plot, stem
        pass


class BBox:  # Bounding box
    
    @staticmethod
    def BBoxFromLTRB(l, t, r, b):
        return BBox(l, t, r, b)
    
    @staticmethod
    def BBoxFromXYWH_array(xywh):
        return BBox(xywh[0], xywh[1], +xywh[0]+xywh[2], xywh[1]+xywh[3])
    
    @staticmethod
    def BBoxFromXYWH(x,y,w,h):
        return BBox(x,y, x+w, y+h)
    
    def top_left(self):
        return (self.top, self.left)
    
    def left_top(self):
        return (self.left, self.top)

    def bottom_right(self):
        return (self.bottom, self.right)
        
    def right_bottom(self):
        return (self.right, self.bottom)
    
    def right_top(self):
        return (self.right, self.top)
    
    def relaxed(self, clip ,relax=3):  #@Unused
        from numpy import array
        _A = array
        maxWidth, maxHeight =  clip[0], clip[1]
        
        nw, nh = self.size()*(1+relax)*.5       
        center = self.center()
        offset=_A([nw,nh])
        lefttop = center - offset
        rightbot= center + offset 
         
        self.left, self.top  = int( max( 0, lefttop[0] ) ), int( max( 0, lefttop[1]) )
        self.right, self.bottom = int( min( rightbot[0], maxWidth ) ), int( min( rightbot[1], maxHeight ) )
        return self

    def clip(self, maxRight, maxBottom):
        self.left = max(self.left, 0)
        self.top = max(self.top, 0)
        self.right = min(self.right, maxRight)
        self.bottom = min(self.bottom, maxBottom)
        
    def size(self):
        from numpy import  array
        return array([self.width(), self.height()])
     
    def center(self):
        from numpy import  array
        return array([(self.left+self.right)/2, (self.top+self.bottom)/2])
                
    def __init__(self,left=0, top=0, right=0, bottom=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        
    def width(self):
        return self.right - self.left
        
    def height(self):
        return self.bottom - self.top
        
    def xywh(self):
        return self.left, self.top, self.width(), self.height()
        
    def offset(self, x, y):
        self.left += x 
        self.right += x
        self.top += y 
        self.bottom += y
         
    def scale(self, rx, ry):
        self.left *= rx 
        self.right *= rx
        self.top *= ry 
        self.bottom *= ry
                        
    def __repr__(self):
        return 'left(%.1f), top(%.1f), right(%.1f), bottom(%.1f) w(%d) h(%d)' % (self.left, self.top, self.right, self.bottom,self.width(), self.height())

    def makeInt(self):
        self.left    = int(self.left)
        self.top     = int(self.top)
        self.right   = int(self.right)
        self.bottom  = int(self.bottom)
        return self



class DataRow:
    global TrainSetMean
    global TrainSetSTD
    
    IMAGE_SIZE = GLOBAL_IMAGE_SIZE
    def __init__(self, path='', leftEye=(0, 0, ), rightEye=(0, 0), middle=(0, 0), leftMouth=(0, 0), rightMouth=(0, 0)):
        self.image = cv2.imread(path)
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.leftMouth = leftMouth
        self.rightMouth = rightMouth
        self.middle = middle
        self.name = os.path.split(path)[-1]
        self.sx = 1.
        self.sy = 1.
        self.offsetX = 0.
        self.offsetY = 0.
        self.top = 0
        self.bottom = 0
        self.left = 0
        self.right = 0
        self.rawData = np.zeros(136)

    def __repr__(self):
        return '{} le:{},{} re:{},{} nose:{},{}, lm:{},{} rm:{},{}'.format(
            self.name,
            self.leftEye[0], self.leftEye[1],
            self.rightEye[0], self.rightEye[1],
            self.middle[0], self.middle[1],
            self.leftMouth[0], self.leftMouth[1],
            self.rightMouth[0], self.rightMouth[1]
            )

    def setLandmarks(self,landMarks):
        """
        @landMarks : np.array
        set the landmarks from array
        """
        self.leftEye = landMarks[0:2]
        self.rightEye = landMarks[2:4]
        self.middle = landMarks[4:6]
        self.leftMouth = landMarks[6:8]
        self.rightMouth = landMarks[8:10]
        
    def setLandmarks_68(self,landMarks):
        """
        @landMarks : np.array
        set the landmarks from array
        """
        self.leftEye = ((landMarks[72] + landMarks[78]) / 2, (landMarks[73] + landMarks[79]) / 2)
        self.rightEye = ((landMarks[84] + landMarks[90]) / 2, (landMarks[85] + landMarks[91]) / 2)
        self.middle = landMarks[60:62]
        self.leftMouth = landMarks[96:98]
        self.rightMouth = landMarks[108:110]
        self.rawData = landMarks.copy()
        
        
    def landmarks(self):
        # return numpy float array with ordered values
        stright = [
            self.leftEye[0],
            self.leftEye[1],
            self.rightEye[0],
            self.rightEye[1],
            self.middle[0],
            self.middle[1],
            self.leftMouth[0],
            self.leftMouth[1],
            self.rightMouth[0],
            self.rightMouth[1]]
        return np.array(stright, dtype='f4')
        
    def landmarks_68(self):
        lm = self.rawData.copy()
        return lm

    def landmarksScaledMinus05_plus05(self):
        # return numpy float array with ordered values
        return self.landmarks().astype('f4')/GLOBAL_IMAGE_SIZE - 0.5
        
    def scale(self, sx, sy):
        self.sx *= sx
        self.sy *= sy

        self.leftEye = (self.leftEye[0]*sx, self.leftEye[1]*sy)
        self.rightEye = (self.rightEye[0]*sx, self.rightEye[1]*sy)
        self.middle = (self.middle[0]*sx, self.middle[1]*sy)
        self.leftMouth = (self.leftMouth[0]*sx, self.leftMouth[1]*sy)
        self.rightMouth = (self.rightMouth[0]*sx, self.rightMouth[1]*sy)
        
        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1, 2)*[sx, sy]

        return self
        
    def scale_68(self, sx, sy):
        self.sx *= sx
        self.sy *= sy

        self.leftEye = (self.leftEye[0]*sx, self.leftEye[1]*sy)
        self.rightEye = (self.rightEye[0]*sx, self.rightEye[1]*sy)
        self.middle = (self.middle[0]*sx, self.middle[1]*sy)
        self.leftMouth = (self.leftMouth[0]*sx, self.leftMouth[1]*sy)
        self.rightMouth = (self.rightMouth[0]*sx, self.rightMouth[1]*sy)
        
        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1, 2)*[sx, sy]
            
        for i in range(68):
          self.rawData[2 * i] *= sx
          self.rawData[2 * i + 1] *= sy

        return self
        
    def scale_patch(self, sx, sy):
        self.sx *= sx
        self.sy *= sy
        
        return self

    def offsetCropped(self, offset=(0., 0.)):
        """ given the cropped values - offset the positions by offset
        """
        self.offsetX -= offset[0]
        self.offsetY -= offset[1]

        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1,2)-offset


        self.leftEye = (self.leftEye[0]-offset[0], self.leftEye[1]-offset[1])
        self.rightEye = (self.rightEye[0]-offset[0], self.rightEye[1]-offset[1])
        self.middle = (self.middle[0]-offset[0], self.middle[1]-offset[1])
        self.leftMouth = (self.leftMouth[0]-offset[0], self.leftMouth[1]-offset[1])
        self.rightMouth = (self.rightMouth[0]-offset[0], self.rightMouth[1]-offset[1])
        return self
        
    def offsetCropped_68(self, offset=(0., 0.)):
        """ given the cropped values - offset the positions by offset
        """
        self.offsetX -= offset[0]
        self.offsetY -= offset[1]

        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1,2)-offset


        self.leftEye = (self.leftEye[0]-offset[0], self.leftEye[1]-offset[1])
        self.rightEye = (self.rightEye[0]-offset[0], self.rightEye[1]-offset[1])
        self.middle = (self.middle[0]-offset[0], self.middle[1]-offset[1])
        self.leftMouth = (self.leftMouth[0]-offset[0], self.leftMouth[1]-offset[1])
        self.rightMouth = (self.rightMouth[0]-offset[0], self.rightMouth[1]-offset[1])
        
        for i in range(68):
          self.rawData[2 * i] -= offset[0]
          self.rawData[2 * i + 1] -= offset[1]  
        
        return self
        
    def offsetCropped_patch(self, offset=(0.,0.)):
        self.offsetX -= offset[0]
        self.offsetY -= offset[1]
        
        return self

    def inverseScaleAndOffset(self, landmarks):
        """ computes the inverse scale and offset of input data according to the inverse scale factor and inverse offset factor
        """
        from numpy import array; _A = array ; # Shothand 
        
        ret = _A(landmarks.reshape(-1,2)) *_A([1./self.sx, 1./self.sy])
        ret += _A([-self.offsetX, -self.offsetY])
        return ret

    @staticmethod
    def DataRowFromNameBoxInterlaved(row, root=''):  # lfw_5590 + net_7876 (interleaved) 
        '''
        name , bounding box(w,h), left eye (x,y) ,right eye (x,y)..nose..left mouth,..right mouth
        '''
        d = DataRow()
        d.path = os.path.join(root, row[0]).replace("\\", "/")
        # d.name = os.path.split(d.path)[-1]
        d.name = row[0]
        d.image = cv2.imread(d.path)
    
        sig_diff = math.sqrt(1.6 * 1.6 - 0.5 * 0.5)
        d.image = cv2.GaussianBlur(d.image, (3,3), sig_diff)
        
        d.left = int(row[1])
        d.top = int(row[2])
        d.right = int(row[1]) + int(row[3]) - 1
        d.bottom = int(row[2]) + int(row[4]) - 1
        d.leftEye = (float(row[5]), float(row[6]))
        d.rightEye = (float(row[7]), float(row[8]))
        d.middle = (float(row[9]), float(row[10]))
        d.leftMouth = (float(row[11]), float(row[12]))
        d.rightMouth = (float(row[13]), float(row[14]))

        return d
     
    @staticmethod   
    def DataRowFromNameBoxInterlaved_68(row, root=''):  # lfw_5590 + net_7876 (interleaved) 
        '''
        name , bounding box(w,h), left eye (x,y) ,right eye (x,y)..nose..left mouth,..right mouth
        '''
        d = DataRow()
        d.path = os.path.join(root, row[0]).replace("\\", "/")
        # d.name = os.path.split(d.path)[-1]
        d.name = row[0]
        d.image = cv2.imread(d.path)
        
        sig_diff = math.sqrt(1.6 * 1.6 - 0.5 * 0.5)
        d.image = cv2.GaussianBlur(d.image, (3,3), sig_diff)
        
        d.left = int(row[1])
        d.top = int(row[2])
        d.right = int(row[1]) + int(row[3]) - 1
        d.bottom = int(row[2]) + int(row[4]) - 1
        #d.leftEye = ((float(row[72]) + float(row[78])) / 2, (float(row[73]) + float(row[79])) / 2)
        #d.rightEye = ((float(row[84]) + float(row[90])) / 2, (float(row[85]) + float(row[91])) / 2)
        #d.middle = (float(row[60]), float(row[61]))
        #d.leftMouth = (float(row[96]), float(row[97]))
        #d.rightMouth = (float(row[108]), float(row[109]))
        for i in range(136):
          d.rawData[i] = float(row[i + 5])
        return d

    @staticmethod
    def DataRowFromMTFL(row, root=''):
        '''
        --x1...x5,y1...y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
        '''
        d = DataRow()
        if len(row[0]) <= 1:
            # bug in the files, it has spaces seperating them, skip it
            row=row[1:]
            
        if len(row)<10:
            print 'error parsing ', row
            return None

        d.path = os.path.join(root, row[0]).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        
        if d.image is None:
            print 'Error reading image', d.path
            return None
        
        d.leftEye = (float(row[1]), float(row[6]))
        d.rightEye = (float(row[2]), float(row[7]))
        d.middle = (float(row[3]), float(row[8]))
        d.leftMouth = (float(row[4]), float(row[9]))
        d.rightMouth = (float(row[5]), float(row[10]))
        return d

    @staticmethod
    def DataRowFromAFW(anno, root=''): # Assume data comming from parsed anno-v7.mat file.
        name = str(anno[0][0])
#        bbox = anno[1][0][0]
#        yaw, pitch, roll = anno[2][0][0][0]
        lm = anno[3][0][0]  # 6 landmarks

        if np.isnan(lm).any():
            return None  # Fail

        d = DataRow()
        d.path = os.path.join(root, name).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        d.leftEye = (float(lm[0][0]), float(lm[0][1]))
        d.rightEye = (float(lm[1][0]), float(lm[1][1]))
        d.middle = (float(lm[2][0]), float(lm[2][1]))
        d.leftMouth = (float(lm[3][0]), float(lm[3][1]))
        # skip point 4 middle mouth - We take 0 left eye, 1 right eye, 2 nose, 3 left mouth, 5 right mouth
        d.rightMouth = (float(lm[5][0]), float(lm[5][1]))

        return d

    @staticmethod
    def DataRowFromPrediction(p, path='', image=None):
        d = DataRow(path)        
        p = (p+0.5)*100.  # scale from -0.5..+0.5 to 0..40
        
        d.leftEye = (p[0], p[1])
        d.rightEye = (p[2], p[3])
        d.middle = (p[4], p[5])
        d.leftMouth = (p[6], p[7])
        d.rightMouth = (p[8], p[9])

        return d

    def drawLandmarks(self, r=2, color=255, other=None, title=None):
        M = self.image
        if hasattr(self, 'prediction'):
            for x,y in self.prediction.reshape(-1,2):
                cv2.circle(M, (int(x), int(y)), r, (0, 255, 255), -1)       
                
        #if hasattr(self, 'rawData'):
        #    for x,y in self.rawData.reshape(-1,2):
        #        cv2.circle(M, (int(x), int(y)), r, (0, 0, 255), -1)      

        #cv2.circle(M, (int(self.leftEye[0]), int(self.leftEye[1])), r, (0, 0, 255), -1)
        #cv2.circle(M, (int(self.rightEye[0]), int(self.rightEye[1])), r, (0, 0, 255), -1)
        #cv2.circle(M, (int(self.leftMouth[0]), int(self.leftMouth[1])), r, (0, 0, 255), -1)
        #cv2.circle(M, (int(self.rightMouth[0]), int(self.rightMouth[1])), r, (0, 0, 255), -1)
        #cv2.circle(M, (int(self.middle[0]), int(self.middle[1])), r, (0, 0, 255), -1)
        #if hasattr(self, 'fbbox'):
            # cv2.rectangle(M, self.fbbox.top_left(), self.fbbox.bottom_right(), color)
            # cv2.rectangle(M, self.fbbox.left_top(), self.fbbox.right_bottom(), (0, 255, 0))
        return M

    def show(self, r=2, color=255, other=None, title=None):
        M = self.drawLandmarks(r, color, other, title)
        # f = open('/data/b0216.yu/caffe/feat.txt','at')
        # f.write(self.name + ',')
        # for x in self.prediction:
        #  f.write('%f,' % x)
        # f.write('\n')
        # f.close()
        
        #cv2.imwrite('/data/b0216.yu/caffe/res/'+self.name, M)
        
        if title is None:
          title = self.name
        if M.shape[0] > 800:
          wTemp = int(800. / M.shape[0] * M.shape[1])
          MShow = cv2.resize(M, (wTemp, 800))
          showImage(title, MShow)
        else:
          showImage(title, M)
          
        return M
        
    def makeInt(self):
        self.leftEye    = (int(self.leftEye[0]), int(self.leftEye[1]))
        self.rightEye   = (int(self.rightEye[0]), int(self.rightEye[1]))
        self.middle     = (int(self.middle[0]), int(self.middle[1]))
        self.leftMouth  = (int(self.leftMouth[0]), int(self.leftMouth[1]))
        self.rightMouth = (int(self.rightMouth[0]), int(self.rightMouth[1]))
        return self        
         
    def copyCroppedByBBox(self,fbbox, siz=np.array([GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE])):
        """
        @ fbbox : BBox
        Returns a copy with cropped, scaled to size
        """  
        
        # print fbbox.top, fbbox.left, fbbox.bottom, fbbox.right
        ########## yubing 20160408 ##############
        wTemp = fbbox.right - fbbox.left + 1
        hTemp = fbbox.bottom - fbbox.top + 1

        ratioTemp = 0.0
        fbbox.top -= hTemp * ratioTemp
        fbbox.left -= wTemp * ratioTemp
        fbbox.bottom += hTemp * ratioTemp
        fbbox.right += wTemp * ratioTemp
        
        fbbox.top = max(0, fbbox.top)
        fbbox.left = max(0, fbbox.left)
        fbbox.bottom = min(self.image.shape[0], fbbox.bottom)
        fbbox.right = min(self.image.shape[1], fbbox.right)
        
        
        # print fbbox.top, fbbox.left, fbbox.bottom, fbbox.right
        ########################################
              
        fbbox.makeInt() # assume BBox class
        if fbbox.width()<10 or fbbox.height()<10:
            print "Invalid bbox size:",fbbox
            return None
            
        faceOnly = self.image[fbbox.top : fbbox.bottom, fbbox.left:fbbox.right, :]
        scaled = DataRow() 
        scaled.image = cv2.resize(faceOnly, (int(siz[0]), int(siz[1])))       
        scaled.setLandmarks(self.landmarks())        
        """ @scaled: DataRow """
        scaled.offsetCropped(fbbox.left_top()) # offset the landmarks
        rx, ry = siz/faceOnly.shape[:2]
        scaled.scale(rx, ry)
        
        return scaled       
        
    def copyCroppedByBBox_68(self,fbbox, siz=np.array([GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE])):
        """
        @ fbbox : BBox
        Returns a copy with cropped, scaled to size
        """  
        
        # print fbbox.top, fbbox.left, fbbox.bottom, fbbox.right
              
        fbbox.makeInt() # assume BBox class
        if fbbox.width()<10 or fbbox.height()<10:
            print "Invalid bbox size:",fbbox
            return None
        #if fbbox.left >= 0 and fbbox.right < self.image.shape[1] and fbbox.top >= 0 and fbbox.bottom < self.image.shape[0]:
        #  faceOnly = self.image[fbbox.top : fbbox.bottom, fbbox.left:fbbox.right, :]
        #else:
        #  faceOnly = np.zeros((fbbox.bottom-fbbox.top+1, fbbox.right-fbbox.left+1, 3),dtype=self.image.dtype)
        #  for h in range(fbbox.top, fbbox.bottom):
        #    if h < 0 or h >= self.image.shape[0]:
        #      continue
        #    for w in range(fbbox.left, fbbox.right):
        #      if w < 0 or w >= self.image.shape[1]:
        #        continue
        #      else:
        #        faceOnly[h-fbbox.top,w-fbbox.left,:] = self.image[h,w,:]
                
        topT = max(0, fbbox.top)
        leftT = max(0, fbbox.left)
        bottomT = min(self.image.shape[0], fbbox.bottom)
        rightT = min(self.image.shape[1], fbbox.right)
        faceOnly = np.zeros((fbbox.bottom-fbbox.top+1, fbbox.right-fbbox.left+1, 3),dtype=self.image.dtype)
        faceOnly[topT-fbbox.top:bottomT-fbbox.top, leftT-fbbox.left:rightT-fbbox.left, :] = self.image[topT:bottomT,leftT:rightT,:].copy()    
        
        scaled = DataRow() 
        scaled.image = cv2.resize(faceOnly, (int(siz[0]), int(siz[1])))      
        scaled.setLandmarks_68(self.landmarks_68())     
        """ @scaled: DataRow """
        scaled.offsetCropped_68(fbbox.left_top()) # offset the landmarks
        rx, ry = siz/faceOnly.shape[:2]
        scaled.scale_68(rx, ry)
        
        return scaled
        
    def copyCropped_68_rot(self,landmarks,fbbox,siz=np.array([GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE])):
        
        angle = math.atan2(landmarks[45,1]-landmarks[36,1],landmarks[45,0]-landmarks[36,0])
        #hLM = landmarks[:,1].max()-landmarks[:,1].min() + 1
        #wLM = landmarks[:,0].max()-landmarks[:,0].min() + 1
        
        height = self.image.shape[0]
        width = self.image.shape[1]
        angle = math.degrees(angle)
        rotmat = cv2.getRotationMatrix2D((width/2,height/2), angle, 1.)
        imageRot = cv2.warpAffine(self.image, rotmat, (width,height))
        
        cx = (fbbox.left + fbbox.right) / 2
        cy = (fbbox.top + fbbox.bottom) / 2
        cx_new = cx * rotmat[0,0] + cy * rotmat[0,1] + rotmat[0,2]
        cy_new = cx * rotmat[1,0] + cy * rotmat[1,1] + rotmat[1,2]
        
        w = fbbox.width()
        h = fbbox.height()
        #print w,h,wLM,hLM
        #s = max(wLM,hLM)
        #s = (landmarks[33,1] - landmarks[27,1]) / 0.364
        #w = s * 1.4
        #h = s * 1.4
        
        fbbox.top = cy_new - h / 2
        fbbox.bottom = cy_new + h / 2
        fbbox.left = cx_new - w / 2
        fbbox.right = cx_new + w / 2
        fbbox.makeInt()
         
        topT = max(0, fbbox.top)
        leftT = max(0, fbbox.left)
        bottomT = min(self.image.shape[0], fbbox.bottom)
        rightT = min(self.image.shape[1], fbbox.right)
        faceOnly = np.zeros((fbbox.bottom-fbbox.top+1, fbbox.right-fbbox.left+1, 3),dtype=self.image.dtype)
        faceOnly[topT-fbbox.top:bottomT-fbbox.top, leftT-fbbox.left:rightT-fbbox.left, :] = imageRot[topT:bottomT,leftT:rightT,:].copy()    
        
        scaled = DataRow() 
        scaled.image = cv2.resize(faceOnly, (int(siz[0]), int(siz[1])))      
        scaled.setLandmarks_68(self.landmarks_68())     
        """ @scaled: DataRow """
        scaled.offsetCropped_68(fbbox.left_top()) # offset the landmarks
        rx, ry = siz/faceOnly.shape[:2]
        scaled.scale_68(rx, ry)
        
        return scaled, angle
        
    def copyCropped_patch(self,landmarks,patchName,angleT):
        
        ratio = 1.5
        if cmp(patchName,'LE') == 0:
          patchLM = np.vstack((landmarks[17:22,:],landmarks[36:42,:])).copy()
          angle = math.atan2(patchLM[8,1]-patchLM[5,1],patchLM[8,0]-patchLM[5,0])
        elif cmp(patchName,'Mo')== 0:
          patchLM = landmarks[48:68,:].copy()
          angle = math.atan2(patchLM[6,1]-patchLM[0,1],patchLM[6,0]-patchLM[0,0])
          ratio = 1.3
        elif cmp(patchName,'No')== 0:
          patchLM = landmarks[27:36,:].copy()
          angle = math.atan2(patchLM[8,1]-patchLM[4,1],patchLM[8,0]-patchLM[4,0])
        elif cmp(patchName,'RE') == 0:
          patchLM = np.vstack((landmarks[22:27,:],landmarks[42:48,:])).copy()
          angle = math.atan2(patchLM[8,1]-patchLM[5,1],patchLM[8,0]-patchLM[5,0])
        elif cmp(patchName,'Co')== 0:
          patchLM = landmarks[0:17,:].copy()
          angle = math.atan2(patchLM[16,1]-patchLM[0,1],patchLM[16,0]-patchLM[0,0])
          ratio = 1.2
        elif cmp(patchName,'Ch')== 0:
          patchLM = landmarks[6:11,:].copy()
          #angle = math.atan2(patchLM[4,1]-patchLM[0,1],patchLM[4,0]-patchLM[0,0])
          angle = math.radians(angleT)
        elif cmp(patchName,'LC')== 0:
          patchLM = landmarks[0:6,:].copy()
          angle = math.radians(angleT)
        elif cmp(patchName,'RC')== 0:
          patchLM = landmarks[11:17,:].copy()
          angle = math.radians(angleT)
        elif cmp(patchName,'UM') == 0:
          patchLM = np.vstack((landmarks[48:55,:],landmarks[61:64,:])).copy()
          angle = math.atan2(patchLM[6,1]-patchLM[0,1],patchLM[6,0]-patchLM[0,0])
          ratio = 1.3
        elif cmp(patchName,'DM') == 0:
          patchLM = np.vstack((landmarks[55:61,:],landmarks[64:68,:])).copy()
          angle = math.atan2(patchLM[6,1]-patchLM[5,1],patchLM[6,0]-patchLM[5,0])
          ratio = 1.3
        
        height = self.image.shape[0]
        width = self.image.shape[1]
        angle = math.degrees(angle)
        rotmat = cv2.getRotationMatrix2D((width/2,height/2), angle, 1.)
        imageRot = cv2.warpAffine(self.image, rotmat, (width,height))
        
        numLM = patchLM.shape[0]
        for i in range(numLM):
          x = patchLM[i,0]
          y = patchLM[i,1]
          patchLM[i,0] = x * rotmat[0,0] + y * rotmat[0,1] + rotmat[0,2]
          patchLM[i,1] = x * rotmat[1,0] + y * rotmat[1,1] + rotmat[1,2]
         
        top = patchLM.min(axis=0)
        bottom = patchLM.max(axis=0)
        size = bottom - top
        '''
        if cmp(patchName,'Co') == 0:
          size_new = size * ratio
          size_new[0] = int(size_new[0])
          size_new[1] = int(size_new[1])
        else:
          size_t = int(size.max() * ratio)
          size_new = [size_t,size_t]
        '''
        size_t = int(size.max() * ratio)
        size_new = [size_t,size_t]
        
        top[0] = int(top[0] + size[0] / 2 - size_new[0] / 2)
        top[1] = int(top[1] + size[1] / 2 - size_new[1] / 2)
        top_new = top.copy()
        top_new[0] = max(top[0],0)
        top_new[1] = max(top[1],0)
        bottom = top + size_new
        bottom_new = bottom.copy()
        bottom_new[0] = min(bottom[0],imageRot.shape[1])
        bottom_new[1] = min(bottom[1],imageRot.shape[0])
        #print size_new,top,bottom,top_new,bottom_new
        localPatch = np.zeros((size_new[1],size_new[0],3),dtype=self.image.dtype)
        localPatch[top_new[1]-top[1]:bottom_new[1]-top[1],top_new[0]-top[0]:bottom_new[0]-top[0],:] = imageRot[top_new[1]:bottom_new[1],top_new[0]:bottom_new[0],:].copy()
        #localPatch[top_new[1]-top[1]:bottom_new[1]-top[1],top_new[0]-top[0]:bottom_new[0]-top[0],:] = self.image[top_new[1]:bottom_new[1],top_new[0]:bottom_new[0],:].copy()
        
        scaled = DataRow() 
        scaled.image = cv2.resize(localPatch, (40, 40))      
        
        scaled.offsetCropped_patch(top) # offset the landmarks
        ry, rx = np.array([40.,40.])/localPatch.shape[:2]
        #print rx,ry
        scaled.scale_patch(rx, ry)
        
        #showImage('Test',scaled.image)
        
        return scaled, angle
        
        
    def reverseRotate(self,landmarks,angle):
        numLM = landmarks.shape[0]
        height = self.image.shape[0]
        width = self.image.shape[1]
        rotmat = cv2.getRotationMatrix2D((width/2,height/2),angle,1.)
        for i in range(numLM):
          x = landmarks[i,0]
          y = landmarks[i,1]
          
          landmarks[i,0] = x * rotmat[0,0] + y * rotmat[0,1] + rotmat[0,2]
          landmarks[i,1] = x * rotmat[1,0] + y * rotmat[1,1] + rotmat[1,2]
    
        return landmarks
        
    def fitMargin(self, landmarks):
        numLM = landmarks.shape[0]
        height = self.image.shape[0]
        width = self.image.shape[1]
        for i in range(numLM):
          landmarks[i,0] = min(width - 1,max(0,landmarks[i,0]))
          landmarks[i,1] = min(height - 1,max(0,landmarks[i,1]))
        
        return landmarks
        
    def copyMirrored(self):
        '''
        Return a copy with mirrored data (and mirrored landmarks).
        '''
        import numpy
        _A=numpy.array
        ret = DataRow() 
        ret.image=cv2.flip(self.image.copy(),1)
        # Now we mirror the landmarks and swap left and right
        width = ret.image.shape[0] 
        ret.leftEye = _A([width-self.rightEye[0], self.rightEye[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.rightEye = _A([width-self.leftEye[0], self.leftEye[1]])
        ret.middle = _A([width-self.middle[0], self.middle[1]])        
        ret.leftMouth = _A([width-self.rightMouth[0], self.rightMouth[1]]) # Toggle mouth positions and mirror x axis only
        ret.rightMouth = _A([width-self.leftMouth[0], self.leftMouth[1]])
        return ret

    @staticmethod
    def dummyDataRow():
        ''' Returns a dummy dataRow object to play with
        '''
        return DataRow('/Users/ishay/Dev/VanilaCNN/data/train/lfw_5590/Abbas_Kiarostami_0001.jpg',
                     leftEye=(106.75, 108.25),
                     rightEye=(143.75,108.75) ,
                     middle = (131.25, 127.25),
                     leftMouth = (106.25, 155.25),
                     rightMouth =(142.75,155.25),
                     
                     )    
        
  
            
class Predictor:
    ROOT = getGitRepFolder() 
    
    def preprocess(self, resized, landmarks):
        ret = (cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)).astype('f4')
        # ret = resized.astype('f4')
        ret -= self.mean
        ret /= (1.e-6 + self.std)
        return  ret, (landmarks / GLOBAL_IMAGE_SIZE) - 0.5
    
    def predict(self, resized):
        """
        @resized: image 40,40 already pre processed 
        """     
        self.net.blobs['data'].data[...] = cv2.split(resized)
        prediction = self.net.forward()['ip2'][0]
        # prediction = self.net.forward()['Dense2'][0]
        # print prediction
        return prediction	
        
    def __init__(self, protoTXTPath, weightsPath, meanPath, stdPath):
        import caffe
        caffe.set_mode_cpu()
        #caffe.set_mode_gpu()
        #caffe.set_device(2)
        
        self.net = caffe.Net(protoTXTPath, weightsPath, caffe.TEST)
        # self.mean = cv2.imread(os.path.join('/data/b0216.yu/caffe/caffe_root/models/VanillaCNN-master/', 'MeanFace.png'),0).astype('f4')
        # self.std  = cv2.imread(os.path.join('/data/b0216.yu/caffe/caffe_root/models/VanillaCNN-master/','StdVar.png'),0).astype('f4')
        self.mean = cv2.imread(meanPath,0).astype('f4')
        self.std  = cv2.imread(stdPath,0).astype('f4')
        #showImage('Test',self.mean.astype('uint8'))
        # print Predictor.ROOT
        # self.mean = cv2.imread(os.path.join(Predictor.ROOT, 'trainMean.png')).astype('f4')
        # self.std  = cv2.imread(os.path.join(Predictor.ROOT,'trainSTD.png')).astype('f4')


    