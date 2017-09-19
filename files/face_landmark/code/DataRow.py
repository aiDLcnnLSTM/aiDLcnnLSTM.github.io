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
    return './'
    #import subprocess
    #return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip()

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
    #interOc = (1e-6 + (delX * delX + delY * delY)) ** 0.5  # Euclidain distance

    # reduce error
    interOc = (1e-12+(delX*delX + delY*delY))**0.5  # Euclidain distance


    #pred_t = np.array([(pred[72] + pred[78]) / 2, (pred[73] + pred[79]) / 2, (pred[84] + pred[90]) / 2, (pred[85] + pred[91]) / 2, pred[60], pred[61], pred[96], pred[97], pred[108], pred[109]])
    #grounf_t = np.array([(groundTruth[72] + groundTruth[78]) / 2, (groundTruth[73] + groundTruth[79]) / 2, (groundTruth[84] + groundTruth[90]) / 2, (groundTruth[85] + groundTruth[91]) / 2, groundTruth[60], groundTruth[61], groundTruth[96], groundTruth[97], groundTruth[108], groundTruth[109]])
    #diff = (pred_t - grounf_t)**2
    diff = (pred - groundTruth)**2
    sumPairs = (diff[0::2]+diff[1::2])**0.5  # Euclidian distance 
    return (sumPairs / interOc)  # normlized 




class RetVal:
    pass  ## A generic class to return multiple values without a need for a dictionary.


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
        

class ErrorAcum:  # Used to count error per landmark
    def __init__(self):
        self.errorPerLandmark = np.zeros(5, dtype ='f4')
        self.errorPerLandmark_68 = np.zeros(68, dtype ='f4')
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
            print("error + 1")
            self.failureCounter +=1
            
    def add_68(self, groundTruth, pred):
        normlized = mse_normlized_68(groundTruth, pred)
        self.errorPerLandmark_68 += normlized
        #self.errorPerLandmark += normlized
        
        self.curMeanError = normlized.mean()*100
        #print("Error: ", normlized.mean()*100)
        
        self.itemsCounter +=1
        # print("Error: ", normlized.mean())
        if normlized.mean() > 0.05: # yubing 20160408
            # Count error above 5% as failure
            self.failureCounter +=1


    def meanError(self):
        if self.itemsCounter > 0:
            return self.errorPerLandmark/self.itemsCounter
        else:
            return self.errorPerLandmark
            
    def meanError_68(self):
        print(self.itemsCounter)
        if self.itemsCounter > 0:
            return self.errorPerLandmark_68/self.itemsCounter
        else:
            return self.errorPerLandmark_68
            
    def meanError_part(self):
        print(self.itemsCounter)
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


