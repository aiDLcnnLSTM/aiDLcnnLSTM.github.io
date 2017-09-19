import os,sys
import numpy as np
import cv2
import random

DEBUG = False

class FaceLandmarkDataReader():
    def __init__(self):
        self.list_path = ""
        self.pts_num = 68
        self.input_image_W = 40
        self.input_image_H = 40
        self.input_image_C = 3
        self.path_prex = ""

        self.all_lines = []
        self.train_lst = []
        self.test_lst = []


    ## read data from list
    ## 300-W_GT_fixFD_20140622.txt
    """
    helen\testset\296814969_3.jpg 300 267 269 269 
    266.000 
    ...
    """
    # read_list all line for debug
    def read_lst_all_lines(self):
        if not os.path.exists(self.list_path):
            raise ValueError('list_path of ' + self.list_path + ' not exists!')

        lst = []
        for line in open(self.list_path):
            line = line
            lst.append(line)

        self.all_lines = lst
        return lst

    def shufle_list(self, train_ratio):
        l = len(self.all_lines)

        if train_ratio < 0 or train_ratio > 1:
            raise ValueError('train_ratio is' + str(train_ratio))

        train_num = int(l*train_ratio)
        all_lines_shufle = np.random.shuffle(self.all_lines)
        all_lines_shufle=self.all_lines
        #print(all_lines_shufle)
        assert all_lines_shufle != None
        self.train_lst = all_lines_shufle[0:train_num]
        self.test_lst = all_lines_shufle[train_num:]


    ### get a image and label from a line
    def get_image_label(self, line="", margin=0.1):
        line = line.strip()
        line = line.replace('\\', '/')
        elems = line.split(' ')
        assert len(elems) == (self.pts_num * 2 + 1 + 4)
        img_path = elems[0]
        box = elems[1:5]
        landmarks = elems[5:]

        assert os.path.exists(self.path_prex + img_path)
        img = cv2.imread(self.path_prex + img_path)
        #assert img!=None

        x0 = float(box[0])
        y0 = float(box[1])
        #w0 = float(box[2])-float(box[0])
        #h0 = float(box[3])-float(box[1])
        w0 = float(box[2])
        h0 = float(box[3])

        margin = np.random.randint(8,20)/100.0
        H,W,_ = img.shape
        x_new = int(round(max(0, x0 - w0 * margin)))
        y_new = int(round(max(0, y0 - h0 * margin)))
        w_new = int(round(min(W, x0 + w0 * (1 + margin)) - x_new))
        h_new = int(round(min(H, y0 + h0 * (1 + margin)) - y_new))

        if DEBUG:
            print(line)
            print((x_new,y_new), (x_new+w_new,y_new+h_new))
            img1 = img.copy()
            cv2.rectangle(img1, (x_new,y_new), (x_new+w_new,y_new+h_new), (0,255,0))
            img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("fd", img1)

            img_org = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            #img_org = img
            cv2.imshow("org", img_org)
            cv2.waitKey(0)

        face_img0 = img[y_new:y_new+h_new, x_new:x_new+w_new]
        face_img = cv2.resize(face_img0, (self.input_image_H, self.input_image_W), cv2.INTER_LINEAR)
        x = face_img / 255.0 * 2.0 - 1.0
        y = np.zeros((self.pts_num * 2), np.float32)
        for i in range(self.pts_num):
            y[i * 2] = (round(float(landmarks[i * 2])) - x_new) / w_new
            y[i * 2 + 1] = (round(float(landmarks[i * 2 + 1])) - y_new) / h_new

        if DEBUG:
            cv2.imshow("face", face_img)
            img_tmp = img.copy()
            for i in range(self.pts_num):
                ptx = int( round(float(landmarks[i*2])) )
                pty = int(round(float(landmarks[i * 2+1])))
                cv2.circle(img_tmp, (ptx,pty), 2, (0,0,255), 2)
            img_tmp = cv2.resize(img_tmp, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("org_kp", img_tmp)

            cv2.imwrite('./tmp_face.jpg', face_img0)
            cv2.imwrite('./tmp_x.jpg', ((x+1.0)/2.0*255.0).astype(np.uint8) )
            img_tmp = self.draw_keypoint(x,y)
            cv2.imwrite('./tmp_kp.jpg', img_tmp)

        return (x,y)


    ### get_next_batch batch_size of images and labels
    def get_next_batch(self, batch_size):
        train_len = len(self.train_lst)
        assert train_len > batch_size
        batch_idx = random.sample( range(train_len), batch_size )

        X = np.zeros( (0, self.input_image_H, self.input_image_W, self.input_image_C), np.float32 )
        Y = np.zeros( (0, self.pts_num*2), np.float32 )
        for i in batch_idx:
            line = self.train_lst[i]
            x1,y1 = self.get_image_label(line)
            x1 = x1[np.newaxis, :,:,:]
            y1 = y1[np.newaxis, :]

            X = np.row_stack( (X,x1) )
            Y = np.row_stack( (Y,y1) )

        return (X, Y)

    # draw_keypoint
    def draw_keypoint(self, x, y):
        img = ((x+1.0)/2.0*255.0).astype(np.uint8)
        H,W,_ = img.shape
        for i in range(self.pts_num):
            ptx = int(round(y[i*2]*W))
            pty = int(round(y[i*2+1]*H))
            cv2.circle(img, (ptx,pty), 1, (0,0,255), 3)

        return img



if __name__ == "__main__":
    print('test')
    face_read = FaceLandmarkDataReader()
    face_read.path_prex = "D:/FaceLandmark/dataset/Data/300W_train_3837/"
    face_read.list_path = "D:/FaceLandmark/dataset/Data/" + "300-W_GT_fixFD_20140622.txt"
    face_read.batch_size = 64

    face_read.input_image_W = 512
    face_read.input_image_H = 512

    face_read.read_lst_all_lines()
    face_read.shufle_list(0.8)

    (X,Y) = face_read.get_next_batch(7)
    for i in range(X.shape[0]):
        img = face_read.draw_keypoint(X[i], Y[i])
        cv2.imshow("debug", img)
        cv2.waitKey(0)

    #print( type(face_read.read_batch_list) )
    #r = face_read.read_batch_list.next()
    #print(r.shape)
    #(batch_X_img, batch_X_face, batch_Y) = r
    #print(batch_X_img.shape)
    #print(batch_X_face.shape)
    #print(batch_Y.shape)