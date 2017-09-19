import os,sys
import inspect
import numpy as np
import h5py
import cv2
import time
import random
caffe_root = '/home/hui89.liu/workspace1/face_landmark1/Face_Alignment_Two_Stage_Re-initialization-master/caffe/'
sys.path.append(caffe_root)
import caffe
import lmdb

import imgaug as ia
from imgaug import augmenters as iaa



DEBUG = False


class FaceLandmarkDataReader():
    def __init__(self):
        self.list_path = ""
        self.pts_num = 68
        self.input_image_W = 224
        self.input_image_H = 224
        self.input_image_C = 3
        self.path_prex = ""

        self.all_lines = []
        self.train_lst = []
        self.test_lst = []

    def print_func_name(self):
        def get_current_function_name():
            return inspect.stack()[1][3]
        print("%s.%s invoked"%(self.__class__.__name__, get_current_function_name()))

    def clear_dir(self, dir_path):
        assert os.path.isdir(dir_path)
        for e in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, e))

    def img_f2i(self, img):
        return ((img+1.0)*127.0).astype(np.float32)

    def img_i2f(self, img):
        return img/127.0-1.0


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


    ### get_org_image and label from a line
    def get_org_image(self, line="", margin=0.1):
        line = line.strip()
        line = line.replace('\\', '/')
        elems = line.split(' ')
        if len(elems) == 1:
            elems = line.split('\t')

        assert len(elems) == (self.pts_num * 2 + 1 + 4)
        img_path = elems[0]
        box = elems[1:5]
        landmarks = elems[5:]

        print(self.path_prex + img_path)
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

        face_img0 = img[y_new:y_new+h_new, x_new:x_new+w_new]
        face_img = cv2.resize(face_img0, (self.input_image_H, self.input_image_W), cv2.INTER_LINEAR)
        #x = face_img / 255.0 * 2.0 - 1.0
        x = face_img
        y = np.zeros((self.pts_num * 2), np.float32)
        for i in range(self.pts_num):
            y[i * 2] = (round(float(landmarks[i * 2])) - x_new) / w_new
            y[i * 2 + 1] = (round(float(landmarks[i * 2 + 1])) - y_new) / h_new

        face_rect = (x0,y0,w0,h0)
        return (img, landmarks, face_rect)


    ### get a image and label from a line
    def get_image_label(self, line="", margin=0.1, x1y1x2y2=True):
        line = line.strip()
        line = line.replace('\\', '/')
        elems = line.split(' ')
        if len(elems) == 1:
            elems = line.split('\t')

        assert len(elems) == (self.pts_num * 2 + 1 + 4)
        img_path = elems[0]
        box = elems[1:5]
        landmarks = elems[5:]

        print(self.path_prex + img_path)
        assert os.path.exists(self.path_prex + img_path)
        img = cv2.imread(self.path_prex + img_path)
        # assert img!=None

        x0 = float(box[0])
        y0 = float(box[1])

        ###### for x1,y1,x2,y2
        if x1y1x2y2:
            w0 = float(box[2])-float(box[0])
            h0 = float(box[3])-float(box[1])

        ###### for x,y,w,h
        if not x1y1x2y2:
            w0 = float(box[2])
            h0 = float(box[3])

        #margin = np.random.randint(8, 20) / 100.0
        margin = 0.1
        H, W, _ = img.shape
        x_new = int(round(max(0, x0 - w0 * margin)))
        y_new = int(round(max(0, y0 - h0 * margin)))
        w_new = int(round(min(W, x0 + w0 * (1 + margin)) - x_new))
        h_new = int(round(min(H, y0 + h0 * (1 + margin)) - y_new))

        if DEBUG:
            print(line)
            print((x_new, y_new), (x_new + w_new, y_new + h_new))
            img1 = img.copy()
            cv2.rectangle(img1, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 255, 0))
            img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            #cv2.imshow("fd", img1)
            cv2.imwrite('face.jpg', img1)

            img_org = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            # img_org = img
            cv2.imwrite('org.jpg', img_org)
            #cv2.imshow("org", img_org)
            #cv2.waitKey(0)

        face_img0 = img[y_new:y_new + h_new, x_new:x_new + w_new]
        face_img = cv2.resize(face_img0, (self.input_image_H, self.input_image_W), cv2.INTER_LINEAR)
        x = face_img
        x = x/127.0-1.0

        if DEBUG:
            cv2.imwrite("debug_face.jpg", face_img)

        y = np.zeros((self.pts_num * 2), np.float32)
        for i in range(self.pts_num):
            y[i * 2] = (round(float(landmarks[i * 2])) - x_new) / w_new
            y[i * 2 + 1] = (round(float(landmarks[i * 2 + 1])) - y_new) / h_new

        if DEBUG:
            cv2.imshow("face", face_img)
            img_tmp = img.copy()
            for i in range(self.pts_num):
                ptx = int(round(float(landmarks[i * 2])))
                pty = int(round(float(landmarks[i * 2 + 1])))
                cv2.circle(img_tmp, (ptx, pty), 2, (0, 0, 255), 2)
            img_tmp = cv2.resize(img_tmp, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("org_kp", img_tmp)

            cv2.imwrite('./tmp_face.jpg', face_img0)
            cv2.imwrite('./tmp_x.jpg', ((x + 1.0) / 2.0 * 255.0).astype(np.uint8))
            img_tmp = self.draw_keypoint(x, y)
            cv2.imwrite('./tmp_kp.jpg', img_tmp)

        return (x, y)


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
        img = x
        H,W,_ = img.shape

        for i in range(self.pts_num):
            ptx = int(round(y[i*2]*W))
            pty = int(round(y[i*2+1]*H))
            cv2.circle(img, (ptx,pty), 1, (0,0,255), 3)

        return img


    def computer_mean(self, path):
        assert os.path.exists(path)

        lst = []
        for e in open(path):
            lst.append(e)

        print(len(lst))
        BATCH_SIZE = len(lst)
        IMAGE_SIZE = self.input_image_W
        HD5Images = np.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
        HD5Landmarks = np.zeros([BATCH_SIZE, 68 * 2], dtype='float32')

        i = 0

        self.path_prex = '/home/hui89.liu/workspace1/face_landmark_dataset/300W_train_3837/'
        mean_file_name = '/home/hui89.liu/workspace1/face_landmark_dataset/300W_train_3837/mean.bin'
        img_mean = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        count = 0
        for line in lst:
            if i % 1000 == 0 or i >= BATCH_SIZE - 1:
                print "Processing row %d " % (i + 1)

            print(line)
            (x, y) = self.get_image_label(line=line, margin=0.1)

            x = x.astype(np.uint8)

            img_mean = (img_mean * count + x) / (count + 1)

        final_mean = np.transpose(img_mean, [2, 0, 1])
        final_mean = np.reshape(final_mean, [1, 3, IMAGE_SIZE, IMAGE_SIZE])

        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.num, blob.channels, blob.height, blob.width = final_mean.shape
        blob.data.extend(final_mean.astype(float).flat)
        binaryproto_file = open(mean_file_name, 'wb')
        binaryproto_file.write(blob.SerializeToString())
        binaryproto_file.close()


    def cvimg_2_caffeimg(self, cvimg):
        x = cv2.split(cvimg)
        return np.array(x)
        pass

    def caffeimg_2_cvimg(self, caffeimg):
        print(caffeimg.shape)
        assert caffeimg.shape[0] == 3
        c0 = caffeimg[0]
        c1 = caffeimg[1]
        c2 = caffeimg[2]
        x = cv2.merge( [c0, c1, c2] )
        x = np.array(x)
        return x


    def make_HDF5(self, path, shuffle=True):
        assert os.path.exists(path)

        lst = []
        for e in open(path):
            if e.find('xm2vts/') > 0:
                print(e)
                continue
            lst.append(e)

        if shuffle:
            np.random.shuffle(lst)

        print(len(lst))
        BATCH_SIZE = len(lst)
        IMAGE_SIZE = self.input_image_W
        HD5Images = np.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
        HD5Landmarks = np.zeros([BATCH_SIZE, 68*2], dtype='float32')
        HD5Landmarks_theta = np.zeros([BATCH_SIZE, 6], dtype='float32')

        i = 0

        self.path_prex = '/home/hui89.liu/workspace1/face_landmark_dataset/300W_train_3837/'
        #self.path_prex = '/home/hui89.liu/workspace1/300W_dataset/'

        for line in lst:
            if i % 1000 == 0 or i >= BATCH_SIZE - 1:
                print "Processing row %d " % (i + 1)

            print(line)
            (x,y) = self.get_image_label(line=line, margin=0.15,x1y1x2y2=False)

            #seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(scale=(0.5, 0.7))])
            #seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start

            #x_aug = seq_det.augment_images(x)
            #y_aug = seq_det.augment_keypoints(y)

            x = self.img_f2i(x)
            image = x
            image = image.astype('f4')
            HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
            HD5Landmarks[i, :] = y-0.5
            theta = self.get_theta(y)
            theta = theta.reshape((1, 6))
            #theta = np.array([1,0,0,0,1,0])
            HD5Landmarks_theta[i, :] = theta
            i += 1

        l = HD5Images.shape[0]
        train_l = int(l*0.8)
        outputPath = '/home/hui89.liu/workspace1/300W_dataset/train_448_255_0915.hdf5'
        #outputPath = '/home/hui89.liu/workspace1/face_landmark_dataset/train_448_0915.hdf5'
        with h5py.File(outputPath, 'w') as T:
            T.create_dataset("X", data=HD5Images[0:train_l,...])
            T.create_dataset("landmarks", data=HD5Landmarks[0:train_l,...])
            T.create_dataset("theta", data=HD5Landmarks_theta[0:train_l,...])

        outputPath = '/home/hui89.liu/workspace1/300W_dataset/test_448_255_0915.hdf5'
        #outputPath = '/home/hui89.liu/workspace1/face_landmark_dataset/test_448_0915.hdf5'
        with h5py.File(outputPath, 'w') as T:
            T.create_dataset("X", data=HD5Images[train_l:,...])
            T.create_dataset("landmarks", data=HD5Landmarks[train_l:,...])
            T.create_dataset("theta", data=HD5Landmarks_theta[train_l:,...])


    def get_theta(self, y):
        assert y.shape[0] == 68*2
        y0 = y
        le_idx = 36
        pt1x, pt1y = y0[le_idx*2],y0[le_idx*2+1]

        re_idx = 45
        pt2x, pt2y = y0[re_idx * 2], y0[re_idx * 2 + 1]

        nouse_idx = 33
        pt3x, pt3y = y0[nouse_idx * 2], y0[nouse_idx * 2 + 1]

        #pt1x = pt1x - pt3x
        #pt1y = pt1y - pt3y

        #pt2x = pt2x - pt3x
        #pt2y = pt2y - pt3y

        #pt3x = pt3x - pt3x
        #pt3y = pt3y - pt3y


        pt1x0 = 475.0 / 1856.0
        pt1y0 = 303.0 / 1496.0

        pt2x0 = 1412.0 / 1856.0
        pt2y0 = 289.0 / 1496.0

        pt3x0 = 972.0 / 1856.0
        pt3y0 = 700.0 / 1496.0

        #pt1x0 = pt1x0 - pt3x0
        #pt1y0 = pt1y0 - pt3y0

        #pt2x0 = pt2x0 - pt3x0
        #pt2y0 = pt2y0 - pt3y0

        #pt3x0 = pt3x0 - pt3x0
        #pt3y0 = pt3y0 - pt3y0

        src = np.array( [ [pt1x, pt1y], [pt2x, pt2y], [pt3x, pt3y] ], dtype=np.float32 )
        dst = np.array( [ [pt1x0, pt1y0], [pt2x0, pt2y0], [pt3x0, pt3y0] ], dtype=np.float32 )
        src = src - 0.5
        dst = dst - 0.5
        theta = cv2.getAffineTransform(src, dst)
        return theta



    def make_HDF5_theta(self, path):
        assert os.path.exists(path)

        lst = []
        for e in open(path):
            lst.append(e)

        print(len(lst))
        BATCH_SIZE = len(lst)
        IMAGE_SIZE = self.input_image_W
        HD5Images = np.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
        HD5Landmarks = np.zeros([BATCH_SIZE, 6], dtype='float32')

        i = 0

        self.path_prex = '/home/hui89.liu/workspace1/face_landmark_dataset/300W_train_3837/'

        for line in lst:
            if i % 1000 == 0 or i >= BATCH_SIZE - 1:
                print "Processing row %d " % (i + 1)

            print(line)
            (x,y) = self.get_image_label(line=line, margin=0.1)

            x = x.astype(np.uint8)
            #cv2.imwrite('t.jpg', x)

            #x= input("pause")

            #image = dataRow.image.astype('f4')
            #image = (image - meanTrainSet) / (1.e-6 + stdTrainSet)
            #image = np.zeros(shape=(3,12,12))
            x = x/127.0 - 1.0
            image = x
            image = image.astype('f4')
            HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
            #HD5Landmarks[i, :] = y-0.5
            theta = self.get_theta(y)
            theta = theta.reshape( (1,6) )
            HD5Landmarks[i, :] = theta

            i += 1

        outputPath = '/home/hui89.liu/workspace1/face_landmark_dataset/train_theta.hdf5'
        with h5py.File(outputPath, 'w') as T:
            T.create_dataset("X", data=HD5Images)
            T.create_dataset("theta", data=HD5Landmarks)



    def make_LMDB(self, path, shuffle=True, x1y1x2y2=False):
        assert os.path.exists(path)

        lst = []
        for e in open(path):
            if e.find('xm2vts/') > 0:
                print(e)
                continue
            lst.append(e)

        if shuffle:
            np.random.shuffle(lst)

        print(len(lst))
        BATCH_SIZE = len(lst)
        IMAGE_SIZE = self.input_image_W
        HD5Images = np.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
        HD5Landmarks = np.zeros([BATCH_SIZE, 68*2], dtype='float32')
        HD5Landmarks_theta = np.zeros([BATCH_SIZE, 6], dtype='float32')

        i = 0

        self.path_prex = '/home/hui89.liu/workspace1/face_landmark_dataset/300W_train_3837/'
        #self.path_prex = '/home/hui89.liu/workspace1/300W_dataset/'

        for line in lst:
            if i % 1000 == 0 or i >= BATCH_SIZE - 1:
                print "Processing row %d " % (i + 1)

            print(line)
            (x,y) = self.get_image_label(line=line, margin=0.1,x1y1x2y2=x1y1x2y2)

            #seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(scale=(0.5, 0.7))])
            #seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start

            #x_aug = seq_det.augment_images(x)
            #y_aug = seq_det.augment_keypoints(y)

            x = self.img_f2i(x)
            image = x
            image = image.astype('f4')
            HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
            HD5Landmarks[i, :] = y-0.5
            theta = self.get_theta(y)
            theta = theta.reshape((1, 6))
            HD5Landmarks_theta[i, :] = theta
            i += 1

        l = HD5Landmarks.shape[0]
        train_l = int(l*0.8)

        map_size = 1e12
        DB_KEY_FORMAT = "{:0>10d}"

        # train data

        data_path = '/home/hui89.liu/workspace1/300W_dataset/train_448_255_0915_data.lmdb'
        data_env = lmdb.open(data_path, map_size=map_size)
        with data_env.begin(write=True) as txn:
            key = 0
            for ii in range(0,train_l):
                datum = caffe.proto.caffe_pb2.Datum()
                xi = HD5Images[ii, ...]
                im_dat = caffe.io.array_to_datum(xi.astype(np.float32))
                #print(im_dat);input('pause!!!');
                key_str = DB_KEY_FORMAT.format(key)
                txn.put(key_str.encode('ascii'), im_dat.SerializeToString())
                key += 1
        data_env.close()


        y_path = '/home/hui89.liu/workspace1/300W_dataset/train_448_255_0915_y.lmdb'
        y_env = lmdb.open(y_path, map_size=map_size)
        with y_env.begin(write=True) as txn:
            key = 0
            for ii in range(0, train_l):
                labels = HD5Landmarks[ii, ...]
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = labels.shape[0]
                datum.height = 1
                datum.width = 1
                datum.data = labels.tostring()  # or .tobytes() if numpy < 1.9
                datum.label = 0
                key_str = DB_KEY_FORMAT.format(key)
                txn.put(key_str.encode('ascii'), datum.SerializeToString())
                key += 1
        y_env.close()

        y_theta_path = '/home/hui89.liu/workspace1/300W_dataset/train_448_255_0915_y_theta.lmdb'
        y_theta_env = lmdb.open(y_theta_path, map_size=map_size)
        with y_theta_env.begin(write=True) as txn:
            key = 0
            for ii in range(0, train_l):
                labels = HD5Landmarks_theta[ii, ...]
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = labels.shape[0]
                datum.height = 1
                datum.width = 1
                datum.data = labels.tostring()  # or .tobytes() if numpy < 1.9
                datum.label = 0
                key_str = DB_KEY_FORMAT.format(key)
                txn.put(key_str.encode('ascii'), datum.SerializeToString())
                key += 1
        y_theta_env.close()



        # test data

        data_path = '/home/hui89.liu/workspace1/300W_dataset/test_448_255_0915_data.lmdb'
        data_env = lmdb.open(data_path, map_size=map_size)
        with data_env.begin(write=True) as txn:
            key = 0
            for ii in range(l-train_l):
                datum = caffe.proto.caffe_pb2.Datum()
                xi = HD5Images[train_l+ii, ...]
                im_dat = caffe.io.array_to_datum(xi.astype(np.float32))
                key_str = DB_KEY_FORMAT.format(key)
                txn.put(key_str.encode('ascii'), im_dat.SerializeToString())
                key += 1
        data_env.close()



        y_path = '/home/hui89.liu/workspace1/300W_dataset/test_448_255_0915_y.lmdb'
        y_env = lmdb.open(y_path, map_size=map_size)
        with y_env.begin(write=True) as txn:
            key = 0
            for ii in range(l - train_l):
                labels = HD5Landmarks[train_l+ii, ...]
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = labels.shape[0]
                datum.height = 1
                datum.width = 1
                datum.data = labels.tostring()  # or .tobytes() if numpy < 1.9
                datum.label = 0
                key_str = DB_KEY_FORMAT.format(key)
                txn.put(key_str.encode('ascii'), datum.SerializeToString())
                key += 1
        y_env.close()

        y_theta_path = '/home/hui89.liu/workspace1/300W_dataset/test_448_255_0915_y_theta.lmdb'
        y_theta_env = lmdb.open(y_theta_path, map_size=map_size)
        with y_theta_env.begin(write=True) as txn:
            key = 0
            for ii in range(l - train_l):
                labels = HD5Landmarks_theta[train_l+ii, ...]
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = labels.shape[0]
                datum.height = 1
                datum.width = 1
                datum.data = labels.tostring()  # or .tobytes() if numpy < 1.9
                datum.label = 0
                key_str = DB_KEY_FORMAT.format(key)
                txn.put(key_str.encode('ascii'), datum.SerializeToString())
                key += 1
        y_theta_env.close()

        print("make LMDB finished!")


    def test_data_LMDB(self, lmdb_path):
        print(lmdb_path)
        assert os.path.exists(lmdb_path)

        map_size = 1e12
        DB_KEY_FORMAT = "{:0>10d}"

        images = []
        landmarks = []
        theta = []

        lmdb_env = lmdb.open(lmdb_path, map_size=map_size)
        with lmdb_env.begin(write=True) as txn:
            key = 0
            cursor = txn.cursor()

            ii = 0
            for key,value in cursor:

                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(value)

                x = caffe.io.datum_to_array(datum)
                images.append(x)
                #print(x.shape)
                #xx = cv2.merge([x[0], x[1], x[2]])
                #cv2.imwrite("tmp/{:04d}.jpg".format(ii), xx)

                ii += 1
                if ii > 20:
                    break

            lmdb_env = lmdb.open("/home/hui89.liu/workspace1/300W_dataset/test_448_255_0915_y.lmdb", map_size=map_size)
            with lmdb_env.begin(write=True) as txn:
                key = 0
                cursor = txn.cursor()

                ii = 0
                for key, value in cursor:

                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(value)

                    #x = caffe.io.datum_to_array(datum)
                    y = np.fromstring(datum.data, np.float32)
                    landmarks.append(y)

                    ii += 1
                    if ii > 20:
                        break

            lmdb_env = lmdb.open("/home/hui89.liu/workspace1/300W_dataset/test_448_255_0915_y_theta.lmdb", map_size=map_size)
            with lmdb_env.begin(write=True) as txn:
                key = 0
                cursor = txn.cursor()

                ii = 0
                for key, value in cursor:

                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(value)

                    #x = caffe.io.datum_to_array(datum)
                    y = np.fromstring(datum.data, np.float32)
                    theta.append(y)

                    ii += 1
                    if ii > 20:
                        break

            for i in range(len(images)):
                img = images[i]
                y = landmarks[i]
                th = theta[i]

                img = self.caffeimg_2_cvimg(img)

                r = self.draw_keypoint(img,y+0.5)
                save_fig_path = 'tmp/'
                path = save_fig_path + '{:04d}-org1.jpg'.format(i)
                cv2.imwrite(path, r)

                M = th
                M = M.reshape((2, 3))
                IMAGE_SIZE = self.input_image_W
                r0 = cv2.warpAffine(r, M, dsize=(IMAGE_SIZE, IMAGE_SIZE))
                path = './tmp/{:04d}-trans.jpg'.format(i)
                cv2.imwrite(path, r0)

            '''
            for ii in range(200):
                key_str = DB_KEY_FORMAT.format(key)
                print(key_str)
                value = cursor.get(key_str.encode('ascii'))
                print(value)
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(value)
                x = np.fromstring(datum.data, dtype=np.uint8)

                print(x.shape)

                key += 1
                cursor = cursor.next()

                if ii > 20:
                    break
            '''



    def test_data_hdf5(self, hdf5_path):
        print(hdf5_path)
        input('xxx')
        assert os.path.exists(hdf5_path)
        with h5py.File(hdf5_path, 'r') as T:
            X = T['X']
            landmarks = T['landmarks']
            theta = T['theta']

            self.clear_dir('./tmp')

            rand_nums = np.random.randint(0, X.shape[0], (100), dtype=np.int)

            for ii in range(X.shape[0]):
                i = rand_nums[ii]
                x0 = X[i]
                y0 = landmarks[i]
                theta0 = theta[i]

                x0 = ((x0+1.0)*127.0).astype(np.uint8)
                IMAGE_SIZE = self.input_image_W
                x0.resize( (3, IMAGE_SIZE, IMAGE_SIZE) )
                #x0 = x0[:,:, [1,2,0] ]
                c1 = x0[0, :]
                c2 = x0[1, :]
                c3 = x0[2, :]
                x00 = cv2.merge( [c1, c2, c3] )

                path = './tmp/{:04d}_img.jpg'.format(i)
                cv2.imwrite(path, x00)

                fa = self.draw_keypoint(x00, y0+0.5)
                path = './tmp/{:04d}_land.jpg'.format(i)
                cv2.imwrite(path, fa)

                #print(theta0)
                M = theta0
                M = M.reshape((2,3))
                x000 = cv2.warpAffine(x00,M, dsize=(IMAGE_SIZE, IMAGE_SIZE))
                path = './tmp/{:04d}-1.jpg'.format(i)
                #cv2.imwrite(path, x000)


                if ii > 20:
                    break



    def caffe_model_test(self, hdf5_path):
        b_save_fig = False
        save_fig_path = './model_result/'

        if b_save_fig:
            if not os.path.exists(save_fig_path):
                os.mkdir(save_fig_path)

        caffe.set_device(15)
        caffe.set_mode_gpu()
        caffe_prex = '/home/hui89.liu/workspace1/face_landmark1/Face_Alignment_Two_Stage_Re-initialization-master/caffe/'
        prototxt = caffe_prex + 'examples/ld_1/network_300W_global_stage_reg000-test.prototxt'
        caffemodel = caffe_prex + 'examples/ld_1/ld_1_global_stage_lr000_iter_3000.caffemodel'
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        assert os.path.exists(hdf5_path)
        with h5py.File(hdf5_path, 'r') as T:
            X = T['X']
            landmarks = T['landmarks']
            theta = T['theta']

            for iii in range(20):
                i = iii + 200
                x0 = X[i]
                y0 = landmarks[i]
                theta0 = theta[i]

                x0 = ((x0 + 1.0) * 127.0).astype(np.uint8)
                IMAGE_SIZE = self.input_image_W
                x0.resize((3, IMAGE_SIZE, IMAGE_SIZE))

                c1 = x0[0, :]
                c2 = x0[1, :]
                c3 = x0[2, :]
                x00 = cv2.merge([c1, c2, c3])
                M = theta0
                M = M.reshape((2, 3))
                x000 = cv2.warpAffine(x00, M, dsize=(IMAGE_SIZE, IMAGE_SIZE))
                path = save_fig_path + '{:04d}-org.jpg'.format(i)
                cv2.imwrite(path, x000)



                net.blobs['X'].data[...] = X[i]
                net.forward()
                pred_landmark = net.blobs['68point'].data[0]

                for ii in range(68):
                    x,y = pred_landmark[ii*2], pred_landmark[ii*2+1]
                    x += 0.5
                    y += 0.5
                    ix,iy = int(x*self.input_image_W+0.5), int(y*self.input_image_W+0.5)
                    cv2.circle(x000, (ix,iy), 2, (0,0,255))
                for ii in range(68):
                    true_landmarks = landmarks[i]
                    x,y = true_landmarks[ii*2], true_landmarks[ii*2+1]
                    x += 0.5
                    y += 0.5
                    ix,iy = int(x*self.input_image_W+0.5), int(y*self.input_image_W+0.5)
                    cv2.circle(x000, (ix,iy), 1, (255,0,0))

                path = save_fig_path + '{:04d}-pred.jpg'.format(i)
                cv2.imwrite(path, x000)


    def caffe_model_test_img_list(self, img_list):
        b_save_fig = False
        save_fig_path = './model_result/'

        if b_save_fig:
            if not os.path.exists(save_fig_path):
                os.mkdir(save_fig_path)

        caffe.set_device(15)
        caffe.set_mode_gpu()
        #caffe.set_mode_cpu()
        caffe_prex = '/home/hui89.liu/workspace1/face_landmark1/Face_Alignment_Two_Stage_Re-initialization-master/caffe/'
        #prototxt = caffe_prex + 'examples/ld_1/network_300W_global_stage_reg000-test.prototxt'
        #caffemodel = caffe_prex + 'examples/ld_1/ld_1_global_stage_lr000_iter_3000.caffemodel'

        #prototxt = caffe_prex + 'fa0913/fa0913_global_stage_reg100-fix-test.prototxt'
        #caffemodel = caffe_prex + 'fa0913/fa0913_global_stage_lr100-fix_iter_20000.caffemodel'

        prototxt = caffe_prex + 'fa0914/train_test_0914-debug.prototxt'
        caffemodel = caffe_prex + 'fa0914/0914_iter_1000.caffemodel'

        assert os.path.exists(prototxt)
        assert os.path.exists(caffemodel)

        net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        assert os.path.exists(img_list)
        i = 0
        idx = 0
        line_lst = []

        for line in open(img_list):
            line_lst.append(line)

        np.random.random(time.time())
        np.random.shuffle(line_lst)

        for line in line_lst:
            idx += 1
            if idx < 230:
                continue
            elems = line.strip().split('\t')

            name = elems[0]
            img_prex = '/home/hui89.liu/workspace1/face_landmark_dataset/300W_train_3837/'
            img_path = img_prex + name
            img_path = img_path.replace('\\', '/')
            img_path = img_path.replace('\\', '/')

            assert os.path.exists(img_path)

            x1, y1 = self.get_image_label(line)

            #ix0 = ((x1 + 1.0) * 127.0).astype(np.uint8)
            ix0 = x1
            path = save_fig_path + '{:04d}-org.jpg'.format(i)
            cv2.imwrite(path, ix0)

            fx = x1/127.0-1.0
            fx = cv2.split(fx)
            fx = np.array(fx)
            fx = fx[np.newaxis, ...]
            net.blobs['X'].data[...] = fx
            net.forward()
            pred_landmark = net.blobs['68point'].data[0]

            for ii in range(68):
                x,y = pred_landmark[ii*2], pred_landmark[ii*2+1]
                x += 0.5
                y += 0.5
                print(x, y)
                ix,iy = int(x*self.input_image_W+0.5), int(y*self.input_image_W+0.5)
                cv2.circle(ix0, (ix,iy), 2, (0,0,255))

            for ii in range(68):
                true_landmarks = np.zeros( shape=(2*self.pts_num) )
                x,y = true_landmarks[ii*2], true_landmarks[ii*2+1]
                x += 0.5
                y += 0.5
                ix,iy = int(x*self.input_image_W+0.5), int(y*self.input_image_W+0.5)
                #cv2.circle(x0, (ix,iy), 1, (255,0,0))

            path = save_fig_path + '{:04d}-pred.jpg'.format(i)
            cv2.imwrite(path, ix0)

            i += 1
            if i>= 20:
                break



    def caffe_model_test_img_list_debug(self, img_list):
        b_save_fig = False
        save_fig_path = './model_result/'

        if b_save_fig:
            if not os.path.exists(save_fig_path):
                os.mkdir(save_fig_path)

        caffe.set_device(15)
        caffe.set_mode_gpu()
        # caffe.set_mode_cpu()
        caffe_prex = '/home/hui89.liu/workspace1/face_landmark1/Face_Alignment_Two_Stage_Re-initialization-master/caffe/'

        #prototxt = caffe_prex + 'fa0914/train_test_0914-debug.prototxt'
        #caffemodel = caffe_prex + 'fa0914/0914_iter_1000.caffemodel'

        prototxt = '/home/hui89.liu/workspace1/face_landmark_dataset/300W/network_300W_parts.prototxt'
        caffemodel = '/home/hui89.liu/workspace1/face_landmark_dataset/300W/network_300W_parts.caffemodel'

        assert os.path.exists(prototxt)
        assert os.path.exists(caffemodel)

        net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        assert os.path.exists(img_list)

        line_lst = []

        for line in open(img_list):
            line_lst.append(line)

        np.random.seed()
        np.random.shuffle(line_lst)

        i = 0
        idx = 0
        for line in line_lst:
            idx += 1
            if idx < 230:
                continue
            elems = line.strip().split(' ')

            name = elems[0]
            #img_prex = '/home/hui89.liu/workspace1/face_landmark_dataset/300W_train_3837/'
            img_prex = '/home/hui89.liu/workspace1/300W_dataset/'
            img_path = img_prex + name
            img_path = img_path.replace('\\', '/')
            img_path = img_path.replace('\\', '/')

            print(img_path)
            assert os.path.exists(img_path)

            x1, y1 = self.get_image_label(line)

            # ix0 = ((x1 + 1.0) * 127.0).astype(np.uint8)
            ix0 = x1
            ix0 = cv2.resize(ix0, dsize=(0,0), fx=0.5, fy=0.5)
            path = save_fig_path + '{:04d}-org.jpg'.format(i)
            cv2.imwrite(path, self.img_f2i(ix0))

            fx = ((x1+1.0)*127.0).astype(np.uint8)
            fx = cv2.split(fx)
            fx = np.array(fx)
            fx = fx[np.newaxis, ...]
            net.blobs['data'].data[...] = fx
            net.forward()
            trans_data = net.blobs['st_data'].data[0]

            print(trans_data.shape)
            #trans_data = ((trans_data+1.0)*127.0).astype(np.uint8)

            trans_data00 = self.caffeimg_2_cvimg(trans_data)

            path = save_fig_path + '{:04d}-debug.jpg'.format(i)
            cv2.imwrite(path, trans_data00)

            theta0 = net.blobs['theta'].data[0]
            np.savetxt(save_fig_path + '{:04d}-theta.npy'.format(i), theta0)

            landmark = net.blobs['pre_label'].data[0]
            img_pred = self.draw_keypoint(self.caffeimg_2_cvimg(fx[0]), (landmark+1)/2)
            path = save_fig_path + '{:04d}-pred.jpg'.format(i)
            cv2.imwrite(path, img_pred)


            i += 1
            if i >= 20:
                break


    def affine_test(self, img_list):
        assert os.path.exists(img_list)

        save_fig_path = './model_result/'

        idx = 0
        for line in open(img_list):
            line = line.strip()
            elems = line.split(' ')
            if len(elems) < 68*2:
                elems = line.split('\t')

            img_path = self.path_prex + elems[0]
            img_path = img_path.replace('\\', '/')
            print(img_path)
            assert os.path.exists(img_path)

            (x, y) = self.get_image_label(line=line, margin=0.1)

            x0 = x / 127.0-1.0
            x0 = ((x0 + 1.0) * 127.0).astype(np.uint8)
            x00 = x0
            x00 = x00.astype(np.uint8)
            theta = self.get_theta(y)
            #theta = theta.reshape((1, 6))
            #theta = np.array([1, 0, 0, 0, 1, 0])
            path = save_fig_path + '{:04d}-org.jpg'.format(idx)
            cv2.imwrite(path, x00)

            M = theta
            M = M.reshape((2, 3))
            #IMAGE_SIZE = self.input_image_W
            #x000 = cv2.warpAffine(x00, M, dsize=(IMAGE_SIZE, IMAGE_SIZE))

            (org_img, landmarks, face_rect) = self.get_org_image(line=line)
            alhpa = 0.96
            org_img = cv2.resize(org_img, dsize=(0,0), fx=alhpa, fy=alhpa)

            x0,y0,w0,h0 = face_rect
            [x0,y0,w0,h0] = [x0*alhpa,y0*alhpa,w0*alhpa,h0*alhpa]

            H, W, _ = org_img.shape
            margin = 0.0
            x_new = int(round(max(0, x0 - w0 * margin)))
            y_new = int(round(max(0, y0 - h0 * margin)))
            w_new = int(round(min(W, x0 + w0 * (1 + margin)) - x_new))
            h_new = int(round(min(H, y0 + h0 * (1 + margin)) - y_new))

            face_img0 = org_img[y_new:y_new + h_new, x_new:x_new + w_new]

            path = save_fig_path + '{:04d}-org-face.jpg'.format(idx)
            cv2.imwrite(path, face_img0)

            path = save_fig_path + '{:04d}-org0.jpg'.format(idx)
            cv2.imwrite(path, org_img)

            ORG_H, ORG_W,_ = org_img.shape
            org_img1 = cv2.warpAffine(org_img, M, dsize=(ORG_W, ORG_H) )

            path = save_fig_path + '{:04d}-org1.jpg'.format(idx)
            cv2.imwrite(path, org_img1)

            idx += 1
            if idx > 20:
                pass
                #break


    def imgaug_test(self, img_list):
        assert os.path.exists(img_list)

        save_fig_path = './model_result/'

        idx = 0
        for line in open(img_list):
            line = line.strip()
            elems = line.split(' ')
            if len(elems) < 68*2:
                elems = line.split('\t')

            img_path = self.path_prex + elems[0]
            img_path = img_path.replace('\\', '/')
            print(img_path)
            assert os.path.exists(img_path)

            (x, y) = self.get_image_label(line=line, margin=0.1)



            seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(scale=(0.5, 0.7))])
            seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start

            npx = np.array(x)
            npy = np.array(y)
            images = npx[np.newaxis, ...]
            print(images.shape)

            keypoints_on_images = []
            keypoints = []
            for k in range(68):
                xx0 = y[k * 2]
                yy0 = y[k * 2+1]
                keypoints.append(ia.Keypoint(x=xx0, y=yy0))
            keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=x.shape))

            # augment keypoints and images
            images_aug = seq_det.augment_images(images)
            keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)

            # Example code to show each image and print the new keypoints coordinates
            for img_idx, (image_before, image_after, keypoints_before, keypoints_after) in enumerate(
                    zip(images, images_aug, keypoints_on_images, keypoints_aug)):
                image_before = keypoints_before.draw_on_image(image_before)
                image_after = keypoints_after.draw_on_image(image_after)
                #misc.imshow(np.concatenate((image_before, image_after), axis=1))  # before and after


            #y_aug.draw_on_image(x_aug1)

                path = save_fig_path + '{:04d}-org1.jpg'.format(idx)
                cv2.imwrite(path, image_after)

            idx += 1
            if idx > 20:
                pass
                #break


    def model_pred(self,test_list_path):
        assert os.path.exists(test_list_path)

        save_fig_path = './model_result/'

        self.clear_dir(save_fig_path)

        caffe.set_device(4)
        caffe.set_mode_gpu()

        #caffe.set_mode_cpu()

        caffe_prex = '/home/hui89.liu/workspace1/face_landmark1/Face_Alignment_Two_Stage_Re-initialization-master/caffe/'

        #prototxt = caffe_prex + 'fa0914/train_test_0914-debug.prototxt'
        #caffemodel = caffe_prex + 'fa0914/0914_iter_1000.caffemodel'

        prototxt = caffe_prex + 'fa0915/step0_train_test-deploy.prototxt'
        caffemodel = caffe_prex + 'fa0915/step0_iter_251.caffemodel'

        assert os.path.exists(prototxt) and os.path.exists(caffemodel)

        net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        line_lst = []

        for line in open(test_list_path):
            line_lst.append(line)

        print(len(line_lst))
        #np.random.seed()
        np.random.shuffle(line_lst)

        i = 0
        self.path_prex = "/home/hui89.liu/workspace1/300W_dataset/"
        for line in line_lst:
            line = line.strip()
            elems = line.split(' ')
            if len(elems) < 68 * 2:
                elems = line.split('\t')

            img_path = self.path_prex + elems[0]
            img_path = img_path.replace('\\', '/')
            print(img_path)
            assert os.path.exists(img_path)

            (x, y) = self.get_image_label(line=line, margin=0.1)
            path = save_fig_path + '{:04d}-org0.jpg'.format(i)
            cv2.imwrite(path, self.img_f2i(x))

            xs = cv2.split(x)
            xs = np.array(xs)
            xs = xs[np.newaxis, ...]
            xs = np.zeros_like(xs)
            net.blobs['X'].data[...] = xs
            net.forward()

            '''
            trans_face = net.blobs['st_data'].data[0]
            pred_face = self.caffeimg_2_cvimg(trans_face)
            pred_face = ((pred_face+1.0)*127.0).astype(np.uint8)
            '''

            landmark_pred = net.blobs['68point'].data[0]
            landmark_pred = landmark_pred+0.5
            pred_face = self.draw_keypoint(self.img_f2i(x), landmark_pred)

            path = save_fig_path + '{:04d}-pred0.jpg'.format(i)
            cv2.imwrite(path, pred_face)

            i += 1
            if i >= 20:
                break

        self.print_func_name()
        print(sys._getframe().f_code.co_name)


    def disp_fd(self, txt_path):
        assert os.path.exists(txt_path)

        save_fig_path = './tmp/'

        line_lst = []

        for line in open(txt_path):
            line_lst.append(line)

        print(len(line_lst))
        # np.random.seed()
        # np.random.shuffle(line_lst)

        print('start process!')
        i = 0
        for line in line_lst:
            line = line.strip()
            elems = line.split(' ')
            if len(elems) < 68 * 2:
                elems = line.split('\t')

            img_path = self.path_prex + elems[0]
            img_path = img_path.replace('\\', '/')
            print(img_path)
            assert os.path.exists(img_path)

            (x, y) = self.get_image_label(line=line, margin=0.1)

            path = save_fig_path + 'face_{:04d}.jpg'.format(i)
            cv2.imwrite(path, x)

            i += 1
            if i >= 20:
                break

        self.print_func_name()
        print(sys._getframe().f_code.co_name)


    def split_txt(self, txt_path):
        assert os.path.exists(txt_path)

        lst = []
        for line in open(txt_path):
            lst.append(line)

        np.random.seed()
        np.random.shuffle(lst)

        l = len(lst)
        l_train = int(l*0.8)

        f_train = open(txt_path[0:-4] + '_train' + txt_path[-4:], 'w')
        for line in lst[0:l_train]:
            f_train.write(line)
        f_train.close()

        f_val = open(txt_path[0:-4] + '_val' + txt_path[-4:], 'w')
        for line in lst[l_train:]:
            f_val.write(line)
        f_val.close()


    def split_txt_as_list(self, txt_path):
        assert os.path.exists(txt_path)

        lst = []
        for line in open(txt_path):
            lst.append(line)

        np.random.seed()
        np.random.shuffle(lst)

        l = len(lst)
        l_train = int(l*0.8)

        f_train = open(txt_path[0:-4] + '_trainList' + txt_path[-4:], 'w')
        for line in lst[0:l_train]:
            f_train.write(line.split(' ')[0] + ' 0\n')
        f_train.close()

        f_val = open(txt_path[0:-4] + '_valList' + txt_path[-4:], 'w')
        for line in lst[l_train:]:
            f_val.write(line.split(' ')[0] + ' 0\n')
        f_val.close()


    def test(self, path):
        assert os.path.exists(path)



if __name__ == "__main__":
    print('main')

    face_reader = FaceLandmarkDataReader()
    #face_reader.path_prex = '/home/hui89.liu/workspace1/face_landmark_dataset/300W_train_3837/'

    # create train val list
    #face_reader.split_txt_as_list('/home/hui89.liu/workspace1/300W_dataset/300W_trainset_GT_list.txt')


    #txt_path = "/home/hui89.liu/workspace1/face_landmark_dataset/300w_train_3148.txt"
    txt_path = "/home/hui89.liu/workspace1/300W_dataset/300W_trainset_OD_list.txt"
    face_reader.input_image_W = 448
    face_reader.input_image_H = 448
    #face_reader.make_LMDB(txt_path, x1y1x2y2 = True)
    #face_reader.clear_dir('./tmp')
    #face_reader.test_data_LMDB("/home/hui89.liu/workspace1/300W_dataset/test_448_255_0915_data.lmdb")
    #face_reader.make_HDF5(txt_path)
    #face_reader.affine_test(txt_path)

    #face_reader.make_HDF5_theta(txt_path)
    #face_reader.make_LMDB(txt_path)

    #hdf5_path = '/home/hui89.liu/workspace1/face_landmark_dataset/test_448_0915.hdf5'
    #hdf5_path = '/home/hui89.liu/workspace1/300W_dataset/OD_test_448_0915.hdf5'
    #face_reader.test_data_hdf5(hdf5_path)
    #face_reader.caffe_model_test(hdf5_path)

    #img_list = '/home/hui89.liu/workspace1/face_landmark_dataset/test_689.txt'
    #face_reader.caffe_model_test_img_list_debug(img_list)

    #txt_path = "/home/hui89.liu/workspace1/300W_dataset/300W_trainset_OD_list.txt"
    #face_reader.imgaug_test(txt_path)

    #txt_path = "/home/hui89.liu/workspace1/300W_dataset/300W_testset_full_OD_list.txt"
    #txt_path = "./300W_test_OD_list.txt"
    #face_reader.model_pred(txt_path)
    #face_reader.disp_fd(txt_path)



    #################
    txt_path = "/home/hui89.liu/workspace1/300W_dataset/300W_testset_full_OD_list.txt"
    face_reader.path_prex = "/home/hui89.liu/workspace1/300W_dataset/"
    face_reader.caffe_model_test_img_list_debug(txt_path)

