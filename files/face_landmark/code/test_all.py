import caffe
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import config
import ordinal_func as ofunc
import DLDL_func as dfunc

prototxt = '/home/tianchu.guo/caffe/y_caffe_master_20160223_aug/data/android_gaze_tcguo/ordinal_DLDL_CASIA_v1/deploy.prototxt'
caffemodel = '/home/tianchu.guo/caffe/y_caffe_master_20160223_aug/data/android_gaze_tcguo/ordinal_DLDL_CASIA_v1/ordinal_DLDL_CASIA_7.1854_.caffemodel'
face_list_file = '/home/tianchu.guo/data/android_data_gaze/tc_guo_data/list/0831_onefolder_data_list/test_0/face.txt'
data_path = '/home/tianchu.guo/data/android_data_gaze/crop_data/'
save_fig_path = '/home/tianchu.guo/data/android_data_gaze/tc_guo_code/save_fig/ordinal_DLDL_test_0'

b_save_fig = False
b_del = True

if b_save_fig:
    if not os.path.exists(save_fig_path):
        os.mkdir(save_fig_path)

mean_blob = caffe.proto.caffe_pb2.BlobProto()
face_mean_name = '/home/tianchu.guo/data/android_data_gaze/tc_guo_data/list/0824_data_list/train_0/face_mean_72.binaryproto'
mean_blob.ParseFromString(open(face_mean_name, 'rb').read())
face_mean_npy = caffe.io.blobproto_to_array(mean_blob)

left_mean_name = '/home/tianchu.guo/data/android_data_gaze/tc_guo_data/list/0824_data_list/train_0/left_eye_0828_w72h56.binaryproto'
mean_blob.ParseFromString(open(left_mean_name, 'rb').read())
left_mean_npy = caffe.io.blobproto_to_array(mean_blob)

right_mean_name = '/home/tianchu.guo/data/android_data_gaze/tc_guo_data/list/0824_data_list/train_0/right_eye_0828_w72h56.binaryproto'
mean_blob.ParseFromString(open(right_mean_name, 'rb').read())
right_mean_npy = caffe.io.blobproto_to_array(mean_blob)


ordinal_bin_size = config.ordinal_bin_size_0828
X_RANGE = config.X_RANGE
Y_RANGE = config.Y_RANGE * 2.0 / 3

caffe.set_device(15)
caffe.set_mode_gpu()
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

err_ordinal = 0.0
err_posterior = 0.0
err_ordinal_x = 0.0
err_ordinal_y = 0.0 
err_DLDL_x = 0.0
err_DLDL_y = 0.0
count = 0
with open(face_list_file,'r') as f:
    for line in f.readlines():
        tokens = line.strip().split('\t')
        sub_tokens = tokens[0].split('/')
        
        face_name = os.path.join(data_path,tokens[0])
        left_name = os.path.join(data_path,sub_tokens[0],'left_eye_0828',sub_tokens[2])
        right_name = os.path.join(data_path,sub_tokens[0],'right_eye_0828',sub_tokens[2])
        
        gt_tokens = sub_tokens[2].split('_')
        gt_x = float(gt_tokens[-2])
        gt_y = float(gt_tokens[-1][:-4])
        
        face_img = cv2.imread(face_name)
        face_img = cv2.resize(face_img,(72,72))
        face_array = np.array(face_img,dtype = np.float32)
        face_array = np.transpose(face_array, [2,0,1])
        face_array = np.reshape(face_array,[1,3,72,72]) - face_mean_npy
        face_array = face_array[:,:,4:-4,4:-4]

        
        left_img = cv2.imread(left_name)
        left_img = cv2.resize(left_img,(72,56))
        left_array = np.array(left_img,dtype = np.float32)
        left_array = np.transpose(left_array, [2,0,1])
        left_array = np.reshape(left_array,[1,3,56,72]) - left_mean_npy
        left_array = left_array[:,:,4:-4,4:-4]
        
        right_img = cv2.imread(right_name)
        right_img = cv2.resize(right_img,(72,56))
        right_array = np.array(right_img,dtype = np.float32)
        right_array = np.transpose(right_array, [2,0,1])
        right_array = np.reshape(right_array,[1,3,56,72]) - right_mean_npy
        right_array = right_array[:,:,4:-4,4:-4]
        
        net.blobs['data_face'].data[...] = face_array
        net.blobs['left_eye'].data[...] = left_array
        net.blobs['right_eye'].data[...] = right_array
        
        net.forward()
        
        ordinal_x = net.blobs['ordinal_x'].data[0]
        ordinal_y = net.blobs['ordinal_y'].data[0]
        if b_del:
            ordinal_x[:8] = 1.0
            ordinal_x[-7:] = 0.0
            ordinal_y[:8] = 1.0
            
        predict_x = ofunc.compute_result(ordinal_x, ordinal_bin_size, 0.5)      
        predict_y = ofunc.compute_result(ordinal_y, ordinal_bin_size, 0.5)
        
        prob_x = net.blobs['p_x_posterior'].data[0]
        prob_y = net.blobs['p_y_posterior'].data[0]
        
        mean_x,sigma_value = dfunc.fit_gaussian(np.arange(len(prob_x)), prob_x)
        fit_x = dfunc.gaussian_prob(np.arange(len(prob_x)), mean_x, sigma_value)
        mean_y,sigma_value = dfunc.fit_gaussian(np.arange(len(prob_y)), prob_y)
        fit_y = dfunc.gaussian_prob(np.arange(len(prob_y)), mean_y, sigma_value)
        
#         p_x = dfunc.compute_E(fit_x, X_RANGE, ordinal_bin_size)
#         p_y = dfunc.compute_E(fit_y,Y_RANGE,ordinal_bin_size)

        p_x = mean_x * ordinal_bin_size
        p_y = mean_y * ordinal_bin_size
        
        
        err = np.sqrt(np.power(predict_x-gt_x,2) + np.power(predict_y-gt_y,2))
        err_ordinal += err
        err = np.sqrt(np.power(p_x-gt_x,2) + np.power(p_y-gt_y,2))
        err_posterior += err
        
        err_ordinal_x += np.abs(predict_x-gt_x)
        err_ordinal_y += np.abs(predict_y-gt_y)
        err_DLDL_x += np.abs(p_x-gt_x)
        err_DLDL_y += np.abs(p_y-gt_y)
        
        count += 1
        
        if count % 1000 == 0:
            print count
            
        if b_save_fig:
            tmp = sub_tokens[0]+'_' + sub_tokens[2][:-4] + '_x.png'
            save_fig_name = os.path.join(save_fig_path,tmp)
            plt.figure(1)
            plt.plot(np.arange(len(ordinal_x)),ordinal_x)
            plt.plot(np.arange(len(prob_x)),prob_x*(1.0/np.max(prob_x)))
            plt.scatter([predict_x/ordinal_bin_size],np.array([1]),marker = 'x')
            plt.scatter([p_x/ordinal_bin_size],np.array([1]),marker = '+')
            plt.scatter([gt_x/ordinal_bin_size],np.array([1]),marker = 'o')
             
             
            plt.savefig(save_fig_name)
            plt.close(1)
             
            tmp = sub_tokens[0]+'_' + sub_tokens[2][:-4] + '_y.png'
            save_fig_name = os.path.join(save_fig_path,tmp)
            plt.figure(1)
            plt.plot(np.arange(len(ordinal_y)),ordinal_y)
            plt.plot(np.arange(len(prob_y)),prob_y*(1.0/np.max(prob_y)))
             
            plt.scatter([predict_y/ordinal_bin_size],np.array([1]),marker = 'x')
            plt.scatter([p_y/ordinal_bin_size],np.array([1]),marker = '+')
            plt.scatter([gt_y/ordinal_bin_size],np.array([1]),marker = 'o')
             
            plt.savefig(save_fig_name)
            plt.close(1)
        
print 'ordinal_err = ',err_ordinal/count
print 'posterior_err = ',err_posterior/count
print 'ordinal_err_x = ',err_ordinal_x/count
print 'ordinal_err_y = ',err_ordinal_y/count
print 'DLDL_err_x = ',err_DLDL_x/count
print 'DLDL_err_y = ',err_DLDL_y/count