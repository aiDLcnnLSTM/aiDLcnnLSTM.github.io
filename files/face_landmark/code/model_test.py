import os,sys
import numpy as np
from DataRow import ErrorAcum


import caffe



b_save_fig = False



save_fig_path = './model_result/'

if b_save_fig:
    if not os.path.exists(save_fig_path):
        os.mkdir(save_fig_path)


