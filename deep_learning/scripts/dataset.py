import numpy as np
import pandas as pd

import cv2

from scripts.data_load import *

class DataSet(object):

    def __init__(self):
        self.train_datas = None
        self.train_labels = None

        self.val_datas = None
        self.val_labels = None

        # self.test_datas = []
        # self.test_labels = []

    def load_kfold_train_val(self, data_df, input_shape):
        train_index, test_index = get_kfold_train_val_idx(data_df)

        train_datas = []
        train_labels = []
        for idx in train_index:
            path = data_df.loc[idx, 'Path']
            image = cv2.resize(cv2.imread(path), (input_shape[0], input_shape[1]))
            label = data_df.loc[idx, 'EncodeLabel']
            train_datas.append(image)  # ravel:返回引用，flatten:返回拷贝
            train_labels.append(label)
        self.train_datas = np.array(train_datas)
        self.train_labels = np.array(train_labels)

        val_datas = []
        val_labels = []
        for idx in test_index:
            path = data_df.loc[idx, 'Path']
            image = cv2.resize(cv2.imread(path), (input_shape[0], input_shape[1]))
            label = data_df.loc[idx, 'EncodeLabel']
            val_datas.append(image)  # ravel:返回引用，flatten:返回拷贝
            val_labels.append(label)
        self.val_datas = np.array(val_datas)
        self.val_labels = np.array(val_labels)
