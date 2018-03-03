import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from glob import glob
import os
import cv2


class Label:
    def __init__(self):
        self.le = preprocessing.LabelEncoder()

    def encode(self, train_label):
        self.le.fit(train_label[0])
        # print("Classes: " + str(le.classes_))
        encode_train_labels = self.le.transform(train_label[0])

        clear_train_label = to_categorical(encode_train_labels)
        num_clases = clear_train_label.shape[1]
        # print("Number of classes: " + str(num_clases))

        return clear_train_label

    def decode(self, hot_code):
        num = np.argmax(hot_code, axis=1)
        return list(self.le.inverse_transform[num])


def get_data_df(basepath):
    img_names = []
    img_labels = []
    img_paths = []
    img_sizes = []
    for path in glob(basepath + "*"):
        label = path.split("/")[-1]
        label_dir = os.path.join(basepath, label)
        for img_name in os.listdir(label_dir):
            img_names.append(img_name)
            img_labels.append(label)
            img_path = os.path.join(label_dir, img_name)
            img_paths.append(img_path)
            img_sizes.append(cv2.imread(img_path).shape)

    data_df = pd.DataFrame({'Name': img_names,
                            'Path': img_paths,
                            'Label': clear_train_label,
                            'Width': list(map(lambda x: x[1], img_sizes)),
                            'Height': list(map(lambda x: x[0], img_sizes))})
    data_df = data_df[['Name', 'Path', 'Width', 'Height', 'Label']]
    data_df.sort_values(by=['Name', 'Label'], inplace=True)
    return data_df


def get_test_df(basepath):
    img_paths = []
    for name in glob(basepath + "*"):
        img_path = os.path.join(basepath, name)
        img_paths.append(img_path)

    data_df = pd.DataFrame({'Path': img_paths})
    return data_df


def get_train_val_idx(df, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2)
    for train_index, test_index in skf.split(df, df['Label']):
        return train_index, test_index


def get_train_val_df(data_dir, k=10):
    data_df = get_data_df(data_dir)
    train_index, val_index = get_train_val_idx(data_df, k)
    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]
    return train_df, val_df
