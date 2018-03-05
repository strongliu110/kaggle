#!/usr/bin/env python

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
import math

from data_load import get_kfold_train_val_idx, load_train_val_df
from data_generator import train_generator, val_generator


class Model(object):

    def __init__(self, model, model_path=''):
        self.model = model
        self.model_path = model_path

    def __register_callbacks(self):
        # learning rate reduction
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        # checkpoints
        best_path = os.path.join(self.model_path, "weights.best_{epoch:02d}-{val_acc:.2f}.hdf5")
        checkpoint = ModelCheckpoint(best_path, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        weights_path = os.path.join(self.model_path, "weights.{epoch:02d}-{val_acc:.2f}.hdf5")
        checkpoint_all = ModelCheckpoint(weights_path, monitor='val_acc',
                                         verbose=1, save_best_only=False, mode='max')
        # all callbacks
        return [checkpoint, learning_rate_reduction, checkpoint_all]

    @staticmethod
    def __generator_batch_data(data_df, select_idxs, input_shape, batch_size=32):
        np.random.shuffle(select_idxs)  # 重洗
        loop_count = range(int(math.ceil(len(select_idxs) / batch_size)))  # 向上取整
        idx_batches = [select_idxs[range(batch_size * i, min(len(select_idxs), batch_size * (i + 1)))] for i in loop_count]

        while True:
            for idx_batch in idx_batches:
                xx, yy = load_train_val_df(data_df, idx_batch, input_shape)

                yield (xx, yy)

    @staticmethod
    def __generator_multiple_batch_data(generator, data_df, select_idxs, input_shape, batch_size=32):
        np.random.shuffle(select_idxs)  # 重洗
        loop_count = range(int(math.ceil(len(select_idxs) / batch_size)))  # 向上取整
        idx_batches = [select_idxs[range(batch_size * i, min(len(select_idxs), batch_size * (i + 1)))] for i in
                       loop_count]

        while True:
            for idx_batch in idx_batches:
                xx, yy = load_train_val_df(data_df, idx_batch, input_shape)
                gen = generator.flow(xx, batch_size=batch_size)
                xx_gen = gen.next()

                yield (xx_gen, yy)

    def fit(self, data_df, input_shape, batch_size=32, epochs=1):
        # 分割数据集
        train_indexs, val_indexs = get_kfold_train_val_idx(data_df)

        # steps
        steps = len(train_indexs) / batch_size
        val_steps = len(val_indexs) / batch_size

        history = self.model.fit_generator(
            generator=self.__generator_batch_data(data_df, train_indexs, input_shape),
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=self.__generator_batch_data(data_df, val_indexs, input_shape),
            validation_steps=val_steps,
            callbacks=self.__register_callbacks())

        return history

    def fit_multiple(self, data_df, input_shape, batch_size=32, epochs=1):
        # 分割数据集
        train_indexs, val_indexs = get_kfold_train_val_idx(data_df)

        # steps
        steps = len(train_indexs) / batch_size
        val_steps = len(val_indexs) / batch_size

        # generator
        train_gen = train_generator()
        val_gen = val_generator()

        history = self.model.fit_generator(
            generator=self.__generator_multiple_batch_data(train_gen, data_df, train_indexs, input_shape),
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=self.__generator_multiple_batch_data(val_gen, data_df, val_indexs, input_shape),
            validation_steps=val_steps,
            callbacks=self.__register_callbacks())

        return history

    def load_weights(self, file_path):
        self.model.load_weights(file_path)
        print("load weights success")

    def predict(self, test_datas, batch_size=32):
        pred_Y = self.model.predict(test_datas, batch_size=batch_size)
        print("predict success")
        return pred_Y

