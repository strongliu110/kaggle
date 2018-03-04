#!/usr/bin/env python

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os

from data_generator import train_generator, val_generator


class Model(object):

    def __init__(self, model, model_path=''):
        self.model = model
        self.model_path = model_path

    def fit(self, dataset, batch_size=32, epochs=1):
        # train_generator
        train_gen = train_generator()
        train_gen.fit(dataset.train_datas)
        generator = train_gen.flow(dataset.train_datas, dataset.train_labels, batch_size=batch_size)

        # val_generator
        val_gen = val_generator()
        val_gen.fit(dataset.val_datas)
        validation_generator = val_gen.flow(dataset.val_datas, dataset.val_labels, batch_size=batch_size)

        # steps
        steps = len(dataset.train_labels) / batch_size

        # learning rate reduction
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        # checkpoints
        filepath = os.path.join(self.model_path, "weights.best_{epoch:02d}-{val_acc:.2f}.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        filepath = os.path.join(self.model_path, "weights.{epoch:02d}-{val_acc:.2f}.hdf5")
        checkpoint_all = ModelCheckpoint(filepath, monitor='val_acc',
                                         verbose=1, save_best_only=False, mode='max')
        # all callbacks
        callbacks_list = [checkpoint, learning_rate_reduction, checkpoint_all]

        history = self.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks_list)

        return history

    def load_weights(self, file_path):
        self.model.load_weights(file_path)
        print("load weights success")

    def predict(self, test_datas, batch_size=32):
        pred_Y = self.model.predict(test_datas, batch_size=batch_size)
        print("predict success")
        return pred_Y

