from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from scripts.data_generator import train_generator
from scripts.data_generator import val_generator


class Model:

    def __init__(self, model):
        self.model = model

    def fit(self, train_df, val_df, batch_size=32, epochs=1):
        train_gen = train_generator()
        train_gen.fit(train_df['Input'])
        generator = train_gen.flow(train_df['Input'], train_df['Label'], batch_size=batch_size)

        val_gen = val_generator()
        val_gen.fit(val_df['Input'])
        validation_generator = val_gen.flow(val_df['Input'], val_df['Label'], batch_size=batch_size)

        # learning rate reduction
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        # checkpoints
        filepath = "./weights.best_{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        filepath = "./weights.{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint_all = ModelCheckpoint(filepath, monitor='val_acc',
                                         verbose=1, save_best_only=False, mode='max')
        # all callbacks
        callbacks_list = [checkpoint, learning_rate_reduction, checkpoint_all]

        history = self.model.fit_generator(
            generator=generator,
            steps_per_epoch=train_df.shape[0] / batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks_list)

        return history

    def predict(self, test_df):
        self.model.predict(test_df)

