from keras.preprocessing.image import ImageDataGenerator


def train_generator():
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    return datagen


def val_generator():
    datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    return datagen
