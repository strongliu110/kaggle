from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

def get_pretrained_model(arch, input_shape):
    if arch == 'vgg16':
        model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'vgg19':
        model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'resnet50':
        model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'InceptionV3':
        model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise Exception("Not supported architecture: {}".format(arch))

    x = model.output

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)

    return model

def basic_model(input_shape, filters=64, kernel=3):
    model = Sequential()

    model.add(Conv2D(filters=filters, kernel_size=(kernel, kernel), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=filters, kernel_size=(kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=filters * 2, kernel_size=(kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=filters * 2, kernel_size=(kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(filters=filters * 4, kernel_size=(kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=filters * 4, kernel_size=(kernel, kernel), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation="softmax"))

    return model