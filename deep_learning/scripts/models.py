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

from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam


def predefined_model(arch, input_shape, num_classes):
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

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)

    return model


def basic_model(input_shape, num_classes, filters=64, kernel=3):
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

    model.add(Dense(num_classes, activation="softmax"))

    return model


def get_compile_model(arch, input_shape, num_classes, opt="adam"):
    if arch == 'basic':
        model = basic_model(input_shape, num_classes)
    else:
        model = predefined_model(arch, input_shape, num_classes)

    print("select arch: {}".format(arch))
    model.summary()

    # 数据稀疏时，推荐采用自适应算法：RMSprop，Adam. lr:学习率, epsilon:防止除0错误
    if opt == "sgd":
        optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)  # 用时长，可能困于鞍点
    elif opt == "rmsProp":
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)  # RNN效果好
    elif opt == "adam":
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    else:
        raise Exception("Not supported opt: {}".format(opt))

    print("select opt: {}".format(arch))
    print(optimizer.get_config())

    if num_classes == 2:
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
