import os
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Input, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from network import VGG16
from keras_preprocessing.image import img_to_array
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
import cv2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import Callback, EarlyStopping
import argparse

from network import VGG16

def train(weight_file=None, storage_file="./model/vgg16.h5"):

    IMAGE_TRAIN = "./staff/train_mask"
    IMAGE_VAIL = "./staff/test_mask"

    nb_class = len(os.listdir(IMAGE_TRAIN))

    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

    vali_datagen = ImageDataGenerator(
    rescale=1./255,
    )

    train_generator = train_datagen.flow_from_directory(
    IMAGE_TRAIN,
    target_size=(224, 224),
    batch_size=8,
    class_mode='sparse'
    )

    vail_generator = vali_datagen.flow_from_directory(
    IMAGE_VAIL,
    target_size=(224, 224),
    batch_size=8,
    class_mode='sparse'
    )

    vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights_file=weight_file)
    last_layer = vgg_model.get_layer("pool5").output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(1024, activation='relu', name='fc6')(x)
    x = Dropout(0.2)(x)
    x = Dense(2048, activation='relu', name='fc7')(x)
    x = Dropout(0.3)(x)
    out = Dense(nb_class, activation='softmax', name='fc8')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    custom_vgg_model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=SGD(lr=1e-4, momentum=0.9),
                        metrics=['accuracy'])

    steps_per_epoch =  train_generator.n // 8
    validation_steps =  vail_generator.n // 8
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto', restore_best_weights=True)
    custom_vgg_model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=300,
    validation_data=vail_generator,
    validation_steps=validation_steps,
    callbacks=[monitor])


    custom_vgg_model.save(storage_file)


def parserInit():

    parser = argparse.ArgumentParser()

    parser.add_argument("-w", help="use weights file", default=None)
    parser.add_argument("-s", help="storage h5 file name", default="./model/vgg16.h5")

    args = parser.parse_args()

    return  args


if __name__ == "__main__":
    
    args = parserInit()

    train(weight_file=args.w, storage_file=args.s)