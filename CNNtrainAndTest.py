import cv2
import os
import warnings
import numpy as np
from glob import glob
import argparse

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Reshape, LeakyReLU
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
from keras import layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils import layer_utils

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


from net.resnet import resnet50


class Train(object):

    def __init__(self, train_folder, vali_folder, weights_file, storage_file, batch_size=8, min_delta=1e-3, patience=3):
        
        self._nb_classes, self._train_genrator, self._vali_generator = self._dataGenerator(train_folder, vali_folder)
        
        self._weights_file = weights_file
        self._storage_file = storage_file
        self._batch_size = batch_size

        self._checkpoint = ModelCheckpoint(storage_file, monitor='val_acc', verbose=1, save_best_only=True)
        self._monitor = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='auto', restore_best_weights=True)

    def _dataGenerator(self, train_folder, vali_folder):

        nb_class = len(os.listdir(train_folder))

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
        train_folder,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical'
        )

        vail_generator = vali_datagen.flow_from_directory(
        vali_folder,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical'
        )

        return nb_class, train_generator, vail_generator

    def Resnet50(self, epochs=100):
        model = resnet50(include_top=False, weight_file=self._weights_file, input_shape=(224, 224, 3))

        last_layer = model.get_layer("avg_pool").output
        x = Flatten(name='flatten')(last_layer)
        x = Dropout(0.3)(x)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        train_model = Model(model.input, out)
        train_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // self._batch_size
        validation_steps =  self._vali_generator.n // self._batch_size

        callbacks =[self._checkpoint, self._monitor]
        
        train_model.fit_generator(
            self._train_genrator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._vali_generator,
            validation_steps=validation_steps,
            callbacks=callbacks)
        


class Test(object):

    def __init__(self, test_folder, model_file):
        self._test_folder = test_folder
        self._nb_classes = os.listdir(test_folder)
        self._model = load_model(model_file)
        self._class_acc = {}
    
    @property
    def class_acc(self):
        return self._class_acc
    
    def verification(self):
        
        all_acc = 0
        all_total = 0
        for p in self._nb_classes:

            images = glob(os.path.join(self._test_folder, p, "*.jpg"))
            acc = 0
            total = 0

            for image in images:

                img = cv2.imread(image)
                img = img[:, :, [2, 1, 0]]
                img = img / 255
                img = np.expand_dims(img, axis=0)

                index = self._model.predict(img)
                list_predict = np.ndarray.tolist(index[0])

                max_index = list_predict.index(max(list_predict))
                #print("{}:{}".format(p, self._nb_classes[max_index]))
                if self._nb_classes[max_index] == p:
                    acc += 1
                    all_acc += 1
                all_total += 1
                total += 1
            print("{} verified!".format(p))
            self._class_acc[p] = round(acc/ total, 2)

    def printResult(self):
        acc = 0
        
        for p in self._nb_classes:
            print("{}: {}".format(p, self._class_acc[p]))
            acc += self._class_acc[p]

        return (acc/len(self._nb_classes))



def addParset():

    parser = argparse.ArgumentParser()
    parser.add_argument("-load", help="load trained weightfile", type=str, default=None, dest="load")
    parser.add_argument("-storage", help="storage weightfile", type=str, default="./models/resenet50.hdf5", dest="storage")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    args = addParset()

    weights_file = args.load
    storage_file = args.storage
    if weights_file is not None:
        weights_extension = os.path.splitext(weights_file)[-1]
        storage_extension = os.path.splitext(storage_file)[-1]

        if (weights_extension == '.hdf5' or weights_extension == '.h5') or \
            (storage_extension == '.hdf5' or storage_extension == '.h5'):
                K.clear_session()
                train = Train(train_folder="./dataset/train", vali_folder="./dataset/veri", weights_file=weights_file, storage_file=storage_file)
                train.Resnet50()
    else:
        storage_extension = os.path.splitext(storage_file)[-1]
        if (storage_extension == '.hdf5' or storage_extension == '.h5'):
            K.clear_session()
            train = Train(train_folder="./dataset/train", vali_folder="./dataset/veri", weights_file=None, storage_file=storage_file)
            train.Resnet50()
        
    

    