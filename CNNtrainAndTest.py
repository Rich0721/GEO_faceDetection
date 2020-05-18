from __future__ import absolute_import
import os
import numpy as np
from glob import glob
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Input, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from net.vgg import VGG16
from keras_preprocessing.image import img_to_array
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
import cv2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import Callback, EarlyStopping
from tensorflow.python.keras.models import load_model
from keras_preprocessing.image import img_to_array
import 

class Train(object):

    def __init__(self, train_folder, vali_folder, net_type, weights_file, storage_file):
        
        self._nb_classes, self._train_genrator, self._vali_generator = self._dataGenerator(train_folder, vali_folder)
        self._net_type = net_type
        self._weights_file = weights_file
        self._storage_file = storage_file

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
        class_mode='sparse'
        )

        vail_generator = vali_datagen.flow_from_directory(
        vali_folder,
        target_size=(224, 224),
        batch_size=8,
        class_mode='sparse'
        )

        return nb_class, train_generator, vail_generator

    def _vgg16(self):


        vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights_file=self._weights_file)
        last_layer = vgg_model.get_layer("pool5").output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(1024, activation='relu', name='fc6')(x)
        x = Dropout(0.2)(x)
        x = Dense(2048, activation='relu', name='fc7')(x)
        x = Dropout(0.3)(x)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        custom_vgg_model = Model(vgg_model.input, out)
        custom_vgg_model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // 8
        validation_steps =  self._vali_generator.n // 8
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
        custom_vgg_model.fit_generator(
        self._train_genrator,
        steps_per_epoch=steps_per_epoch,
        epochs=300,
        validation_data=self._vali_generator,
        validation_steps=validation_steps,
        callbacks=[monitor])


        custom_vgg_model.save(self._storage_file)


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
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                index = self._model.predict(img)
                list_predict = np.ndarray.tolist(index[0])

                max_index = list_predict.index(max(list_predict))
                if self._nb_classes[max_index] == p:
                    acc += 1
                    all_acc += 1
                all_total += 1
                total += 1
            
            self._class_acc[p] = round(acc/ total, 2)
            