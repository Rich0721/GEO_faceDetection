import os
from glob import glob
import shutil
import numpy as np
from argparse import ArgumentParser

class classification(object):

    def __init__(self, src_folder:str, des_folder:str, train_number : int=None, class_type:int=1):
        '''
            src_folder: origination folder
            des_folder: destination folder
            train_number: train images numbers.
            class_type: 1) mask and no mask images storage same folders.
                        2) only no mask images.
                        3) only mask images. 
                        4) mask and no mask images storage different folders
        '''
        
        
        self._src_folder = src_folder
        self._des_folder = des_folder
        self._class_type = class_type
        self._folders = []
        folders = os.listdir(self._src_folder)
       
        if class_type == 2:
            for i, f in enumerate(folders):
                if i % 2 == 0:
                    self._folders.append(f)
        elif class_type == 3:
            for i, f in enumerate(folders):
                if i % 2 == 1:
                    self._folders.append(f)
        elif class_type == 1 or class_type == 4:
            self._folders = folders
        else:
            TypeError("Class type need 1~4.")
        
        self._train_folder = os.path.join(self._des_folder, "train")
        self._mkdir(self._train_folder)
        self._test_folder = os.path.join(self._des_folder, "test")
        self._mkdir(self._test_folder)
        self._veri_folder = os.path.join(self._des_folder, "veri")
        self._mkdir(self._veri_folder)
        

        if train_number is None:
            self._train_numbers = 200
        else:
            self._train_numbers = train_number
    
    @property
    def src_folders(self):
        return self._src_folder

    @property
    def des_folders(self):
        return self._des_folder
    
    def classify(self):
        
        if self._class_type == 1:
            for i, f in enumerate(self._folders):
                
                if i % 2 == 0:


                    name = f
                    storage_train_folder = os.path.join(self._train_folder, f)
                    self._mkdir(storage_train_folder)
                    storage_test_folder = os.path.join(self._test_folder, f)
                    self._mkdir(storage_test_folder)
                    storage_veri_folder = os.path.join(self._veri_folder, f)
                    self._mkdir(storage_veri_folder)

                    train_numbers = self._train_numbers
                    veri_numbers = int(self._train_numbers/4)
                    half_train = int(self._train_numbers / 2) - 1
                    half_veri = int(veri_numbers / 2) - 1
                    j = 0
                images = glob(os.path.join(self._src_folder, f, "*.jpg"))
                
                for image in images:
                    choice = np.random.randint(0, 3)
                    if i % 2 == 0:
                        if choice == 0 and train_numbers >= half_train:
                            shutil.copy(image, os.path.join(storage_train_folder, name + "_" + str(j).zfill(4) + ".jpg"))
                            train_numbers -= 1
                        elif choice == 1 and veri_numbers >= half_veri:
                            shutil.copy(image, os.path.join(storage_veri_folder, name + "_" + str(j).zfill(4) + ".jpg"))
                            veri_numbers -= 1
                        elif choice == 2:
                            shutil.copy(image, os.path.join(storage_test_folder, name + "_" + str(j).zfill(4) + ".jpg"))
                    else:

                        if choice == 0 and train_numbers > 0:
                            shutil.copy(image, os.path.join(storage_train_folder, name + "_" + str(j).zfill(4) + ".jpg"))
                            train_numbers -= 1
                        elif choice == 1 and veri_numbers > 0:
                            shutil.copy(image, os.path.join(storage_veri_folder, name + "_" + str(j).zfill(4) + ".jpg"))
                            veri_numbers -= 1
                        elif choice == 2:
                            shutil.copy(image, os.path.join(storage_test_folder, name + "_" + str(j).zfill(4) + ".jpg"))
                    j+=1
                print("{} finish!".format(f))
        else:

             for i, f in enumerate(self._folders):
                train_numbers = self._train_numbers
                veri_numbers = int(self._train_numbers/4)
                name = f
                storage_train_folder = os.path.join(self._train_folder, f)
                self._mkdir(storage_train_folder)
                storage_test_folder = os.path.join(self._test_folder, f)
                self._mkdir(storage_test_folder)
                storage_veri_folder = os.path.join(self._veri_folder, f)
                self._mkdir(storage_veri_folder)
              
                j = 0
                images = glob(os.path.join(self._src_folder, f, "*.jpg"))
                for image in images:
                    choice = np.random.randint(0, 3)

                    if choice == 0 and train_numbers > 0:
                        shutil.copy(image, os.path.join(storage_train_folder, name + "_" + str(j).zfill(4) + ".jpg"))
                        train_numbers -= 1
                    elif choice == 1 and veri_numbers > 0:
                        shutil.copy(image, os.path.join(storage_veri_folder, name + "_" + str(j).zfill(4) + ".jpg"))
                        veri_numbers -= 1
                    elif choice == 2:
                        shutil.copy(image, os.path.join(storage_test_folder, name + "_" + str(j).zfill(4) + ".jpg"))

                    j += 1
                        

    def _mkdir(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)


def createParser():       
    parser = ArgumentParser()

    parser.add_argument("-s", type=str, help="src_folder location", default='./facephoto')
    parser.add_argument("-d", type=str, help="destination folder", default="./dataset")
    parser.add_argument("-n", type=int, help="crop train data numbers", default=200)
    parser.add_argument("-t", type=int, help="1:mask and no mask images storage same folders.\n\
                                            2: only no mask images.\n\
                                            3: only mask images.\n \
                                            4: mask and no mask images storage different folders.")
    return parser.parse_args()

if __name__ == "__main__":
    parser = createParser()

    c = classification(src_folder=parser.s, des_folder=parser.d, train_number=parser.n, class_type=parser.t)
    c.classify()
    
