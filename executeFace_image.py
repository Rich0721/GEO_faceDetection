import cv2
from keras.models import load_model
import keras.backend as K
import os
import numpy as np
from MTCNN.mtcnn import MTCNN
import argparse

class faceDetection(object):

    def __init__(self, image=None, mtcnn_npy="./data/mtcnn_weights.npy", hdf5_file="./models/resnet50_fusion.hdf5", nb_classes_folder="./dataset/test"):

        self._mtcnn_detector = MTCNN()
        self._image = image
        self._face_recognition = load_model(hdf5_file)
        self._nb_classes = ['Albert', 'Amanda', 'Amber', 'Amy', 'AmyYT', 'Andy Li', 'AnnieHsu', 'Ansley', 'BellaHong', 'Ben', 
                            'Betty', 'BettyHung', 'Calvin', 'Chi', 'Clare', 'Danny', 'Eirk', 'Epic', 'Eunice', 'Eve', 'FishYang', 
                            'Frank', 'Gavin', 'George', 'Gogan', 'Ivy', 'Jane', 'Jerry', 'Jet', 'Joe', 'JoeHsu', 'Joy', 'JoyChen', 
                            'Kim', 'Lucy', 'Maggie', 'Mandy', 'Mark', 'Marsh', 'Mountain', 'Nicole', 'Night', 'Ning', 'Penny_Wu', 
                            'Phil', 'Polly', 'RayTsai', 'Samael', 'SamChen', 'Sandy', 'SharonWang', 'Shawn', 'Sherie', 'Shirley_Wu', 
                            'Sunny', 'Ted', 'Tim', 'Wei', 'WendyCheng', 'WendyDu', 'Willian', 'Yoyo', 'Zoezhang', 'Zona']#os.listdir(nb_classes_folder)

    def recognition(self):

        frame = cv2.imread(self._image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faceboxes = self._mtcnn_detector.detect_faces(frame)
        
        if len(faceboxes) == 1:
            for f in faceboxes:
                
                if f[0] < 0:
                    f[0] = 0
                elif  f[1] < 0:
                    f[1] = 0 
                elif f[2] > 960:
                    f[2] = 960
                elif f[3] > 540:
                    f[3] = 540
                

            face = frame[faceboxes[0][1]:faceboxes[0][3], faceboxes[0][0]:faceboxes[0][2]]
            if len(face) != 0:
                face = cv2.resize(face, (224, 224))
                face = face / 255
                face = np.expand_dims(face, axis=0)

                index = self._face_recognition.predict(face)
                list_index = np.ndarray.tolist(index[0])

                if max(list_index) < 0.5:
                    name = "unkown"
                else:
                    ind = list_index.index(max(list_index))
                    name = self._nb_classes[ind]
                cv2.rectangle(frame, (faceboxes[0][0], faceboxes[0][1]), (faceboxes[0][2], faceboxes[0][3]), (0, 255, 0), thickness=1)
                cv2.putText(frame, name, (faceboxes[0][0] - 10, faceboxes[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self._image[:-4]+"_result.jpg", frame)



def addParset():

    parser = argparse.ArgumentParser()
    parser.add_argument("-image", help="recognition face", type=str, default=None, dest="image")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = addParset()

    t = faceDetection(image=args.image)
    t.recognition()
   

            
