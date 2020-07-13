import PIL
from PIL import Image,ImageTk
#import pytesseract
import cv2
from tkinter import *
import time
import datetime
import tkinter as tk
import sys
#import pandas as pd
#import xlsxwriter
import tkinter.messagebox as messagebox
import csv
from keras.models import load_model
import keras.backend as K
import os
import numpy as np
from mtcnn import MTCNN
from PIL import Image, ImageTk
from time import sleep

class MyVideoCapture(object):
    
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source,cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
            
        # Set video source width and height
        self._width = 960
        self._height = 540

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)



    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()

            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class GUI(object):

    def __init__(self, window, window_title, video_source=0):

        self.baseline_rectangle = [368, 158, 592, 382]
        self.tempInput=36.0
        self.time = datetime.datetime.now().strftime("%Y/%m/%d %A\n%H:%M:%S\n\n")
        self.name = "unknown"
        self.detector = MTCNN()
        self.model = load_model("./resnet50_fusion.hdf5")
        self.eyes_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")
        self.nb_classes = nb_classes =  ['Albert','Amanda','Amber', 'Amy', 'AmyYT', 'Andy Li', 'AnnieHsu', 'Ansley', 'BellaHong', 'Ben', 
                            'Betty', 'BettyHung', 'Calvin', 'Chi', 'Clare', 'Danny', 'Eirk', 'Epic', 'Eunice', 'Eve', 'FishYang', 
                            'Frank', 'Gavin', 'George', 'Gogan', 'Ivy', 'Jane', 'Jerry', 'Jet', 'Joe', 'JoeHsu', 'Joy', 'JoyChen', 
                            'Kim', 'Lucy', 'Maggie', 'Mandy', 'Mark', 'Marsh', 'Mountain', 'Nicole', 'Night', 'Ning', 'Penny_Wu', 
                            'Phil', 'Polly', 'RayTsai', 'Samael', 'SamChen', 'Sandy', 'SharonWang', 'Shawn', 'Sherie', 'Shirley_Wu', 
                            'Sunny', 'Ted', 'Tim', 'Wei', 'WendyCheng', 'WendyDu', 'Willian', 'Yoyo', 'Zoezhang', 'Zona']
        self.people_image_folder = "./staff_image"
        self.gui_image_folder = "./ICON"
        self.count = 0
        self.rectangle_image = cv2.resize(cv2.imread(os.path.join(self.gui_image_folder, "identify.png")), (224, 224))
        self.rectangle_image = cv2.cvtColor(self.rectangle_image, cv2.COLOR_BGR2RGB)
        self.window = window
        self.window_title = window_title
        self.video_source = video_source
        
    
    def GUIset(self):
        self.window.geometry("960x1100")
        self.window.title(self.window_title)
        

        # open video source
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
        
        templogo_image = Image.open(os.path.join(self.gui_image_folder, "thermometer.png")).resize((45, 45))
        Templogo = ImageTk.PhotoImage(templogo_image)
        label_image = tk.Label(self.window, image=Templogo)
        label_image.place(x=100, y=637)

        temp_disply_frame = Image.open(os.path.join(self.gui_image_folder, "thermometer_display.png")).resize((193, 47))
        temp_disply_frame_image = ImageTk.PhotoImage(temp_disply_frame)
        self.label_temp=Label(self.window, font='Helvetica 21 bold', image=temp_disply_frame_image, compound="center", fg="#00BFFF")
        self.label_temp.config(text=str(self.tempInput)+"°C")
        self.label_temp.place(x=200,y=634)
        
        add_image = Image.open(os.path.join(self.gui_image_folder, "button_up.png")).resize((32, 32))
        btn_add_image = ImageTk.PhotoImage(add_image)
        self.add_button=Button(self.window, image=btn_add_image, command=self.add)
        self.add_button.place(x=398,y=604)

        sub_image = Image.open(os.path.join(self.gui_image_folder, "button_down.png")).resize((32, 32))
        btn_sub_image = ImageTk.PhotoImage(sub_image)
        self.sub_button=Button(self.window, image=btn_sub_image ,command=self.sub)
        self.sub_button.place(x=398,y=662)
        
        time_image = Image.open(os.path.join(self.gui_image_folder, "time.png")).resize((268, 84))
        time_image_label = ImageTk.PhotoImage(time_image)
        self.time_label = Label(self.window, font='Helvetica 15 bold', image=time_image_label, compound="center", justify="center")
        self.time_label.config(text=time)
        self.time_label.place(x=100, y=800)
        
        staff_frame_image = Image.open(os.path.join(self.gui_image_folder, "photo_frame.png")).resize((256, 256))
        staff_frame_image_tk = ImageTk.PhotoImage(staff_frame_image)
        self.staff_frame_label = tk.Label(self.window, image=staff_frame_image_tk)
        self.staff_frame_label.place(x=500, y=604)

        staff_inti_image = Image.open(os.path.join(self.gui_image_folder, "unknown.png")).resize((160,160))
        staff_inti_image_tk = ImageTk.PhotoImage(staff_inti_image)
        
        self.staff_image_label = tk.Label(self.window, image=staff_inti_image_tk)
        self.staff_image_label.place(x=540, y=640)

        self.staff_name_label = tk.Label(self.window, text="Name: unknown", font='Helvetica 15 bold')
        self.staff_name_label.place(x=500, y=870)

        check_button_image = Image.open(os.path.join(self.gui_image_folder, "check.png")).resize((96, 50))
        check_button_image_tk = ImageTk.PhotoImage(check_button_image)
        self.check_button = Button(self.window, image=check_button_image_tk, command=self.check)
        self.check_button.place(x=250, y=720)
        
        self.delay = 1
        self.update()
        self.clock()
        
        self.window.mainloop()
    
    def update(self):
        
        ret, frame = self.vid.get_frame()
        #v2.transpose()
        #frame = cv2.flip(frame, -1)
        frame = np.rot90(frame)
        self.coverRetangle(frame, self.rectangle_image)
        
        if ret:
            
            ISOTIMEFORMAT = '%S,%f'
            nowtime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            target = nowtime[3]
            target2 =nowtime[4]
            
            x = frame[self.baseline_rectangle[1]:self.baseline_rectangle[3], self.baseline_rectangle[0]:self.baseline_rectangle[2]]
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            eyes = self.eyes_cascade.detectMultiScale(x)
            if len(eyes)>0 and target=='5'and int(target2) >=5 :
                boxes = self.detector.detect_faces(frame)
            else:
                boxes = []
            
            for box in boxes:
                iou = self.IoU(box)

                if iou:
                    face_image = frame[box[1]:box[3], box[0]:box[2]]
                    face_image = cv2.resize(face_image, (224, 224))
                    face_image = face_image / 255.
                    face_image = np.expand_dims(face_image, axis=0)

                    predict = self.model.predict(face_image)
                    predict_list = np.ndarray.tolist(predict[0])
                    
                    if max(predict_list) < 0.5:
                        self.name = 'unknown'
                        file_name = os.path.join(self.gui_image_folder, "unknown.png")

                    else:
                        index = predict_list.index(max(predict_list))
                        self.name = self.nb_classes[index]
                        file_name = os.path.join(self.people_image_folder, self.name + ".png")

                    #image = Image.open(file_name).resize((160, 160))
                    
                    self.PhotoShow(imageFile=file_name)
                    break
                else:
                    self.name = 'unknown'
                    file_name = os.path.join(self.gui_image_folder, "unknown.png")
                    self.PhotoShow(imageFile=file_name)
                    
            
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
        self.window.after(self.delay, self.update)
    

    def IoU(self, box):

        x1 = np.maximum(box[0], self.baseline_rectangle[0])
        y1 = np.maximum(box[1], self.baseline_rectangle[1])
        x2 = np.minimum(box[2], self.baseline_rectangle[2])
        y2 = np.minimum(box[3], self.baseline_rectangle[3])

        interArea = (x2 - x1 + 1) * (y2 - y1 + 1) # compute the area of intersection rectangle

        # Compute the area of box and baseline
        baseline_area = (self.baseline_rectangle[2] - self.baseline_rectangle[0] + 1) * (self.baseline_rectangle[3] - self.baseline_rectangle[1] + 1)
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

        # Compute the intersection over union
        iou = interArea / float(baseline_area + box_area - interArea)
        
        if iou > 0.3:
            return True
        else:
            return False
            
    def add(self):
        self.tempInput+=0.1
        self.label_temp.config(text=str(round(self.tempInput,2))+"°C")
    

    def sub(self):
        self.tempInput-=0.1
        self.label_temp.config(text=str(round(self.tempInput,2))+"°C")
     
    def clock(self):
        
        self.time = datetime.datetime.now().strftime("%Y/%m/%d %A\n%H:%M:%S")
        self.time_label.config(text=self.time, font='Helvetica 15 bold')
        
        self.window.after(1000, self.clock) # run itself again after 1000 ms
    
    def check(self):
        if self.name != 'unknown':
            message = Toplevel()
            message.title("簽到完成")
            Message(message, text="姓名:" +self.name +"\n" +  "時間:" + self.time + "\n" + "體溫:"+self.label_temp.cget("text"),padx=20, pady=20).pack()
            message.after(2000, message.destroy)
            self.name = "unknown"
            self.PhotoShow(imageFile=os.path.join(self.gui_image_folder, "unknown.png"))
    
    def coverRetangle(self, frame ,retangleImage):
        
        for h in range(158, 382):
            for w in range(368, 592):
                retangle_height = 224 - (382 - h)
                retangle_width = 224 - (592 - w)
                if retangleImage[retangle_height][retangle_width][0] != 255 and retangleImage[retangle_height][retangle_width][1] != 255 and retangleImage[retangle_height][retangle_width][2] != 255:
                    frame[h][w] = retangleImage[retangle_height][retangle_width]
        
        return frame

    def PhotoShow(self, imageFile=None):
        print(imageFile)
        image = Image.open(imageFile).resize((160, 160))
        self.predict_staff_image = ImageTk.PhotoImage(image)
        self.staff_image_label.config(image=self.predict_staff_image)
        self.staff_name_label.config(text='Name: ')
        self.staff_name_label.config(text="Name: " + self.name)
        

if __name__ == "__main__":
    gui = GUI(tk.Tk(), "TEST")
    gui.GUIset()

    