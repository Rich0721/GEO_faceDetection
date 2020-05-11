from __future__ import absolute_import

import tkinter as tk
import tkinter.messagebox
import cv2
import os
from time import sleep
from time import time
import math
import numpy as np
##################################################
staff= "./staff"
global next_step
next_step = False
##################################################

def checkAVI(name, result):
    global next_step
    avifile = os.path.join(staff, name+".avi")
    if os.path.exists(avifile):
        result.configure(text="有相同檔案")
    else:
        result.configure(text="可以開始錄影")
        next_step = True
    
def opencamera(cap, name, camera_label, result_label):
    global next_step
    if not os.path.exists(os.path.join(staff, name+".avi")):
        if next_step:
            
            opened = True
            while opened:
                total_time = 0
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(os.path.join(staff, name+".avi"), fourcc, 30, (1920, 1080))
                for i in range(5, 0, -1):
                    ret, frame = cap.read()
                    cv2.putText(frame, str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        total_time = 61
                        break
                    sleep(1)
                cv2.destroyAllWindows()

                start_time = time()
                
                while total_time <= 60:
                    ret, frame = cap.read()

                    out.write(frame)
                    frame = cv2.putText(frame, str(math.floor(total_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        out.release()
                        cv2.destroyAllWindows()
                        os.remove(os.path.join(staff, name + ".avi"))
                        break
                    total_time = time() - start_time
                    print(total_time)
                cv2.destroyAllWindows()
                opened = False
            next_step = False
            result_label.configure(text="完成路影")
            camera_label.configure(text="可以進行下一位錄影")
    else:
        camera_label.configure(text="無法啟動相機")
    


def graph_user_interface():


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    windows = tk.Tk()
    windows.title("拍照小程式")
    windows.geometry("800x600")
    windows.configure(background="white")

    name_frame = tk.Frame(windows)
    name_frame.pack(side=tk.TOP)
    name_label = tk.Label(name_frame, text="名字")
    name_label.pack(side=tk.LEFT)
    name_entry = tk.Entry(name_frame)
    name_entry.pack(side=tk.LEFT)

    result_label = tk.Label(windows)
    result_label.pack()

    name_butoon = tk.Button(name_frame, text="確認", command=lambda: checkAVI(name=name_entry.get(), result=result_label))
    name_butoon.pack(side=tk.LEFT)

    camera_frame = tk.Frame(windows)
    camera_frame.pack(side=tk.TOP)
    camera_butoon = tk.Button(camera_frame, text="拍照", command=lambda: opencamera(cap=cap, name=name_entry.get(), camera_label=camera_label, result_label=result_label))
    camera_butoon.pack()
    camera_label = tk.Label(camera_frame)
    camera_label.pack()

    
    windows.mainloop()

if __name__ == "__main__":
    graph_user_interface()