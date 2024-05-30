#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:43:21 2020

@author: - Papa

Usando alternativa free pro Sift: ORB
Essa função carrega os dados obtidos do rastreamento do iou e usando orb 
calcula a distância em pixels dos veículos

"""

import pickle
import numpy as np
import math
import cv2


def save_vector(file, output):
    f = open(output, 'wb')
    pickle.dump(file, f)
    f.close()


def load_data(addr_data):
    f = open(addr_data, 'rb')
    data = pickle.load(f)
    f.close()
    
    return data

class speed_data(object):
    def __init__(self):
        self.speeds = []
        self.positions = []
        self.frame_init = 0
        self.frame_end = 0
        
    def add_speed(self, value):
        self.speeds.append(value)
        
    def add_position(self, points):
        self.positions.append(points)
        
    def add_frame_init(self, frame_init):
        self.frame_init = frame_init
        
    def add_frame_end(self, frame_end):
        self.frame_end = frame_end
        
    def get_average_speed(self):
        return np.average(self.speeds)
        

class vehicle_data(object):
    def __init__(self, vehicle_id, current_speed, current_point, current_frame):
        self.vehicle_id = vehicle_id
        self.exited = True
        self.data_speeds = []
        self._added_new_instance = False

        self.instance_speed = speed_data()
        self.instance_speed.add_frame_init(current_frame)
        self.instance_speed.add_speed(current_speed)
        self.instance_speed.add_position(current_point)
        
        
    def create_new_instace_speed(self, current_speed, current_point, current_frame):
        self.instance_speed = speed_data()
        self.instance_speed.add_frame_init(current_frame)
        self.instance_speed.add_speed(current_speed)
        self.instance_speed.add_position(current_point)
        self.exited = True

    def add_current_instace_speed(self, current_speed, current_point):
        self.instance_speed.add_speed(current_speed)
        self.instance_speed.add_position(current_point)
        
    def add_instance_frame_init(self, current_frame):
        self.instance_speed.add_frame_init(current_frame)
        
    def add_instance_frame_end(self, current_frame):
        self.instance_speed.add_frame_end(current_frame)

    def get_id(self):
        return self.vehicle_id  

    def get_average_speed(self):
        return self.data_speeds[len(self.data_speeds)-1].get_average_speed()
    

    def end(self, current_frame):
        self.data_speeds.append(self.instance_speed)
        self.instance_speed.add_frame_end(current_frame)
        self.exited = False
        self._added = True


addr_video = './samples/v_0_4.mp4'
addr_data = './samples/v_0_4.pckl'

#addr_video = './samples/v_5_6.mp4'
#addr_data = './samples/v_5_6.pckl'
data = load_data(addr_data)
    
ids = []
cap = cv2.VideoCapture(addr_video)
   
#print(' ======== [INFO] ========= ')
print('[INFO] Video: ', addr_video)
#frame_n1 =  {'img_boxes_1':[]}
key = 0
cont_frame = 0

def get_total_instances():
    cont = 0
    for instance in data:
        cont+=len(instance.data_speeds)
    print('Total instances = ', cont)

get_total_instances()
while cap.isOpened() and key != 113:  # for 'q' key
    ret, img = cap.read()
    if ret == True:

        #lane 1
        cv2.line(img, (200, 94), (1550, 94), (0, 0, 255), 2)
        cv2.line(img, (1700, 316), (60, 316), (0, 0, 255), 2)
        
        
        
        #text start and end
        cv2.rectangle(img, (500, 700), (886, 780), (0,204,0), -1)
        
        #Rectangles
        cv2.rectangle(img, (500, 800), (1400, 930), (255,0,0), -1)

        cv2.putText(img,'Real Speed = ',(620,840),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255,255,255),
                    thickness=2) 
        #cv2.putText(img,'Predict speed = ',(620,880),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,255,255),thickness=2) 
        
        
        
        for vehicle in data:
            for instance_data in vehicle.data_speeds:
                if instance_data.frame_init == cont_frame:
                    cv2.putText(img,'start estimate', (530,750),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (255,255,255),
                        thickness=2)
                    #points = instance_data.positions[0]
                    #cv2.circle(img, points[2], radius=0, color=(0, 0, 255), thickness=15)
                    
                if instance_data.frame_end == cont_frame:
                    cv2.putText(img,'Real Speed = ',(620,840),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255,255,255),
                        thickness=2) 
                    cv2.putText(img,str(instance_data.get_average_speed()),(1000,840),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255,255,255),
                        thickness=2) 
                    
                    cv2.putText(img,'End estimate', (530,750),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (255,255,255),
                        thickness=2)  
                    
                    #cv2.rectangle(img, (x_plate,y_plate), ((x2_plate), (y2_plate)),(int(cl[0]), int(cl[1]), int(cl[2])), 3) 
                    
                    #points = instance_data.positions[len(instance_data.positions)-1]
                    #cv2.circle(img, points[2], radius=0, color=(0, 0, 255), thickness=15)

        resized_image = cv2.resize(img, (1280, 720)) 
        cv2.imshow("frame", resized_image)
        
        key = cv2.waitKey(0)
        cont_frame+=1
        #print(self.cont_frame)
    else:
        break

cap.release()
cv2.destroyAllWindows()
    
print('[INFO] End ')
       
            