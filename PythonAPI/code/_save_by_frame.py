#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:39:55 2020

@author: - Papa
"""
from PIL import Image
import pickle
import numpy as np
import cv2

  
def save_detections(tracks, addr_detec):
    f = open(addr_detec, 'wb')
    pickle.dump(tracks, f)
    f.close()
    
    '''
    detections = util.load_mot(tracks)
    
    #Fazer rastreamento e salvar bounding boxes resultantes
    sigma_h = 0.6
    sigma_iou = 0.5
    t_min = 5

    res_tracker = iou_tracker.track_iou(detections, sigma_h, sigma_iou, t_min)
    util.save_track(addr_track, res_tracker)
    '''
    

class run():
    def __init__(self, addr, output, numFrameToSave):
        cont_frame = 0
            
        if 'svo' in addr:
            import pyzed.sl as sl
            init = sl.InitParameters(svo_input_filename=addr, svo_real_time_mode=False)
               
            cam = sl.Camera()
            status = cam.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
 
            runtime = sl.RuntimeParameters()
            mat = sl.Mat()
            resolution_video = (1280,720)

            running = True
            while running:  # for 'q' key
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat, sl.VIEW.VIEW_RIGHT_UNRECTIFIED)
                    #cam.retrieve_measure(depth_map, sl.MEASURE.MEASURE_DEPTH)
                    im = Image.fromarray(mat.get_data()) #Convert to PIL format
                    im = im.convert('RGB')
                    
                    if (cont_frame % numFrameToSave == 0):
                        img = np.array(im) 
                        resized_image = cv2.resize(img, resolution_video)
                        cv2.imwrite(output+str(cont_frame)+'.jpg', resized_image)
                        
                    print('Total Frame = ', cam.get_svo_number_of_frames())
                    print('frame = ', cont_frame)

                    cont_frame+=1
                else:
                    running = False
            cam.close()
        else:
            cap = cv2.VideoCapture(addr)
            cont_frame = 0

            while(cap.isOpened()): 
                
                ret, frame = cap.read()
                if ret == True:
                    print('frame = ', cont_frame)
                    
                    if (cont_frame % numFrameToSave == 0):
                        cv2.imwrite(output+str(cont_frame)+'.jpg', frame)

              
                    cont_frame+=1
                else:
                    break
            cap.release()

                    
addr = '../v_0_5.mp4'   
run(addr, './samples/temp/', 1)       
                    
    