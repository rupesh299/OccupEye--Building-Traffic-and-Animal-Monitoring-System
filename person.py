# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 22:27:45 2023

@author: 5593
"""

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        #print(colorsBGR)
        

cv2.namedWindow('person')
cv2.setMouseCallback('person', RGB)

cap=cv2.VideoCapture('person_test_04.mov')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = total_frames / fps

output_video = cv2.VideoWriter('output3.avi', fourcc, fps/2, (frame_width,frame_height))

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
print(class_list)
count=0

tracker=Tracker()
counter_in=[]
counter_out=[]
cy1=370
cy2=410
offset=6
down={}
up={}
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(frame_width,frame_height))
   

    results=model.predict(frame)
    #print(results)
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
    #print(px)
    list=[]
             
    for index,row in px.iterrows():
        #print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        #list.append([x1,y1,x2,y2])
        if 'person' in c:
            list.append([x1,y1,x2,y2])
            
            
            
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        if  (cy+offset) >cy1 and cy1 >(cy-offset):
            down[id]=cy
        if id in down:
            if  (cy+offset) >cy2 and cy2 >(cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                cv2.putText(frame,str(id)+" IN",(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),1)
                if counter_in.count(id)==0:
                    counter_in.append(id)
                    
                output_video.write(frame)
        
        if  (cy+offset) >cy2 and cy2 >(cy-offset):
            up[id]=cy
        if id in up:
            if  (cy+offset) >cy1 and cy1 >(cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),1)
                if counter_out.count(id)==0:
                    counter_out.append(id)
                output_video.write(frame)


    cv2.line(frame,(70,cy1),(390,cy1),(0,0,0),2)
    cv2.putText(frame,('Line 1'),(49,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),1)
    cv2.line(frame,(70,cy2),(390,cy2),(0,0,0),2)
    cv2.putText(frame,('Line 2'),(49,cy2),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),1)
    #print(counter_in,counter_out)
    cv2.putText(frame,('Persons In: ')+str(len(counter_in)),(60,45),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),1)
    cv2.putText(frame,('Persons Out: ')+str(len(counter_out)),(60,65),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),1)
    cv2.putText(frame,('Total Persons In The Building: ')+str(len(counter_in)-len(counter_out)),(60,90),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)
    cv2.imshow("Person", frame)
    output_video.write(frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

