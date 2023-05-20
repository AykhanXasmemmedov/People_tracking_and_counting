
import cv2
import numpy as np

import time
import random
import math

from sort import *
import humandetect

def intersection_over_union(real,predict):
    x0=max(real[0],predict[0])
    y0=max(real[1],predict[1])
    x1=min(real[2],predict[2])
    y1=min(real[3],predict[3])
    
    interArea=max(0,x1-x0+1)*max(0,y1-y0+1)
    
    realArea=(real[2]-real[0]+1)*(real[3]-real[1]+1)
    predictArea=(predict[2]-predict[0]+1)*(predict[3]-predict[1]+1)
    
    iou=interArea/float(realArea+predictArea-interArea)
    
    return iou

tracker=Sort(max_age=30,min_hits=5,iou_threshold=.3)
local_tracks=[]

### testing

video=cv2.VideoCapture('lv_0_20230421011412.mp4')

mask_first=np.zeros((220,320,3),dtype='uint8')
mask_first[:,:,0]=255

mask_second=np.zeros((220,320,3),dtype='uint8')
mask_second[:,:,1]=255

mask_third=np.zeros((220,310,3),dtype='uint8')
mask_third[:,:,2]=255

colors=[]
for i in range(100):
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    colors.append([b,g,r])
    
    
count_first=0
count_second=0
count_third=0

counted_ID_first=[]
counted_ID_second=[]
counted_ID_third=[]


prev_time = 0
new_time = 0

while video.isOpened():
    
    ret,frame=video.read()
    if not ret:
        break
    
    frame=cv2.resize(frame,(1280, 720))
    coordinates=humandetect.detect_human(frame)
    
    if coordinates:
        track_bBox_ids=tracker.update(np.asarray(coordinates))
        
        d_value=[]
        
        for track in reversed(track_bBox_ids):
            if local_tracks[int(track[4])-1:int(track[4])]:
                
                pr_x=local_tracks[int(track[4])-1][1][-1][0]
                pr_y=local_tracks[int(track[4])-1][1][-1][1]
        
                now_x=(track[0]+track[2])/2
                now_y=(track[1]+track[3])/2 
                d=math.sqrt((now_x - pr_x) ** 2 + (now_y - pr_y) ** 2)
                
                if d>100:
                    continue
                d_value.append(d)
                local_tracks[int(track[4]) - 1][1].append((now_x, now_y))
                
            else:
                local_tracks.append([int(track[4]) - 1, [((track[0] + track[2]) / 2, (track[1] + track[3]) / 2)]])
            
            color_id=int(int(track[4])%100)
            color=colors[color_id]
                
            cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), color, 2)
    
            roi_first=intersection_over_union([180,500,500,720],[int(track[0]), int(track[1]), int(track[2]), int(track[3])])
            roi_first=int(roi_first*100)  
            
            roi_second=intersection_over_union([550,500,870,720],[int(track[0]), int(track[1]), int(track[2]), int(track[3])])
            roi_second=int(roi_second*100)  
            
            roi_third=intersection_over_union([910,500,1220,720],[int(track[0]), int(track[1]), int(track[2]), int(track[3])])
            roi_third=int(roi_third*100) 
            
            
            if roi_first>5 and track[4] not in counted_ID_first:
                count_first=count_first+1
                counted_ID_first.append(track[4])
            
            if roi_second>5 and track[4] not in counted_ID_second:
                count_second=count_second+1
                counted_ID_second.append(track[4])
                
            if roi_third>5 and track[4] not in counted_ID_third:
                count_third=count_third+1
                counted_ID_third.append(track[4])
                
    cv2.putText(frame, f'first stair:{str(count_first)}', (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'second stair:{str(count_second)}', (50, 135),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'third stair:{str(count_third)}', (50, 170),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    coupled_img_first=cv2.addWeighted(frame[500:720,180:500,:],.6,mask_first,.4,0)
    frame[500:720,180:500,:]=coupled_img_first
    
    coupled_img_second=cv2.addWeighted(frame[500:720,550:870,:],.6,mask_second,.4,0)
    frame[500:720,550:870,:]=coupled_img_second
    
    coupled_img_third=cv2.addWeighted(frame[500:720,910:1220,:],.6,mask_third,.4,0)
    frame[500:720,910:1220,:]=coupled_img_third
    
    new_time = time.time()
    fps = int(1/(new_time-prev_time))
    cv2.putText(frame, f'FPS:{str(fps)}', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    prev_time = new_time
    
    cv2.imshow('frame', frame)
    key = cv2.waitKey(7)
    if key == ord('q'):
        break
    elif key==ord('w'):
        cv2.waitKey(0)

cv2.destroyAllWindows()
video.release()


