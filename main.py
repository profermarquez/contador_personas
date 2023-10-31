import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone


model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        #print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture(0)


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
persondown={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]
count2=0#cantidad personas
cy1=194
cy2=220
offset=6
list=[]
while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

    count += 1
    if count % 6!= 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
   
    for index,row in px.iterrows():
#        #print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:
            #print([x1,y1,x2,y2])
            count2=count2 +1
            list.append([x1,y1,x2,y2])
       
        
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        #print(id)
        cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
        

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        print(len(list))
        print(list)
        break
cap.release()
cv2.destroyAllWindows()

