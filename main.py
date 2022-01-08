import cv2
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import keras
image=cv2.imread("./adem.jpeg")
image=cv2.resize(image,dsize=(600,600))

ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("start")
rects=ss.process()

proposals=[]
boxes=[]
output=image.copy()
print(rects.shape)
for (x,y,w,h) in rects[:500]:
    color=[random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output,(x,y),(x+w,y+h),color,2)
    roi=image[y:y+h,x:x+w]
    roi=cv2.resize(roi,dsize=(224,224),interpolation=cv2.INTER_LANCZOS4)
    proposals.append(roi)
    boxes.append((x,y,w+x,h+y))
proposals=np.array(proposals,dtype="float64")
boxes=np.array(boxes,dtype="int32")

model=keras.models.load_model("first_model")

proba=model.predict(proposals)

number_list=[]

idx=[]

for i in range(len(proba)):
    max_prob=np.max(proba[i,:])
    
    if max_prob >=0.9999990:
        print(max_prob)
        idx.append(i)
        number_list.append(np.argmax(proba[i]))
for i in range(len(number_list)):
    j=idx[i]
    cv2.rectangle(image,(boxes[j,0],boxes[j,1],boxes[j,2],boxes[j,3]),(0,0,255),2)
    cv2.putText(image,str(np.argmax(proba[j])),(boxes[j,0]+5,boxes[j,1]+5),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),1)
cv2.imshow("Image",image)
cv2.waitKey(0)